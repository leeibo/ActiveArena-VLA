"""Planner-OFT dataloaders for Astribot LeRobot data.

This module intentionally lives beside the existing dataloaders instead of
modifying them.  It implements two related sampling modes:

- planner_vlm: fixed-stride history capped by ``history.max_frames`` plus the
  current frame, with retrieval indices supervised from subtask keyframes.
- planner_oft_memory: fixed-stride history without a max-frame cap, reduced to
  subtask-memory frames plus the current frame.
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import transformers
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.vlm_datasets import (
    DataCollatorForSupervisedDataset,
    _add_qwen_position_ids,
    _as_bool,
    _qwen_user_content_end_index,
    _select_lerobot_instruction,
    get_rope_index_2,
    get_rope_index_25,
    get_rope_index_3,
    preprocess_qwen_messages,
    update_processor_pixels,
)

logger = logging.getLogger(__name__)


PLANNER_VLM_MODES = {"planner_vlm", "planner_input", "subtask_retrieval"}
PLANNER_MEMORY_MODES = {"planner_oft_memory", "planner_memory", "subtask_window_memory"}


def collate_fn(batch):
    return batch


def _rank0() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _normalize_scalar(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value).reshape(-1)
        if arr.size == 0:
            return None
        value = arr[0]
        if hasattr(value, "item"):
            value = value.item()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def build_stride_frame_indices(
    current_index: int,
    stride: int = 16,
    max_history_frames: int | None = None,
) -> list[int]:
    """Return fixed-stride frames ordered earliest to latest, including current."""
    stride = max(int(stride), 1)
    current_index = int(current_index)
    frames = list(range(current_index, -1, -stride))
    frames.reverse()
    if max_history_frames is not None:
        keep = int(max_history_frames) + 1
        if keep > 0:
            frames = frames[-keep:]
    return frames


def match_keyframes_to_stride_indices(
    keyframes: Sequence[int],
    sampled_frames: Sequence[int],
    stride: int = 16,
) -> list[int]:
    """Map subtask keyframes to sampled-frame indices via k <= s < k + stride."""
    stride = max(int(stride), 1)
    retrieval_indices: list[int] = []
    seen: set[int] = set()
    sampled = [int(frame) for frame in sampled_frames]

    for keyframe in sorted({int(k) for k in keyframes}):
        for idx, frame in enumerate(sampled):
            if keyframe <= frame < keyframe + stride:
                if idx not in seen:
                    retrieval_indices.append(idx)
                    seen.add(idx)
                break
    return retrieval_indices


class PlannerOFTSingleDataset(LeRobotSingleDataset):
    """LeRobot dataset with planner-specific frame selection."""

    def _history_mode(self) -> str:
        return str(self._history_cfg_get("mode", "action_keyframe"))

    def _planner_stride(self) -> int:
        return self._history_stride()

    def _find_optional_trajectory_column(self, configured_column, aliases: list[str]) -> str | None:
        try:
            return self._find_trajectory_column(configured_column, aliases)
        except KeyError:
            return None

    def _subtask_keyframe_indices(self, base_index: int) -> list[int]:
        assert self.curr_traj_data is not None, "No trajectory data loaded"
        end = min(int(base_index), len(self.curr_traj_data) - 1)
        if end < 0:
            return []

        keyframe_column = self._find_optional_trajectory_column(
            self._history_cfg_get("subtask_keyframe_column", None),
            ["subtask_keyframe", "is_subtask_keyframe"],
        )
        if keyframe_column is not None:
            values = self.curr_traj_data[keyframe_column].to_numpy()
            return [i for i in range(end + 1) if self._is_keyframe_value(values[i])]

        index_column = self._find_optional_trajectory_column(
            self._history_cfg_get("subtask_index_column", None),
            ["subtask_instruction_index", "subtask_index"],
        )
        if index_column is None:
            return []

        keyframes: list[int] = []
        previous = object()
        for idx in range(end + 1):
            value = _normalize_scalar(self.curr_traj_data[index_column].iloc[idx])
            if value is None:
                continue
            if not keyframes or value != previous:
                keyframes.append(idx)
            previous = value
        return keyframes

    def _planner_retrieval_indices(self, base_index: int, sampled_frames: Sequence[int]) -> list[int]:
        return match_keyframes_to_stride_indices(
            self._subtask_keyframe_indices(base_index),
            sampled_frames,
            stride=self._planner_stride(),
        )

    def _history_frame_indices(
        self,
        base_index: int,
        rng: np.random.Generator | None = None,
    ) -> list[int]:
        mode = self._history_mode()
        if mode in PLANNER_VLM_MODES:
            return build_stride_frame_indices(
                base_index,
                stride=self._planner_stride(),
                max_history_frames=self._max_history_frames(),
            )

        if mode in PLANNER_MEMORY_MODES:
            sampled_frames = build_stride_frame_indices(
                base_index,
                stride=self._planner_stride(),
                max_history_frames=None,
            )
            retrieval_indices = self._planner_retrieval_indices(base_index, sampled_frames)
            memory_frames = [sampled_frames[idx] for idx in retrieval_indices]
            if not memory_frames or memory_frames[-1] != int(base_index):
                memory_frames.append(int(base_index))
            return memory_frames

        return super()._history_frame_indices(base_index, rng=rng)

    def _pack_sample(
        self,
        data: dict,
        trajectory_id: int | None = None,
        base_index: int | None = None,
        history_rng: np.random.Generator | None = None,
    ) -> dict:
        sample = super()._pack_sample(
            data,
            trajectory_id=trajectory_id,
            base_index=base_index,
            history_rng=history_rng,
        )
        if trajectory_id is None or base_index is None:
            return sample

        mode = self._history_mode()
        if mode not in PLANNER_VLM_MODES and mode not in PLANNER_MEMORY_MODES:
            return sample

        sampled_frames = [int(frame) for frame in sample.get("history_frame_indices", [])]
        sample["planner_oft_mode"] = mode
        sample["planner_stride"] = self._planner_stride()
        sample["trajectory_id"] = int(trajectory_id)
        sample["base_index"] = int(base_index)

        if mode in PLANNER_VLM_MODES:
            retrieval_indices = self._planner_retrieval_indices(base_index, sampled_frames)
            sample["retrieval_indices"] = retrieval_indices
            sample["retrieval_frame_indices"] = [sampled_frames[idx] for idx in retrieval_indices]
            sample["subtask_keyframe_indices"] = self._subtask_keyframe_indices(base_index)

        if mode in PLANNER_MEMORY_MODES:
            sample["memory_frame_indices"] = sampled_frames[:-1] if sampled_frames and sampled_frames[-1] == int(base_index) else sampled_frames

        return sample


def make_planner_oft_single_dataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> PlannerOFTSingleDataset:
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = Path(data_root_dir) / data_name
    embodiment_tag = getattr(data_config, "embodiment_tag", None)
    if embodiment_tag is None:
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    video_backend = _cfg_get(data_cfg, "video_backend", "torchvision_av")
    return PlannerOFTSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
    )


def get_planner_oft_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    data_root_dir = _cfg_get(data_cfg, "data_root_dir")
    data_mix = _cfg_get(data_cfg, "data_mix")
    delete_pause_frame = _cfg_get(data_cfg, "delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    logger.info("[planner_oft] Using mixture %s: %s", data_mix, [(d, w, r) for d, w, r in mixture_spec])

    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:
        if float(d_weight) <= 0.0:
            continue
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            continue
        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = [
        (
            make_planner_oft_single_dataset(
                Path(data_root_dir),
                d_name,
                robot_type,
                delete_pause_frame=delete_pause_frame,
                data_cfg=data_cfg,
            ),
            d_weight,
        )
        for d_name, d_weight, robot_type in filtered_mixture_spec
    ]

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )


class PlannerOFTThinkDataset(Dataset):
    """Qwen-VL supervised planner data with subtask and retrieval outputs."""

    def __init__(self, processor, cfg, data_args):
        super().__init__()
        self.processor = processor
        self.data_args = data_args
        self.prompt_template = getattr(data_args, "CoT_prompt", "{instruction}")
        self.prompt_instruction_source = getattr(data_args, "prompt_instruction_source", "instruction")
        self.prompt_instruction_key = getattr(data_args, "prompt_instruction_key", None)
        self.include_state = _as_bool(getattr(data_args, "include_state", False))

        answer_cfg = getattr(data_args, "think_answer", {}) or {}
        if isinstance(answer_cfg, SimpleNamespace):
            answer_cfg = vars(answer_cfg)
        self.answer_template = answer_cfg.get(
            "template",
            '<think>Frames: {num_frames} total ({num_history} history + current). '
            'Now the subtask is "{subtask_instruction}"</think>'
            '<subtask>{subtask_instruction}</subtask><retrieval>{retrieval_indices}</retrieval>',
        )
        self.answer_instruction_source = answer_cfg.get("instruction_source", "subtask_instruction")
        self.answer_instruction_key = answer_cfg.get("instruction_key", "subtask_lang")

        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)

        dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.datasets.vlm_data, resolve=True))
        dataset_cfg.include_state = self.include_state
        dataset_cfg.history.enabled = True
        dataset_cfg.history.mode = "planner_vlm"

        self.vla_dataset = get_planner_oft_dataset(
            data_cfg=dataset_cfg,
            balance_dataset_weights=dataset_cfg.get("balance_dataset_weights", False),
            balance_trajectory_weights=dataset_cfg.get("balance_trajectory_weights", False),
        )
        self.max_samples = getattr(data_args, "max_samples", None)

    def __len__(self):
        if self.max_samples in (None, "", 0, "0"):
            return len(self.vla_dataset)
        return min(len(self.vla_dataset), int(self.max_samples))

    def _format_answer(self, sample: Dict[str, Any]) -> tuple[str, str]:
        prompt_instruction = _select_lerobot_instruction(
            sample,
            self.prompt_instruction_source,
            self.prompt_instruction_key,
        )
        subtask_instruction = _select_lerobot_instruction(
            sample,
            self.answer_instruction_source,
            self.answer_instruction_key,
        )
        num_frames = int(sample.get("num_frames", len(sample["image"])))
        num_history = int(sample.get("num_history_frames", max(num_frames - 1, 0)))
        retrieval_indices = json.dumps([int(idx) for idx in sample.get("retrieval_indices", [])])

        answer = self.answer_template.format(
            instruction=prompt_instruction,
            target_instruction=subtask_instruction,
            subtask_instruction=subtask_instruction,
            num_frames=num_frames,
            num_history=num_history,
            retrieval_indices=retrieval_indices,
        )
        prompt = self.prompt_template.replace("{instruction}", prompt_instruction)
        return prompt, answer

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self.vla_dataset[i]
        prompt, answer = self._format_answer(sample)
        images = sample["image"]
        if not isinstance(images, list):
            images = [images]

        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]

        data_dict = preprocess_qwen_messages(messages, self.processor, response_text=answer)
        data_dict = _add_qwen_position_ids(
            data_dict,
            self.processor,
            self.get_rope_index,
            self.merge_size,
        )
        if self.include_state:
            data_dict["state_insert_index"] = _qwen_user_content_end_index(messages, self.processor)
            state_history = sample.get("state_history", sample.get("state", None))
            if state_history is None:
                raise KeyError("PlannerOFTThinkDataset requires state_history when include_state is true.")
            state_history = np.asarray(state_history, dtype=np.float32)
            if state_history.ndim == 1:
                state_history = state_history[None, :]
            if state_history.shape[0] != len(images):
                raise ValueError(
                    f"Expected one state vector per image, got {state_history.shape[0]} states for {len(images)} images."
                )
            data_dict["state_history"] = state_history
        return data_dict


def make_planner_oft_vlm_dataloader(cfg):
    data_args = cfg.datasets.vlm_data
    processor = transformers.AutoProcessor.from_pretrained(cfg.framework.qwenvl.base_vlm)
    processor.tokenizer.model_max_length = int(data_args.model_max_length)
    processor.tokenizer.padding_side = "left"

    data_args_ns = SimpleNamespace(**OmegaConf.to_container(data_args, resolve=True))
    data_args_ns.data_flatten = getattr(data_args_ns, "data_flatten", False)
    data_args_ns.data_packing = getattr(data_args_ns, "data_packing", False)
    processor = update_processor_pixels(processor, data_args_ns)

    train_dataset = PlannerOFTThinkDataset(processor=processor, cfg=cfg, data_args=data_args_ns)
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)

    return DataLoader(
        train_dataset,
        batch_size=int(data_args.per_device_batch_size),
        collate_fn=data_collator,
        num_workers=int(data_args.get("num_workers", 4)),
        pin_memory=bool(data_args.get("pin_memory", True)),
    )


def make_planner_oft_vla_dataloader(cfg):
    data_args = cfg.datasets.vla_data
    train_dataset = get_planner_oft_dataset(
        data_cfg=data_args,
        balance_dataset_weights=data_args.get("balance_dataset_weights", False),
        balance_trajectory_weights=data_args.get("balance_trajectory_weights", False),
    )

    output_dir = getattr(cfg, "output_dir", None)
    if output_dir and _rank0():
        train_dataset.save_dataset_statistics(Path(output_dir) / "dataset_statistics.json")

    num_workers = int(data_args.get("num_workers", 4))
    dataloader_kwargs = {
        "batch_size": int(data_args.per_device_batch_size),
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": bool(data_args.get("pin_memory", True)),
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = bool(data_args.get("persistent_workers", True))
        dataloader_kwargs["prefetch_factor"] = int(data_args.get("prefetch_factor", 2))

    return DataLoader(train_dataset, **dataloader_kwargs)
