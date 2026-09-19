# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""
QwenGR00TState Framework

GR00T DiT action generation conditioned on Qwen3 state-history soft tokens.
The state path matches QwenOFTState: each image frame has one robot-state
vector, projected to a Qwen hidden-size token and inserted into the VLM prompt.
The GR00T action head receives the resulting Qwen hidden states and does not
use its own proprioception branch, avoiding duplicate state conditioning.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenFastState import StateHistoryEncoder
from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
from starVLA.model.framework.VLM4A.QwenOFT import IGNORE_INDEX
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class QwenGR00TStateDefaultConfig(QwenGR00TDefaultConfig):
    """QwenGR00T plus Qwen-side state soft-token conditioning."""

    name: str = "QwenGR00TState"
    state_model: dict = field(
        default_factory=lambda: {
            "hidden_dim": None,
            "dropout": 0.0,
            "use_frame_embedding": True,
            "max_frames": 256,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenGR00TState")
class Qwen_GR00T_State(baseframework):
    """GR00T DiT action model with OFT-style Qwen state soft tokens."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenGR00TStateDefaultConfig, config)

        original_state_dim = self._ensure_state_model_cfg()
        self.config.framework.action_model.state_dim = 0

        self.qwen_vl_interface = get_vlm_model(config=self.config)
        base_vlm = str(self.config.framework.qwenvl.base_vlm)
        if "Qwen3-VL" not in base_vlm:
            raise ValueError(f"QwenGR00TState currently supports Qwen3-VL only, got {base_vlm}")

        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = hidden_size
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        state_cfg = self.config.framework.get("state_model", {})
        state_dim = int(state_cfg.get("state_dim", original_state_dim))
        self.state_encoder = StateHistoryEncoder(
            state_dim=state_dim,
            hidden_size=hidden_size,
            mlp_hidden_dim=state_cfg.get("hidden_dim", None),
            dropout=float(state_cfg.get("dropout", 0.0)),
            use_frame_embedding=bool(state_cfg.get("use_frame_embedding", True)),
            max_frames=int(state_cfg.get("max_frames", 256)),
        )

        tokenizer = self.qwen_vl_interface.processor.tokenizer
        self._state_position_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
        self._im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self._im_end_token_id is None:
            self._im_end_token_id = tokenizer.eos_token_id

    def _ensure_state_model_cfg(self) -> int:
        state_cfg = self.config.framework.get("state_model", None)
        action_cfg = self.config.framework.action_model
        configured_state_dim = None
        if state_cfg is not None:
            configured_state_dim = state_cfg.get("state_dim", None)
        if configured_state_dim is None:
            configured_state_dim = action_cfg.get("state_dim", action_cfg.action_dim)
        if int(configured_state_dim) <= 0:
            configured_state_dim = action_cfg.action_dim

        if state_cfg is None:
            self.config.framework.state_model = OmegaConf.create({})
            state_cfg = self.config.framework.state_model
        if state_cfg.get("state_dim", None) is None:
            state_cfg.state_dim = int(configured_state_dim)
        return int(configured_state_dim)

    def _example_state_history(self, example: dict) -> np.ndarray:
        state_history = example.get("state_history", None)
        if state_history is None:
            state_history = example.get("state", None)
        if state_history is None:
            raise KeyError(
                "QwenGR00TState requires sample['state_history'] or sample['state']. "
                "Set datasets.vla_data.include_state: true."
            )

        state_history = np.asarray(state_history, dtype=np.float32)
        if state_history.ndim == 1:
            state_history = state_history[None, :]
        if state_history.ndim != 2:
            raise ValueError(f"state_history must be [num_frames, state_dim], got {state_history.shape}")

        images = example.get("image", [])
        num_images = len(images) if isinstance(images, list) else 1
        if state_history.shape[0] != num_images:
            raise ValueError(
                f"Expected one state token per image frame, got {state_history.shape[0]} states "
                f"for {num_images} images."
            )
        return state_history

    def _embedding_layer(self):
        qwen_model = self.qwen_vl_interface.model
        if hasattr(qwen_model, "model") and hasattr(qwen_model.model, "get_input_embeddings"):
            return qwen_model.model.get_input_embeddings()
        return qwen_model.get_input_embeddings()

    def _qwen3_core_model(self):
        qwen_model = self.qwen_vl_interface.model
        if not hasattr(qwen_model, "model") or not hasattr(qwen_model.model, "get_rope_index"):
            raise AttributeError("QwenGR00TState expects Qwen3VLForConditionalGeneration.model.get_rope_index")
        return qwen_model.model

    def _build_position_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, qwen_inputs: dict):
        core_model = self._qwen3_core_model()
        position_ids, rope_deltas = core_model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=qwen_inputs.get("image_grid_thw", None),
            video_grid_thw=qwen_inputs.get("video_grid_thw", None),
            attention_mask=attention_mask,
        )
        core_model.rope_deltas = rope_deltas
        return position_ids

    def _prepare_state_conditioned_inputs(
        self,
        qwen_inputs: dict,
        state_histories: List[np.ndarray],
        insert_before: str = "user_end",
        insert_indices=None,
    ) -> tuple[dict, torch.Tensor]:
        input_ids = qwen_inputs["input_ids"]
        attention_mask = qwen_inputs["attention_mask"]
        labels = qwen_inputs.get("labels", None)
        device = input_ids.device
        embed_layer = self._embedding_layer()
        base_embeds = embed_layer(input_ids)
        batch_size, _, hidden_size = base_embeds.shape
        if len(state_histories) != batch_size:
            raise ValueError(f"Expected {batch_size} state histories, got {len(state_histories)}")

        valid_embeds = []
        valid_input_ids = []
        valid_attention = []
        valid_labels = [] if labels is not None else None

        for batch_idx in range(batch_size):
            valid_positions = torch.nonzero(attention_mask[batch_idx] != 0, as_tuple=False).flatten()
            seq_len = int(valid_positions.numel())

            embeds_i = base_embeds[batch_idx, valid_positions]
            ids_i = input_ids[batch_idx, valid_positions]
            attn_i = attention_mask[batch_idx, valid_positions]
            labels_i = labels[batch_idx, valid_positions] if labels is not None else None

            if insert_indices is not None:
                insert_at = max(0, min(int(insert_indices[batch_idx]), seq_len))
            elif insert_before == "user_end":
                end_positions = torch.nonzero(ids_i == int(self._im_end_token_id), as_tuple=False).flatten()
                insert_at = int(end_positions[0].item()) if end_positions.numel() > 0 else seq_len
            elif insert_before == "supervised":
                if labels_i is None:
                    insert_at = seq_len
                else:
                    supervised = torch.nonzero(labels_i != IGNORE_INDEX, as_tuple=False).flatten()
                    insert_at = int(supervised[0].item()) if supervised.numel() > 0 else seq_len
            else:
                raise ValueError(f"Invalid insert_before={insert_before!r}; expected 'user_end' or 'supervised'")

            state_tensor = torch.as_tensor(state_histories[batch_idx], dtype=torch.float32, device=device)
            state_tokens = self.state_encoder(state_tensor).to(device=device, dtype=base_embeds.dtype)
            num_state_tokens = state_tokens.shape[0]

            ids_state = ids_i.new_full((num_state_tokens,), int(self._state_position_token_id))
            attn_state = attn_i.new_ones((num_state_tokens,))

            valid_embeds.append(torch.cat([embeds_i[:insert_at], state_tokens, embeds_i[insert_at:]], dim=0))
            valid_input_ids.append(torch.cat([ids_i[:insert_at], ids_state, ids_i[insert_at:]], dim=0))
            valid_attention.append(torch.cat([attn_i[:insert_at], attn_state, attn_i[insert_at:]], dim=0))

            if labels_i is not None:
                labels_state = labels_i.new_full((num_state_tokens,), IGNORE_INDEX)
                valid_labels.append(torch.cat([labels_i[:insert_at], labels_state, labels_i[insert_at:]], dim=0))

        max_len = max(item.shape[0] for item in valid_embeds)
        pad_token_id = self.qwen_vl_interface.processor.tokenizer.pad_token_id or 0
        padded_embeds = base_embeds.new_zeros((batch_size, max_len, hidden_size))
        padded_input_ids = input_ids.new_full((batch_size, max_len), int(pad_token_id))
        padded_attention = attention_mask.new_zeros((batch_size, max_len))
        padded_labels = labels.new_full((batch_size, max_len), IGNORE_INDEX) if labels is not None else None

        for batch_idx in range(batch_size):
            item_len = valid_embeds[batch_idx].shape[0]
            start = max_len - item_len
            padded_embeds[batch_idx, start:] = valid_embeds[batch_idx]
            padded_input_ids[batch_idx, start:] = valid_input_ids[batch_idx]
            padded_attention[batch_idx, start:] = valid_attention[batch_idx]
            if padded_labels is not None:
                padded_labels[batch_idx, start:] = valid_labels[batch_idx]

        prepared = {
            key: value
            for key, value in qwen_inputs.items()
            if key not in {
                "input_ids",
                "inputs_embeds",
                "attention_mask",
                "labels",
                "position_ids",
                "state",
                "state_history",
                "state_insert_index",
            }
        }
        prepared["inputs_embeds"] = padded_embeds
        prepared["attention_mask"] = padded_attention
        prepared["position_ids"] = self._build_position_ids(padded_input_ids, padded_attention, qwen_inputs)
        if padded_labels is not None:
            prepared["labels"] = padded_labels
        return prepared, padded_input_ids

    def prepare_vlm_state_conditioned_inputs(self, qwen_inputs: dict) -> dict:
        state_histories = qwen_inputs.get("state_history", None)
        if state_histories is None:
            state_histories = qwen_inputs.get("state", None)
        if state_histories is None:
            raise KeyError(
                "QwenGR00TState VLM cotrain requires batch['state_history'] or batch['state']. "
                "Set datasets.vlm_data.include_state: true only when the VLM dataloader provides state."
            )
        insert_indices = qwen_inputs.get("state_insert_index", None)

        prepared, _ = self._prepare_state_conditioned_inputs(
            qwen_inputs,
            state_histories,
            insert_before="supervised",
            insert_indices=insert_indices,
        )
        return prepared

    def _encode_state_conditioned_hidden(
        self,
        batch_images: List,
        instructions: List[str],
        state_histories: List[np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        qwen_inputs, _ = self._prepare_state_conditioned_inputs(
            qwen_inputs,
            state_histories,
            insert_before="user_end",
        )
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
        return last_hidden, backbone_attention_mask

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        batch_images = [example["image"] for example in examples]
        instructions = [self._select_prompt_instruction(example) for example in examples]
        actions = [example["action"] for example in examples]
        state_histories = [self._example_state_history(example) for example in examples]

        last_hidden, backbone_attention_mask = self._encode_state_conditioned_hidden(
            batch_images,
            instructions,
            state_histories,
        )

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.repeat(repeated_diffusion_steps, 1).to(
                    dtype=torch.bool
                )

            action_loss = self.action_model(
                last_hidden_repeated,
                actions_target_repeated,
                None,
                encoder_attention_mask=backbone_attention_mask,
            )

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs: str) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [self._select_prompt_instruction(example) for example in examples]
        state_histories = [self._example_state_history(example) for example in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        last_hidden, backbone_attention_mask = self._encode_state_conditioned_hidden(
            batch_images,
            instructions,
            state_histories,
        )
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                last_hidden,
                None,
                encoder_attention_mask=backbone_attention_mask,
            )

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    def _vla_data_cfg_get(self, key: str, default=None):
        datasets_cfg = getattr(self.config, "datasets", None)
        vla_data_cfg = getattr(datasets_cfg, "vla_data", None) if datasets_cfg is not None else None
        if vla_data_cfg is None:
            return default
        if hasattr(vla_data_cfg, "get"):
            return vla_data_cfg.get(key, default)
        return getattr(vla_data_cfg, key, default)

    def _oft_instruction_cfg_get(self, key: str, default=None):
        instruction_cfg = self._vla_data_cfg_get("oft_instruction", None)
        if instruction_cfg is not None:
            if hasattr(instruction_cfg, "get"):
                value = instruction_cfg.get(key, None)
            else:
                value = getattr(instruction_cfg, key, None)
            if value is not None:
                return value
        return self._vla_data_cfg_get(f"oft_instruction_{key}", default)

    @staticmethod
    def _first_example_text(example: dict, candidate_keys: List[str]) -> Optional[str]:
        for key in candidate_keys:
            if not key:
                continue
            value = example.get(key, None)
            if value is not None and value != "":
                return str(value)
        return None

    def _select_prompt_instruction(self, example: dict) -> str:
        source = str(
            self._oft_instruction_cfg_get(
                "source",
                self._oft_instruction_cfg_get("instruction_source", "instruction"),
            )
        ).lower()
        configured_key = self._oft_instruction_cfg_get(
            "key",
            self._oft_instruction_cfg_get("instruction_key", None),
        )
        subtask_instruction = self._first_example_text(
            example,
            [configured_key, "subtask_lang", "subtask_instruction", "subtask"],
        )
        task_instruction = self._first_example_text(example, ["task_lang", "lang"]) or ""

        if source in {"subtask", "subtask_instruction", "subtask_lang"}:
            return subtask_instruction or task_instruction
        if source in {"auto", "subtask_or_instruction", "subtask_instruction_or_instruction"}:
            return subtask_instruction or task_instruction
        if source in {"instruction", "task", "task_instruction", "lang"}:
            return task_instruction
        custom_instruction = self._first_example_text(example, [source, configured_key])
        return custom_instruction or task_instruction
