from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset


class ZeroHistoryDataset(Dataset):
    """Paired eval view that keeps only current head/left/right images."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = copy.copy(self.dataset[index])
        num_current = int(sample.get("num_current_frames", 3))
        if num_current != 3:
            raise ValueError(f"Zero-history RoboTwin eval expects 3 current views, got {num_current}")

        sample["image"] = sample["image"][-3:]
        sample["state_history"] = np.asarray(sample["state_history"])[-1:]
        sample["history_frame_indices"] = sample["history_frame_indices"][-1:]
        sample["state_frame_indices"] = sample["state_frame_indices"][-1:]
        sample["state_time_offsets"] = [0]
        sample["image_frame_indices"] = sample["image_frame_indices"][-3:]
        sample["num_frames"] = 3
        sample["num_history_frames"] = 0
        sample["history_sampling_strategy"] = "fixed_count_uniform"
        return sample


def _to_plain_dict(config) -> dict:
    if hasattr(config, "to_dict"):
        return config.to_dict(resolve=True)
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError(f"Cannot convert config of type {type(config)} to dict")


def _apply_training_normalization(eval_dataset, train_dataset) -> None:
    train_metadata = train_dataset.merged_metadata
    for dataset in eval_dataset.datasets:
        if dataset.tag not in train_metadata:
            raise KeyError(f"No training normalization metadata for embodiment tag {dataset.tag!r}")
        dataset.set_transforms_metadata(train_metadata[dataset.tag])


def build_memory_eval_dataloaders(cfg, train_dataset) -> dict[str, DataLoader]:
    eval_cfg = cfg.datasets.get("eval_data", None)
    if eval_cfg is None or not bool(eval_cfg.get("enabled", False)):
        return {}

    base_data_cfg = OmegaConf.create(_to_plain_dict(cfg.datasets.vla_data))
    sources = eval_cfg.get("sources", {})
    batch_size = int(eval_cfg.get("per_device_batch_size", 1))
    num_workers = int(eval_cfg.get("num_workers", 2))
    include_zero_history = bool(eval_cfg.get("zero_history", False))

    dataloaders: dict[str, DataLoader] = {}
    for domain, source_cfg in sources.items():
        data_cfg = OmegaConf.merge(base_data_cfg, OmegaConf.create(_to_plain_dict(source_cfg)))
        eval_dataset = get_vla_dataset(
            data_cfg=data_cfg,
            mode="eval",
            balance_dataset_weights=data_cfg.get("balance_dataset_weights", False),
            balance_trajectory_weights=data_cfg.get("balance_trajectory_weights", False),
            seed=int(eval_cfg.get("seed", 42)),
        )
        _apply_training_normalization(eval_dataset, train_dataset)

        loader_kwargs = {
            "batch_size": batch_size,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
            "pin_memory": bool(eval_cfg.get("pin_memory", True)),
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(eval_cfg.get("persistent_workers", True))
            loader_kwargs["prefetch_factor"] = int(eval_cfg.get("prefetch_factor", 2))

        dataloaders[f"native/{domain}"] = DataLoader(eval_dataset, **loader_kwargs)
        if include_zero_history:
            dataloaders[f"h0/{domain}"] = DataLoader(ZeroHistoryDataset(eval_dataset), **loader_kwargs)

    return dataloaders


@torch.inference_mode()
def evaluate_action_loader(model, dataloader, accelerator, max_batches: int | None = None) -> dict[str, float]:
    totals = torch.zeros(3, dtype=torch.float64, device=accelerator.device)
    unwrapped_model = accelerator.unwrap_model(model)

    for batch_index, examples in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        predicted = unwrapped_model.predict_action(examples=examples)["normalized_actions"]
        target = np.asarray([example["action"] for example in examples], dtype=np.float32)
        difference = torch.as_tensor(predicted - target, dtype=torch.float64, device=accelerator.device)
        totals[0] += difference.abs().sum()
        totals[1] += difference.square().sum()
        totals[2] += difference.numel()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    if totals[2].item() == 0:
        raise RuntimeError("Memory eval dataloader produced no action elements")

    l1 = (totals[0] / totals[2]).item()
    mse = (totals[1] / totals[2]).item()
    return {"action_l1": l1, "action_mse": mse, "action_rmse": float(np.sqrt(mse))}
