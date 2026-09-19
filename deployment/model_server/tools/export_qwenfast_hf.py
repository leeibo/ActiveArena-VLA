#!/usr/bin/env python
"""Export a trained QwenFast checkpoint's VLM submodel to HF/vLLM format.

The StarVLA training checkpoint is a wrapper-level state_dict. vLLM expects a
standard Hugging Face model directory, so this script restores the StarVLA
checkpoint and saves only the Qwen-VL model/processor with the trained weights
and action-token tokenizer.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from starVLA.model.framework.base_framework import baseframework


def _run_dir_from_checkpoint(ckpt_path: Path) -> Path:
    if ckpt_path.is_file() and ckpt_path.parent.name == "checkpoints":
        return ckpt_path.parents[1]
    if ckpt_path.is_file() and ckpt_path.parent.name == "final_model":
        return ckpt_path.parents[1]
    if ckpt_path.is_dir():
        return ckpt_path
    raise ValueError(f"Cannot infer run directory from checkpoint path: {ckpt_path}")


def _copy_if_exists(src: Path, dst_dir: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst_dir / src.name)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="StarVLA .pt/.safetensors checkpoint, e.g. checkpoints/steps_*.pt",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output Hugging Face model directory for vLLM.",
    )
    parser.add_argument("--max_shard_size", default="5GB")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    framework = baseframework.from_pretrained(str(ckpt_path))
    if not hasattr(framework, "qwen_vl_interface"):
        raise TypeError(f"{type(framework).__name__} has no qwen_vl_interface; expected QwenFast.")

    qwen = framework.qwen_vl_interface
    qwen.model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    qwen.processor.save_pretrained(output_dir)

    run_dir = _run_dir_from_checkpoint(ckpt_path)
    _copy_if_exists(run_dir / "config.yaml", output_dir)
    _copy_if_exists(run_dir / "config.full.yaml", output_dir)
    _copy_if_exists(run_dir / "dataset_statistics.json", output_dir)

    metadata = {
        "source_checkpoint": str(ckpt_path),
        "source_run_dir": str(run_dir),
        "framework": type(framework).__name__,
        "action_dim": int(framework.config.framework.action_model.action_dim),
        "action_horizon": int(framework.config.framework.action_model.action_horizon),
    }
    with open(output_dir / "starvla_export_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported QwenFast VLM to: {output_dir}")


if __name__ == "__main__":
    main()
