#!/usr/bin/env python
"""Smoke-test an exported QwenFast HF model with vLLM.

This is intentionally an offline benchmark/test script rather than a server.
It reuses the StarVLA dataloader, FAST tokenizer, and normalization processor
so the output can be compared against the existing HF-transformers policy path.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoProcessor

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.modules.action_model.fast_ActionHeader import get_action_model


ACTION_RE = re.compile(r"<robot_action_(\d+)>")
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True, help="HF directory produced by export_qwenfast_hf.py")
    parser.add_argument("--ckpt_path", required=True, help="Original StarVLA checkpoint for norm stats")
    parser.add_argument(
        "--config_yaml",
        default="examples/ActiveArena_Astribot/train_files/managed_runs/fast_subtask_action_12_wos/config.yaml",
    )
    parser.add_argument("--indices", default="12345,456789,876543,1234567,1987654")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--limit_images", type=int, default=16)
    return parser


def _as_pil_list(images: list[Any]) -> list[Image.Image]:
    out = []
    for img in images:
        if isinstance(img, Image.Image):
            out.append(img)
        else:
            out.append(Image.fromarray(np.asarray(img, dtype=np.uint8)))
    return out


def _build_prompt(processor: AutoProcessor, cfg: Any, sample: dict, images: list[Image.Image]) -> str:
    instruction = str(sample["lang"])
    if "CoT_prompt" in cfg.datasets.vla_data:
        prompt = str(cfg.datasets.vla_data.CoT_prompt).replace("{instruction}", instruction)
    else:
        prompt = instruction

    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _make_sampling_params(max_tokens: int):
    from vllm import SamplingParams

    kwargs = {
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "skip_special_tokens": False,
    }
    try:
        return SamplingParams(**kwargs)
    except TypeError:
        kwargs.pop("skip_special_tokens")
        return SamplingParams(**kwargs)


def _extract_think(text: str) -> str:
    match = THINK_RE.search(text)
    return match.group(1).strip() if match else ""


def _extract_fast_ids(text: str) -> list[int]:
    return [int(m.group(1)) for m in ACTION_RE.finditer(text)]


def main() -> None:
    args = build_argparser().parse_args()
    cfg = OmegaConf.load(args.config_yaml)
    cfg.datasets.vla_data.data_root_dir = Path(cfg.datasets.vla_data.data_root_dir)
    cfg.datasets.vla_data.per_device_batch_size = 1
    cfg.datasets.vla_data.num_workers = 0
    cfg.datasets.vla_data.pin_memory = False

    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data, seed=42)
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"

    action_model = get_action_model(cfg.framework.action_model)
    norm_proc = PolicyNormProcessor(args.ckpt_path)

    requests = []
    samples = []
    for idx in indices:
        sample = dataset[idx]
        images = _as_pil_list(sample["image"])
        prompt = _build_prompt(processor, cfg, sample, images)
        requests.append({"prompt": prompt, "multi_modal_data": {"image": images}})
        samples.append((idx, sample, images, prompt))
        print(
            f"prepared idx={idx} frames={len(images)} "
            f"lang={str(sample['lang'])[:120]!r} subtask={str(sample.get('subtask_lang', ''))[:120]!r}",
            flush=True,
        )

    from vllm import LLM

    llm = LLM(
        model=args.model_dir,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": args.limit_images},
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = _make_sampling_params(args.max_tokens)

    start = time.time()
    outputs = llm.generate(requests, sampling_params=sampling_params)
    total = time.time() - start

    pred_actions = []
    gt_actions = []
    for (idx, sample, images, _prompt), output in zip(samples, outputs):
        text = output.outputs[0].text
        fast_ids = _extract_fast_ids(text)
        think = _extract_think(text)
        print(f"--- idx={idx} ---", flush=True)
        print(f"think: {think}", flush=True)
        print(f"text_len={len(text)} action_token_count={len(fast_ids)}", flush=True)
        print(f"text_preview: {text[:500].replace(chr(10), chr(92) + 'n')}", flush=True)

        if not fast_ids:
            print("no action tokens decoded; skipping action comparison", flush=True)
            continue

        normalized = np.asarray(action_model.fast_tokenizer.decode([fast_ids]), dtype=np.float32)[0]
        pred = np.asarray(norm_proc.unapply_actions(normalized), dtype=np.float32)
        gt = np.asarray(norm_proc.unapply_actions(np.asarray(sample["action"], dtype=np.float32)), dtype=np.float32)
        diff = pred - gt
        pred_actions.append(pred)
        gt_actions.append(gt)
        print(
            "metrics: MAE={:.6f} RMSE={:.6f} MaxAbs={:.6f}".format(
                float(np.mean(np.abs(diff))),
                float(np.sqrt(np.mean(diff * diff))),
                float(np.max(np.abs(diff))),
            ),
            flush=True,
        )

    print("=== vLLM timing ===", flush=True)
    print(f"num_requests={len(requests)} total_sec={total:.3f} avg_sec={total / max(len(requests), 1):.3f}", flush=True)
    if pred_actions:
        pred_stack = np.stack(pred_actions, axis=0)
        gt_stack = np.stack(gt_actions, axis=0)
        diff = pred_stack - gt_stack
        print(
            "aggregate: MAE={:.6f} RMSE={:.6f} MaxAbs={:.6f}".format(
                float(np.mean(np.abs(diff))),
                float(np.sqrt(np.mean(diff * diff))),
                float(np.max(np.abs(diff))),
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
