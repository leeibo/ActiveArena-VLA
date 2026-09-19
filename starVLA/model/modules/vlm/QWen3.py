# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

from pathlib import Path
import time
from typing import Optional

import torch
from starVLA.model.tools import has_flash_attn  # unified flash-attn detection (GPU / NPU)
from starVLA.training.trainer_utils import initialize_overwatch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

from starVLA.model.modules.vlm.chat_label_utils import mask_labels_to_response

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 151669  # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = (
    153716  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
)


import torch.nn as nn


def _resolve_local_model_id(model_id: str) -> str:
    model_path = Path(model_id).expanduser()
    if model_path.is_absolute():
        resolved = model_path
    elif model_id.startswith(("./", "../")) or "/" in model_id:
        repo_root = Path(__file__).resolve().parents[4]
        resolved = repo_root / model_path
    else:
        return model_id

    if resolved.exists():
        return str(resolved)

    # Check both release and pre-release locations. This keeps older checkpoint
    # sidecars usable after the ActiveArena model directory was namespaced.
    candidates = []
    parts = model_path.parts
    if "ActiveArena" in parts:
        # Canonical release path -> pre-release path.
        idx = parts.index("ActiveArena")
        candidates.append(repo_root / Path(*parts[:idx], *parts[idx + 1 :]))
    elif "Pretrained_models" in parts:
        # Legacy path -> canonical release path.
        idx = parts.index("Pretrained_models")
        candidates.append(repo_root / Path(*parts[: idx + 1], "ActiveArena", *parts[idx + 1 :]))
    for candidate in candidates:
        if candidate.exists():
            logger.warning(
                "Configured base_vlm %r is absent; using compatibility path %s. "
                "Set ACTIVEARENA_VLA_BASE_VLM to the canonical snapshot.",
                model_id,
                candidate,
            )
            return str(candidate)

    if model_id.startswith(("./", "../")) or model_path.is_absolute():
        raise FileNotFoundError(
            "Configured Qwen3-VL base_vlm does not exist locally: "
            f"{model_id!r} resolved to {resolved}. "
            "Create/symlink this directory or set ACTIVEARENA_VLA_BASE_VLM "
            "(legacy: STARVLA_BASE_VLM) to a valid local model path."
        )
    return model_id


class _QWen3_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")
        model_id = _resolve_local_model_id(model_id)
        base_vlm_revision = qwenvl_config.get("base_vlm_revision") or qwenvl_config.get("revision")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")
        # Fallback to sdpa if flash_attention_2 is requested but flash_attn is not installed
        if attn_implementation == "flash_attention_2":
            if not has_flash_attn():
                print("[WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"
        logger.info(f"[Qwen3-VL] Loading {model_id} with attn_implementation={attn_implementation}")

        model_kwargs = {
            "attn_implementation": attn_implementation,
            "dtype": torch.bfloat16,
            "ignore_mismatched_sizes": True,
        }
        processor_kwargs = {}
        if base_vlm_revision and not Path(model_id).is_dir():
            model_kwargs["revision"] = base_vlm_revision
            processor_kwargs["revision"] = base_vlm_revision
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        # alin qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen2.5-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
        """

        # Create messages: one message per sample
        messages = []
        prompt_messages = []
        prompt_suffixes = kwargs.get("prompt_suffixes", None)
        if prompt_suffixes is not None:
            assert len(prompt_suffixes) == len(instructions), "Prompt suffixes and instructions must have the same length"

        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for sample_idx, (imgs, instruction) in enumerate(zip(images, instructions)):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction
            if prompt_suffixes is not None:
                prompt = f"{prompt}{prompt_suffixes[sample_idx]}"

            content.append({"type": "text", "text": prompt})
            prompt_messages.append([{"role": "user", "content": content}])
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        collect_timing = bool(getattr(self, "_collect_step_timing", False))
        processor_started = time.perf_counter() if collect_timing else None
        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=solutions is None,
            return_dict=True,
            return_tensors="pt",
        )
        processor_elapsed = time.perf_counter() - processor_started if collect_timing else None

        # If solutions are provided, supervise the whole assistant response.
        if solutions is not None:
            prompt_texts = [
                self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in prompt_messages
            ]
            batch_inputs["labels"] = mask_labels_to_response(
                batch_inputs,
                self.processor.tokenizer,
                prompt_texts,
                ignore_index=IGNORE_INDEX,
                response_texts=solutions,
            )

        h2d_started = time.perf_counter() if collect_timing else None
        h2d_events = None
        if collect_timing:
            h2d_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            h2d_events[0].record()
        batch_inputs = batch_inputs.to(self.model.device)
        if collect_timing:
            h2d_events[1].record()
            self._last_input_timing = {
                "host": {
                    "timing/input_processor_host": processor_elapsed,
                    "timing/input_h2d_host": time.perf_counter() - h2d_started,
                },
                "cuda_events": {"timing/input_h2d_gpu": h2d_events},
            }
        else:
            self._last_input_timing = None
        return batch_inputs


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    qwen_vl = _QWen3_VL_Interface(cfg)
    pass
