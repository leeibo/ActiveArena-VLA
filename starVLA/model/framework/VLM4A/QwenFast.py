# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
Qwen-Fast Framework

A lightweight implementation for autoregressive discrete action prediction conditioned on multi-view images + instruction.
fast tokenizer is copyright from physical-intelligence/fast

Key Points:
  - Qwen2.5 vision-language backbone
  - Unified action learning via next-token prediction (fast tokenizer)
  - Autoregressive action tokens derived from discretized / symbolized continuous actions

Note: How to add special tokens to Qwen2.5:
  download our model checkpoint with special tokens added: https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.fast_ActionHeader import get_action_model
from starVLA.model.modules.vlm import get_vlm_model


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenFast
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenFastDefaultConfig:
    """QwenFast framework default parameters.

    Autoregressive discrete action prediction via FAST tokenizer.
    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.
    """

    # --- Registry identifier ---
    name: str = "QwenFast"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL with FAST action tokens) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to VLM checkpoint. FAST training should normally use an
            # offline *-Action checkpoint whose tokenizer already contains
            # <robot_action_*> tokens.
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
        }
    )

    # === Action head (FAST tokenizer — discrete next-token prediction) ===
    action_model: dict = field(
        default_factory=lambda: {
            # Action head architecture type
            "action_model_type": "FAST",
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # How many future steps to predict
            "future_action_window_size": 15,
            # How many past steps included in action chunk (usually 0)
            "past_action_window_size": 0,
            # Development fallback only. Managed training configs should keep
            # this false and point base_vlm to an offline *-Action checkpoint.
            "auto_add_action_tokens": False,
            # Initialization for the optional auto-added rows: normal | avg | zero.
            "action_token_init_strategy": "normal",
        }
    )
    fast_decode_failure_preview_chars: int = 4096
    fast_decode_failure_tail_chars: int = 1024


@FRAMEWORK_REGISTRY.register("QwenFast")
class Qwenvl_Fast(baseframework):
    """
    Multimodal vision-language-action model (FAST variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - FAST tokenizer for discretized / symbolized continuous action encoding
      - Autoregressive next-token prediction over action tokens

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenFastDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.action_model = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        # self.hidden_dim = config.framework.action_model.action_hidden_dim

        self.action_model.fast_tokenizer.time_horizon = self.action_horizon
        self.action_model.fast_tokenizer.action_dim = self.config.framework.action_model.action_dim
        self._action_token_id_to_fast_id: Optional[Dict[int, int]] = None
        self._action_fast_id_to_token_id: Optional[Dict[int, int]] = None
        self._ensure_action_token_maps()

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Training forward: directly predict future actions via next-token prediction (no diffusion).

        Flow:
          1. Build QwenVL inputs (images + instruction tokens)
          2. Extract hidden states from configured layer range
          7. Predict action and compute L1 loss

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
            **kwargs: Reserved.

        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  #  [B, [PIL]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B, len, 7]

        # step 0: map_raw_action_to_vlm_action
        batch_fast_tokens = self.action_model.encoder_action2fastoken(actions)  # List[str]

        # batch_fast_tokens = [self.fast_tokenizer(raw_action)[0] for raw_action in raw_actions]
        vlm_action_tokens = [self.map_fast_token_to_vlm_action(fast_tokens) for fast_tokens in batch_fast_tokens]

        solutions = [
            self._build_fast_solution(example, instruction, action_tokens)
            for example, instruction, action_tokens in zip(examples, instructions, vlm_action_tokens)
        ]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions, solutions=solutions
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

        vlm_action_loss = qwenvl_outputs.loss
        if vlm_action_loss is None or torch.isnan(vlm_action_loss):
            vlm_action_loss = torch.tensor(0.0, device=self.qwen_vl_interface.model.device)

        return {"action_loss": vlm_action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Inference: single forward pass to obtain future actions (no diffusion sampling).
        # can be batch forward
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        # train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        # if train_obs_image_size:
        #     batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        instructions = [instruction for instruction in instructions]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        max_new_tokens = int(kwargs.get("max_new_tokens", self.config.framework.get("max_new_tokens", 2048)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.qwen_vl_interface.model.generate(
                **qwen_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        prompt_length = qwen_inputs["input_ids"].shape[1]
        generated_text = self.qwen_vl_interface.processor.tokenizer.batch_decode(
            generated_ids[:, prompt_length:],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        target_fast_tokens = kwargs.get("target_fast_tokens", None)
        # Only consume action tokens from the generated <action>...</action> block.
        batch_fast_action_token_idx = self._extract_fast_action_token_ids_from_text(
            generated_text,
            target_fast_token_ids=target_fast_tokens,
        )
        self._validate_fast_action_token_sequences(
            batch_fast_action_token_idx,
            generated_text=generated_text,
            target_fast_token_ids=target_fast_tokens,
        )
        # --- decode fast tokenizer index to action semantic ---
        normalized_actions = self.action_model.fast_tokenizer.decode(batch_fast_action_token_idx)

        return {"normalized_actions": normalized_actions, "text": generated_text}

    def _validate_fast_action_token_sequences(
        self,
        batch_fast_token_ids: List[List[int]],
        generated_text: Optional[List[str]] = None,
        target_fast_token_ids: Optional[List[List[int]]] = None,
    ) -> None:
        fast_tokenizer = self.action_model.fast_tokenizer
        bpe_tokenizer = getattr(fast_tokenizer, "bpe_tokenizer", None)
        if bpe_tokenizer is None:
            return

        action_dim = int(self.config.framework.action_model.action_dim)
        time_horizon = int(self.action_horizon)
        expected_coeff_count = time_horizon * action_dim
        action_block_pattern = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)
        failures = []

        for batch_idx, fast_token_ids in enumerate(batch_fast_token_ids):
            target_token_ids = (
                target_fast_token_ids[batch_idx]
                if target_fast_token_ids is not None and batch_idx < len(target_fast_token_ids)
                else None
            )
            full_text = generated_text[batch_idx] if generated_text is not None and batch_idx < len(generated_text) else None
            action_text = None
            if generated_text is not None and batch_idx < len(generated_text):
                match = action_block_pattern.search(generated_text[batch_idx])
                action_text = match.group(1) if match is not None else None

            if not fast_token_ids:
                failures.append(
                    self._build_fast_decode_failure_detail(
                        batch_idx=batch_idx,
                        actual_token_ids=list(fast_token_ids),
                        actual_coeff_count=0,
                        target_token_ids=target_token_ids,
                        action_text=action_text,
                        generated_text=full_text,
                    )
                )
                continue
            try:
                decoded_text = bpe_tokenizer.decode([int(token_id) for token_id in fast_token_ids])
                coeff_count = len(decoded_text)
            except Exception:
                coeff_count = -1

            if coeff_count != expected_coeff_count:
                failures.append(
                    self._build_fast_decode_failure_detail(
                        batch_idx=batch_idx,
                        actual_token_ids=list(fast_token_ids),
                        actual_coeff_count=coeff_count,
                        target_token_ids=target_token_ids,
                        action_text=action_text,
                        generated_text=full_text,
                    )
                )

        if failures:
            raise RuntimeError(
                "QwenFast generation produced invalid FAST action token sequence(s): "
                f"expected decoded coefficient count {expected_coeff_count} "
                f"(time_horizon={time_horizon}, action_dim={action_dim}), got {failures}."
            )

    def _build_fast_decode_failure_detail(
        self,
        *,
        batch_idx: int,
        actual_token_ids: List[int],
        actual_coeff_count: int,
        target_token_ids: Optional[List[int]] = None,
        action_text: Optional[str] = None,
        generated_text: Optional[str] = None,
    ) -> dict:
        fast_tokenizer = self.action_model.fast_tokenizer
        bpe_tokenizer = getattr(fast_tokenizer, "bpe_tokenizer", None)
        generated_text = generated_text or ""
        action_text = action_text or ""
        preview_chars = int(self.config.framework.get("fast_decode_failure_preview_chars", 4096))
        tail_chars = int(self.config.framework.get("fast_decode_failure_tail_chars", 1024))

        def _tail_preview(text: str) -> str:
            if tail_chars <= 0 or len(text) <= preview_chars:
                return ""
            return text[-tail_chars:]

        detail = {
            "sample": batch_idx,
            "actual_decoded_coeff_count": actual_coeff_count,
            "actual_token_count": len(actual_token_ids),
            "actual_tokens_preview": actual_token_ids[:64],
            "actual_action_text_char_count": len(action_text),
            "actual_action_text_preview": action_text[:preview_chars],
            "actual_action_text_tail_preview": _tail_preview(action_text),
            "actual_action_text_preview_truncated": len(action_text) > preview_chars,
            "generated_text_char_count": len(generated_text),
            "text_preview": generated_text[:preview_chars],
            "text_tail_preview": _tail_preview(generated_text),
            "text_preview_truncated": len(generated_text) > preview_chars,
            "preview_chars": preview_chars,
        }
        if target_token_ids is not None:
            target_coeff_count = None
            if bpe_tokenizer is not None:
                try:
                    target_coeff_count = len(bpe_tokenizer.decode([int(token_id) for token_id in target_token_ids]))
                except Exception:
                    target_coeff_count = -1
            detail.update(
                {
                    "target_decoded_coeff_count": target_coeff_count,
                    "target_token_count": len(target_token_ids),
                    "target_tokens_preview": target_token_ids[:64],
                }
            )
        return detail

    def _extract_fast_action_token_ids_from_text(
        self,
        generated_text: List[str],
        target_fast_token_ids: Optional[List[List[int]]] = None,
    ) -> List[List[int]]:
        """Extract FAST token ids strictly from each generated <action>...</action> block."""
        action_block_pattern = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)
        action_token_pattern = re.compile(r"<robot_action_(\d+)>")
        fast_vocab_size = self._fast_action_vocab_size()

        batch_fast_token_ids = []
        failures = []
        for batch_idx, text in enumerate(generated_text):
            target_token_ids = (
                target_fast_token_ids[batch_idx]
                if target_fast_token_ids is not None and batch_idx < len(target_fast_token_ids)
                else None
            )
            match = action_block_pattern.search(text)
            if match is None:
                batch_fast_token_ids.append([])
                partial_action_text = self._extract_partial_action_text(text)
                detail = self._build_fast_decode_failure_detail(
                    batch_idx=batch_idx,
                    actual_token_ids=[],
                    actual_coeff_count=0,
                    target_token_ids=target_token_ids,
                    action_text=partial_action_text,
                    generated_text=text,
                )
                detail["reason"] = "missing_complete_action_block"
                detail["has_open_action_tag"] = "<action>" in text
                detail["has_close_action_tag"] = "</action>" in text
                failures.append(detail)
                continue

            action_text = match.group(1)
            token_ids = [int(value) for value in action_token_pattern.findall(action_text)]
            invalid_token_ids = [token_id for token_id in token_ids if token_id < 0 or token_id >= fast_vocab_size]
            if not token_ids:
                detail = self._build_fast_decode_failure_detail(
                    batch_idx=batch_idx,
                    actual_token_ids=[],
                    actual_coeff_count=0,
                    target_token_ids=target_token_ids,
                    action_text=action_text,
                    generated_text=text,
                )
                detail["reason"] = "empty_action_block"
                failures.append(detail)
            elif invalid_token_ids:
                detail = self._build_fast_decode_failure_detail(
                    batch_idx=batch_idx,
                    actual_token_ids=token_ids,
                    actual_coeff_count=-1,
                    target_token_ids=target_token_ids,
                    action_text=action_text,
                    generated_text=text,
                )
                detail.update(
                    {
                        "reason": "action_token_out_of_range",
                        "invalid_token_ids": invalid_token_ids[:16],
                        "fast_vocab_size": fast_vocab_size,
                    }
                )
                failures.append(detail)

            batch_fast_token_ids.append(token_ids)

        if failures:
            raise RuntimeError(
                "QwenFast generation produced invalid <action> block(s) while extracting FAST tokens: "
                f"{failures}."
            )

        return batch_fast_token_ids

    @staticmethod
    def _extract_partial_action_text(text: str) -> str:
        text = str(text or "")
        start = text.find("<action>")
        if start < 0:
            return ""
        return text[start + len("<action>") :]

    def _extract_action_token_ids(
        self,
        generated_ids: torch.LongTensor,
    ) -> List[List[int]]:
        """
        Extract action tokens (with offset) from the generated token sequence and return a 2D list:
        ret[b] = [vlm_action_token_id_0, vlm_action_token_id_1, ...]
        Rule: keep all tokens falling within [_ACTION_TOKEN_MIN, _ACTION_TOKEN_MAX] in order of appearance.
        You may change it to "take only the first occurrence followed by continuous segment" as needed.
        """
        action_token_id_to_fast_id, _ = self._ensure_action_token_maps()
        results = []
        for b in range(generated_ids.size(0)):
            tokens = [
                int(token_id)
                for token_id in generated_ids[b].tolist()
                if int(token_id) in action_token_id_to_fast_id
            ]
            results.append(tokens)
        return results

    def _decode_action_tokens(self, batch_vlm_tokens: List[List[int]]) -> List[Any]:
        """
        Decode the offset VLM action token list back to fast tokenizer semantics.
        fast_tokenizer.decode expects the original fast token id sequence (without offset).
        """
        action_token_id_to_fast_id, _ = self._ensure_action_token_maps()
        batch_fast_token_ids = []
        for seq in batch_vlm_tokens:
            if not seq:
                batch_fast_token_ids.append([])
                continue
            fast_ids = [action_token_id_to_fast_id[int(t)] for t in seq]

            batch_fast_token_ids.append(fast_ids)

        return batch_fast_token_ids

    def _fast_action_vocab_size(self) -> int:
        fast_tokenizer = getattr(self.action_model, "fast_tokenizer", None)
        vocab_size = getattr(fast_tokenizer, "vocab_size", None)
        if vocab_size is None:
            nested_tokenizer = getattr(fast_tokenizer, "tokenizer", None)
            vocab_size = getattr(nested_tokenizer, "vocab_size", None)
        return int(vocab_size or 2048)

    def _ensure_action_token_maps(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        if self._action_token_id_to_fast_id is not None and self._action_fast_id_to_token_id is not None:
            return self._action_token_id_to_fast_id, self._action_fast_id_to_token_id

        tokenizer = self.qwen_vl_interface.processor.tokenizer
        vocab = tokenizer.get_vocab()
        fast_vocab_size = self._fast_action_vocab_size()
        token_id_to_fast_id: Dict[int, int] = {}
        fast_id_to_token_id: Dict[int, int] = {}
        missing_tokens = []

        for fast_id in range(fast_vocab_size):
            token = f"<robot_action_{fast_id}>"
            token_id = vocab.get(token, None)
            if token_id is None:
                missing_tokens.append(token)
                continue
            token_id = int(token_id)
            token_id_to_fast_id[token_id] = fast_id
            fast_id_to_token_id[fast_id] = token_id

        if missing_tokens:
            auto_add = bool(self.config.framework.action_model.get("auto_add_action_tokens", False))
            if not auto_add:
                model_id = self.config.framework.qwenvl.get("base_vlm", "<unknown>")
                sample = ", ".join(missing_tokens[:5])
                raise ValueError(
                    f"QwenFast requires {fast_vocab_size} tokenizer special tokens named "
                    f"<robot_action_0>...<robot_action_{fast_vocab_size - 1}>, but base_vlm={model_id!r} "
                    f"is missing {len(missing_tokens)} of them. Missing examples: {sample}. "
                    "Use an offline action-token checkpoint such as Qwen3-VL-2B-Instruct-Action, "
                    "or explicitly set framework.action_model.auto_add_action_tokens=true for a development run."
                )

            init_strategy = str(
                self.config.framework.action_model.get("action_token_init_strategy", "normal")
            ).lower()
            self._add_action_tokens_to_vlm(missing_tokens, init_strategy=init_strategy)

            vocab = tokenizer.get_vocab()
            token_id_to_fast_id.clear()
            fast_id_to_token_id.clear()
            still_missing = []
            for fast_id in range(fast_vocab_size):
                token = f"<robot_action_{fast_id}>"
                token_id = vocab.get(token, None)
                if token_id is None:
                    still_missing.append(token)
                    continue
                token_id = int(token_id)
                token_id_to_fast_id[token_id] = fast_id
                fast_id_to_token_id[fast_id] = token_id
            if still_missing:
                raise RuntimeError(
                    "Failed to add all FAST action tokens to the VLM tokenizer. "
                    f"Still missing {len(still_missing)} tokens, e.g. {still_missing[:5]}."
                )

        self._action_token_id_to_fast_id = token_id_to_fast_id
        self._action_fast_id_to_token_id = fast_id_to_token_id
        return token_id_to_fast_id, fast_id_to_token_id

    def _add_action_tokens_to_vlm(self, missing_tokens: List[str], init_strategy: str = "normal") -> None:
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        model = self.qwen_vl_interface.model
        input_embeddings = model.get_input_embeddings()
        old_input_size = int(input_embeddings.weight.shape[0])
        old_mean = input_embeddings.weight.detach().mean(dim=0)

        try:
            added = tokenizer.add_special_tokens(
                {"additional_special_tokens": list(missing_tokens)},
                replace_additional_special_tokens=False,
            )
        except TypeError:
            existing_special_tokens = list(getattr(tokenizer, "additional_special_tokens", []) or [])
            merged_special_tokens = existing_special_tokens + [
                token for token in missing_tokens if token not in existing_special_tokens
            ]
            added = tokenizer.add_special_tokens({"additional_special_tokens": merged_special_tokens})
        token_ids = [int(tokenizer.convert_tokens_to_ids(token)) for token in missing_tokens]
        target_size = max(old_input_size, len(tokenizer), max(token_ids) + 1)
        if target_size > old_input_size:
            try:
                model.resize_token_embeddings(target_size, mean_resizing=False)
            except TypeError:
                model.resize_token_embeddings(target_size)

        self._initialize_action_token_rows(token_ids, old_mean, init_strategy)
        logger.info(
            f"[QwenFast] Added {added} FAST action tokens to VLM tokenizer; "
            f"embedding size {old_input_size} -> {model.get_input_embeddings().weight.shape[0]}."
        )

    def _initialize_action_token_rows(
        self,
        token_ids: List[int],
        old_mean: torch.Tensor,
        init_strategy: str,
    ) -> None:
        model = self.qwen_vl_interface.model
        modules = [model.get_input_embeddings()]
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None and output_embeddings is not modules[0]:
            modules.append(output_embeddings)

        with torch.no_grad():
            for embedding in modules:
                weight = embedding.weight
                valid_ids = [token_id for token_id in token_ids if token_id < weight.shape[0]]
                if not valid_ids:
                    continue
                idx = torch.tensor(valid_ids, device=weight.device, dtype=torch.long)
                if init_strategy == "normal":
                    values = torch.empty(
                        (len(valid_ids), weight.shape[1]),
                        device=weight.device,
                        dtype=weight.dtype,
                    )
                    torch.nn.init.normal_(values, mean=0.0, std=0.02)
                    weight[idx] = values
                elif init_strategy == "zero":
                    weight[idx].zero_()
                elif init_strategy == "avg":
                    mean = old_mean.to(device=weight.device, dtype=weight.dtype)
                    weight[idx] = mean.expand(len(valid_ids), -1)
                else:
                    raise ValueError(
                        f"Unknown framework.action_model.action_token_init_strategy={init_strategy!r}; "
                        "expected normal, avg, or zero."
                    )

    def map_fast_token_to_vlm_action(self, tokens) -> str:
        """Maps fast action tokens to the VLM action format.
        Action token 0 is mapped to the string <robot_action_0>  ... and so on
        """
        return "".join(
            [f"<robot_action_{token}>" for token in tokens]
        )  # you should add <robot_action_{token}> to VLM as special tokens,

    def _vla_data_cfg_get(self, key: str, default=None):
        datasets_cfg = getattr(self.config, "datasets", None)
        vla_data_cfg = getattr(datasets_cfg, "vla_data", None) if datasets_cfg is not None else None
        if vla_data_cfg is None:
            return default
        if hasattr(vla_data_cfg, "get"):
            return vla_data_cfg.get(key, default)
        return getattr(vla_data_cfg, key, default)

    def _fast_answer_cfg_get(self, key: str, default=None):
        answer_cfg = self._vla_data_cfg_get("fast_answer", None)
        if answer_cfg is not None:
            if hasattr(answer_cfg, "get"):
                value = answer_cfg.get(key, None)
            else:
                value = getattr(answer_cfg, key, None)
            if value is not None:
                return value
        return self._vla_data_cfg_get(f"fast_answer_{key}", default)

    def _first_example_text(self, example: dict, candidate_keys: List[str]) -> Optional[str]:
        for key in candidate_keys:
            if not key:
                continue
            value = example.get(key, None)
            if value is not None and value != "":
                return str(value)
        return None

    def _select_solution_instruction(self, example: dict, instruction: str) -> Tuple[str, str, str]:
        source = str(self._fast_answer_cfg_get("instruction_source", "auto")).lower()
        configured_key = self._fast_answer_cfg_get("instruction_key", None)
        subtask_instruction = self._first_example_text(
            example,
            [configured_key, "subtask_lang", "subtask_instruction", "subtask"],
        )
        task_instruction = self._first_example_text(example, ["task_lang", "lang"]) or instruction

        if source in {"instruction", "task", "task_instruction", "lang"}:
            target_instruction = task_instruction
        elif source in {"subtask", "subtask_instruction", "subtask_lang"}:
            target_instruction = subtask_instruction or task_instruction
        elif source in {"auto", "subtask_or_instruction", "subtask_instruction_or_instruction"}:
            target_instruction = subtask_instruction or task_instruction
        else:
            custom_instruction = self._first_example_text(example, [source, configured_key])
            target_instruction = custom_instruction or task_instruction

        return target_instruction, task_instruction, subtask_instruction or task_instruction

    def _build_fast_solution(self, example: dict, instruction: str, fast_tokens: str) -> str:
        template = self._fast_answer_cfg_get("template", None)
        if not template:
            return fast_tokens

        template = str(template).replace("{FAST-TOKEN}", "{fast_tokens}")
        num_frames = int(example.get("num_frames", len(example.get("image", []) or [])))
        num_history = int(example.get("num_history_frames", max(num_frames - 1, 0)))
        target_instruction, task_instruction, subtask_instruction = self._select_solution_instruction(
            example, instruction
        )

        values = _SafeFormatDict(
            {
                "instruction": instruction,
                "target_instruction": target_instruction,
                "task_instruction": task_instruction,
                "subtask_instruction": subtask_instruction,
                "fast_tokens": fast_tokens,
                "action_tokens": fast_tokens,
                "num_frames": num_frames,
                "num_history": num_history,
                "history_mode": example.get("history_mode", "none"),
            }
        )
        return template.format_map(values)


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model = Qwenvl_Fast(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "lang": "This is a fake instruction for testing.",
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    # Untrained models haven't learned the action tokens, so predictions may be empty.
    predict_output = model.predict_action([sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")
