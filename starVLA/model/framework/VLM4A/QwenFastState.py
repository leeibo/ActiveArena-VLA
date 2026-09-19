# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""
QwenFastState Framework

Qwen3-only FAST variant that conditions action-token prediction on robot
state history. Each state vector is projected to one soft token and inserted
before the supervised assistant response.
"""

from contextlib import nullcontext
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenFast import IGNORE_INDEX, Qwenvl_Fast
from starVLA.model.tools import FRAMEWORK_REGISTRY


class StateHistoryEncoder(nn.Module):
    """Project each low-dimensional robot state to one Qwen hidden-size token."""

    def __init__(
        self,
        state_dim: int,
        hidden_size: int,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_frame_embedding: bool = True,
        max_frames: int = 256,
        time_encoding: str = "none",
        time_fourier_dim: int = 64,
    ) -> None:
        super().__init__()
        mlp_hidden_dim = int(mlp_hidden_dim or hidden_size)
        self.state_dim = int(state_dim)
        self.hidden_size = int(hidden_size)
        self.use_frame_embedding = bool(use_frame_embedding)
        self.max_frames = int(max_frames)
        self.time_encoding = str(time_encoding)

        self.net = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(mlp_hidden_dim, self.hidden_size),
        )
        self.frame_embedding = nn.Embedding(self.max_frames, self.hidden_size) if self.use_frame_embedding else None
        if self.time_encoding == "relative_fourier":
            if int(time_fourier_dim) <= 0 or int(time_fourier_dim) % 2 != 0:
                raise ValueError("time_fourier_dim must be a positive even integer")
            half_dim = int(time_fourier_dim) // 2
            frequencies = torch.exp(
                torch.arange(half_dim, dtype=torch.float32) * (-np.log(10000.0) / max(half_dim - 1, 1))
            )
            self.register_buffer("time_frequencies", frequencies, persistent=False)
            self.time_projection = nn.Linear(int(time_fourier_dim), self.hidden_size, bias=False)
        elif self.time_encoding == "none":
            self.time_frequencies = None
            self.time_projection = None
        else:
            raise ValueError(
                f"Unsupported state time_encoding={self.time_encoding!r}; expected 'none' or 'relative_fourier'"
            )

    def forward(
        self,
        state_history: torch.Tensor,
        time_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state_history.ndim != 2:
            raise ValueError(f"state_history must have shape [num_frames, state_dim], got {state_history.shape}")
        if state_history.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got {state_history.shape[-1]}")

        ref_param = next(self.parameters())
        state_history = state_history.to(device=ref_param.device, dtype=ref_param.dtype)
        if not state_history.is_contiguous():
            state_history = state_history.contiguous()

        autocast_context = (
            torch.autocast(state_history.device.type, enabled=False)
            if state_history.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        with autocast_context:
            state_tokens = self.net(state_history)
        if self.frame_embedding is not None:
            num_frames = state_tokens.shape[0]
            if num_frames > self.max_frames:
                raise ValueError(
                    f"state_history has {num_frames} frames, larger than state_model.max_frames={self.max_frames}"
                )
            frame_ids = torch.arange(num_frames, device=state_tokens.device)
            frame_tokens = self.frame_embedding(frame_ids).to(dtype=state_tokens.dtype)
            state_tokens = state_tokens + frame_tokens
        if self.time_projection is not None:
            if time_offsets is None:
                raise ValueError("relative_fourier time encoding requires one time offset per state token")
            time_offsets = torch.as_tensor(
                time_offsets,
                device=state_tokens.device,
                dtype=torch.float32,
            ).reshape(-1)
            if time_offsets.shape[0] != state_tokens.shape[0]:
                raise ValueError(
                    f"Expected {state_tokens.shape[0]} time offsets, got {time_offsets.shape[0]}"
                )
            angles = time_offsets[:, None] * self.time_frequencies[None, :]
            time_features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
            time_features = time_features.to(dtype=self.time_projection.weight.dtype)
            state_tokens = state_tokens + self.time_projection(time_features).to(dtype=state_tokens.dtype)
        return state_tokens


@FRAMEWORK_REGISTRY.register("QwenFastState")
class Qwenvl_Fast_State(Qwenvl_Fast):
    """
    FAST action-token training with state-history soft tokens.

    The action target remains a FAST token sequence. The state branch only
    provides extra conditioning to the Qwen3 language model.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        base_vlm = str(self.config.framework.qwenvl.base_vlm)
        if "Qwen3-VL" not in base_vlm:
            raise ValueError(f"QwenFastState currently supports Qwen3-VL only, got {base_vlm}")

        state_cfg = self.config.framework.get("state_model", {})
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        state_dim = int(state_cfg.get("state_dim", self.config.framework.action_model.action_dim))
        self.state_encoder = StateHistoryEncoder(
            state_dim=state_dim,
            hidden_size=hidden_size,
            mlp_hidden_dim=state_cfg.get("hidden_dim", None),
            dropout=float(state_cfg.get("dropout", 0.0)),
            use_frame_embedding=bool(state_cfg.get("use_frame_embedding", True)),
            max_frames=int(state_cfg.get("max_frames", 256)),
            time_encoding=str(state_cfg.get("time_encoding", "none")),
            time_fourier_dim=int(state_cfg.get("time_fourier_dim", 64)),
        )

        tokenizer = self.qwen_vl_interface.processor.tokenizer
        self._state_position_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0

    def _example_state_history(self, example: dict) -> np.ndarray:
        state_history = example.get("state_history", None)
        if state_history is None:
            state_history = example.get("state", None)
        if state_history is None:
            raise KeyError(
                "QwenFastState requires sample['state_history'] or sample['state']. "
                "Set datasets.vla_data.include_state: true."
            )

        state_history = np.asarray(state_history, dtype=np.float32)
        if state_history.ndim == 1:
            state_history = state_history[None, :]
        if state_history.ndim != 2:
            raise ValueError(f"state_history must be [num_frames, state_dim], got {state_history.shape}")
        return state_history

    def _example_state_time_offsets(
        self,
        example: dict,
        num_states: int,
    ) -> np.ndarray | None:
        offsets = example.get("state_time_offsets", None)
        if offsets is None:
            if self.state_encoder.time_encoding == "none":
                return None
            raise KeyError("relative_fourier state encoding requires sample['state_time_offsets']")
        offsets = np.asarray(offsets, dtype=np.float32).reshape(-1)
        if offsets.shape[0] != num_states:
            raise ValueError(f"Expected {num_states} state time offsets, got {offsets.shape[0]}")
        return offsets

    def _embedding_layer(self):
        qwen_model = self.qwen_vl_interface.model
        if hasattr(qwen_model, "model") and hasattr(qwen_model.model, "get_input_embeddings"):
            return qwen_model.model.get_input_embeddings()
        return qwen_model.get_input_embeddings()

    def _qwen3_core_model(self):
        qwen_model = self.qwen_vl_interface.model
        if not hasattr(qwen_model, "model") or not hasattr(qwen_model.model, "get_rope_index"):
            raise AttributeError("QwenFastState expects Qwen3VLForConditionalGeneration.model.get_rope_index")
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
        qwen_inputs,
        state_histories: List[np.ndarray],
        state_time_offsets: List[np.ndarray | None] | None = None,
        include_input_ids: bool = False,
    ) -> dict:
        input_ids = qwen_inputs["input_ids"]
        attention_mask = qwen_inputs["attention_mask"]
        labels = qwen_inputs.get("labels", None)
        device = input_ids.device
        embed_layer = self._embedding_layer()
        base_embeds = embed_layer(input_ids)
        batch_size, seq_width, hidden_size = base_embeds.shape
        if len(state_histories) != batch_size:
            raise ValueError(f"Expected {batch_size} state histories, got {len(state_histories)}")
        if state_time_offsets is None:
            state_time_offsets = [None] * batch_size
        if len(state_time_offsets) != batch_size:
            raise ValueError(f"Expected {batch_size} state time-offset arrays, got {len(state_time_offsets)}")

        valid_embeds = []
        valid_input_ids = []
        valid_attention = []
        valid_labels = [] if labels is not None else None

        for batch_idx in range(batch_size):
            seq_len = int(attention_mask[batch_idx].sum().item())
            left_pad = seq_width - seq_len

            embeds_i = base_embeds[batch_idx, left_pad:]
            ids_i = input_ids[batch_idx, left_pad:]
            attn_i = attention_mask[batch_idx, left_pad:]
            if labels is not None:
                labels_i = labels[batch_idx, left_pad:]
                supervised = torch.nonzero(labels_i != IGNORE_INDEX, as_tuple=False).flatten()
                insert_at = int(supervised[0].item()) if supervised.numel() > 0 else seq_len
            else:
                labels_i = None
                insert_at = seq_len

            state_tensor = torch.as_tensor(state_histories[batch_idx], dtype=torch.float32, device=device)
            state_tokens = self.state_encoder(
                state_tensor,
                time_offsets=state_time_offsets[batch_idx],
            ).to(device=device, dtype=base_embeds.dtype)
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
            if key not in {"input_ids", "inputs_embeds", "attention_mask", "labels", "position_ids"}
        }
        prepared["inputs_embeds"] = padded_embeds
        prepared["attention_mask"] = padded_attention
        prepared["position_ids"] = self._build_position_ids(padded_input_ids, padded_attention, qwen_inputs)
        if include_input_ids:
            prepared["input_ids"] = padded_input_ids
        if padded_labels is not None:
            prepared["labels"] = padded_labels
        return prepared

    def _generate_state_conditioned_ids(
        self,
        qwen_inputs: dict,
        max_new_tokens: int,
    ) -> torch.LongTensor:
        """Greedy generation path that preserves state soft-token embeddings.

        Qwen3-VL position ids depend on image-grid tokens. After inserting
        learned state embeddings, cached token-by-token decoding must keep the
        multimodal rope/cache positions exactly aligned. The earlier cached
        path drifted after a few generated tokens and produced invalid FAST
        sequences even when teacher-forced loss was near zero. Recomputing the
        short action response without KV cache is slower, but it keeps the
        position ids consistent and stops once ``<|im_end|>``/EOS is emitted.
        """
        model = self.qwen_vl_interface.model
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        input_ids = qwen_inputs["input_ids"]
        inputs_embeds = qwen_inputs["inputs_embeds"]
        attention_mask = qwen_inputs["attention_mask"]
        device = inputs_embeds.device
        batch_size = inputs_embeds.shape[0]
        embed_layer = self._embedding_layer()

        generation_config = model.generation_config
        pad_token_id = generation_config.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        pad_token_id = int(pad_token_id)

        eos_token_id = generation_config.eos_token_id
        if eos_token_id is None:
            eos_token_ids = None
        elif isinstance(eos_token_id, int):
            eos_token_ids = torch.tensor([eos_token_id], device=device)
        else:
                eos_token_ids = torch.tensor(list(eos_token_id), device=device)

        generation_kwargs = {
            key: value
            for key, value in qwen_inputs.items()
            if key not in {"input_ids", "inputs_embeds", "attention_mask", "position_ids", "labels"}
        }
        generated_ids = input_ids
        unfinished = torch.ones(batch_size, dtype=torch.long, device=device)

        for _ in range(int(max_new_tokens)):
            position_ids = self._build_position_ids(generated_ids, attention_mask, qwen_inputs)
            outputs = model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
                **generation_kwargs,
            )

            next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)

            if eos_token_ids is not None:
                next_tokens = next_tokens * unfinished + pad_token_id * (1 - unfinished)

            generated_ids = torch.cat([generated_ids, next_tokens[:, None]], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)],
                dim=-1,
            )

            if eos_token_ids is not None:
                is_eos = (next_tokens[:, None] == eos_token_ids[None, :]).any(dim=-1)
                unfinished = unfinished & ~is_eos
                if int(unfinished.max().item()) == 0:
                    break

            next_input_ids = next_tokens[:, None]
            next_embeds = embed_layer(next_input_ids).to(device=device, dtype=inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, next_embeds], dim=1)

        return generated_ids

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> dict:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state_histories = [self._example_state_history(example) for example in examples]
        state_time_offsets = [
            self._example_state_time_offsets(example, len(state_history))
            for example, state_history in zip(examples, state_histories, strict=True)
        ]

        batch_fast_tokens = self.action_model.encoder_action2fastoken(actions)
        vlm_action_tokens = [self.map_fast_token_to_vlm_action(fast_tokens) for fast_tokens in batch_fast_tokens]
        solutions = [
            self._build_fast_solution(example, instruction, action_tokens)
            for example, instruction, action_tokens in zip(examples, instructions, vlm_action_tokens)
        ]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions, solutions=solutions
        )
        qwen_inputs = self._prepare_state_conditioned_inputs(
            qwen_inputs,
            state_histories,
            state_time_offsets=state_time_offsets,
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
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state_histories = [self._example_state_history(example) for example in examples]
        state_time_offsets = [
            self._example_state_time_offsets(example, len(state_history))
            for example, state_history in zip(examples, state_histories, strict=True)
        ]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        qwen_inputs = self._prepare_state_conditioned_inputs(
            qwen_inputs,
            state_histories,
            state_time_offsets=state_time_offsets,
            include_input_ids=True,
        )
        max_new_tokens = int(kwargs.get("max_new_tokens", self.config.framework.get("max_new_tokens", 2048)))

        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self._generate_state_conditioned_ids(qwen_inputs, max_new_tokens=max_new_tokens)

        batch_vlm_action_token_ids = self._extract_action_token_ids(generated_ids)
        batch_fast_action_token_idx = self._decode_action_tokens(batch_vlm_action_token_ids)
        self._validate_fast_action_token_sequences(batch_fast_action_token_idx)
        normalized_actions = self.action_model.fast_tokenizer.decode(batch_fast_action_token_idx)
        return {"normalized_actions": normalized_actions}
