# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""
QwenOFTState Framework

OFT action regression with Qwen3 state-history soft tokens. State vectors are
aligned to the historical and current head-camera timesteps, projected by an
MLP, and inserted between the user prompt embeddings and OFT action-query
tokens. Current wrist views do not receive duplicated state tokens.
"""

import time
from typing import List

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenFastState import StateHistoryEncoder
from starVLA.model.framework.VLM4A.QwenOFT import IGNORE_INDEX, Qwenvl_OFT
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register("QwenOFTState")
class Qwenvl_OFT_State(Qwenvl_OFT):
    """OFT MLP action regression conditioned on head-aligned state tokens."""

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        base_vlm = str(self.config.framework.qwenvl.base_vlm)
        if "Qwen3-VL" not in base_vlm:
            raise ValueError(f"QwenOFTState currently supports Qwen3-VL only, got {base_vlm}")

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
                "QwenOFTState requires sample['state_history'] or sample['state']. "
                "Set datasets.vla_data.include_state: true."
            )

        state_history = np.asarray(state_history, dtype=np.float32)
        if state_history.ndim == 1:
            state_history = state_history[None, :]
        if state_history.ndim != 2:
            raise ValueError(f"state_history must be [num_frames, state_dim], got {state_history.shape}")

        expected_states = int(example.get("num_history_frames", 0)) + 1
        if state_history.shape[0] != expected_states:
            raise ValueError(
                f"Expected one state token per head-camera timestep, got {state_history.shape[0]} "
                f"states for {expected_states} head timesteps."
            )
        return state_history

    def _example_state_time_offsets(self, example: dict) -> np.ndarray | None:
        offsets = example.get("state_time_offsets", None)
        if offsets is None:
            if self.state_encoder.time_encoding == "none":
                return None
            raise KeyError("relative_fourier state encoding requires sample['state_time_offsets']")
        offsets = np.asarray(offsets, dtype=np.float32).reshape(-1)
        expected_states = int(example.get("num_history_frames", 0)) + 1
        if offsets.shape[0] != expected_states:
            raise ValueError(f"Expected {expected_states} state time offsets, got {offsets.shape[0]}")
        return offsets

    def _embedding_layer(self):
        qwen_model = self.qwen_vl_interface.model
        if hasattr(qwen_model, "model") and hasattr(qwen_model.model, "get_input_embeddings"):
            return qwen_model.model.get_input_embeddings()
        return qwen_model.get_input_embeddings()

    def _qwen3_core_model(self):
        qwen_model = self.qwen_vl_interface.model
        if not hasattr(qwen_model, "model") or not hasattr(qwen_model.model, "get_rope_index"):
            raise AttributeError("QwenOFTState expects Qwen3VLForConditionalGeneration.model.get_rope_index")
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
        state_time_offsets: List[np.ndarray | None] | None = None,
        insert_before: str = "action",
        insert_indices=None,
    ) -> tuple[dict, torch.Tensor]:
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
            valid_positions = torch.nonzero(attention_mask[batch_idx] != 0, as_tuple=False).flatten()
            seq_len = int(valid_positions.numel())

            embeds_i = base_embeds[batch_idx, valid_positions]
            ids_i = input_ids[batch_idx, valid_positions]
            attn_i = attention_mask[batch_idx, valid_positions]
            labels_i = labels[batch_idx, valid_positions] if labels is not None else None

            if insert_indices is not None:
                insert_at = max(0, min(int(insert_indices[batch_idx]), seq_len))
            elif insert_before == "action":
                action_positions = torch.nonzero(ids_i == self.action_token_id, as_tuple=False).flatten()
                insert_at = int(action_positions[0].item()) if action_positions.numel() > 0 else seq_len
            elif insert_before == "supervised":
                if labels_i is None:
                    insert_at = seq_len
                else:
                    supervised = torch.nonzero(labels_i != IGNORE_INDEX, as_tuple=False).flatten()
                    insert_at = int(supervised[0].item()) if supervised.numel() > 0 else seq_len
            else:
                raise ValueError(f"Invalid insert_before={insert_before!r}; expected 'action' or 'supervised'")

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
                "QwenOFTState VLM cotrain requires batch['state_history'] or batch['state']. "
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

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        collect_timing = bool(getattr(self, "_collect_step_timing", False))
        self.qwen_vl_interface._collect_step_timing = collect_timing
        timing_host = {}
        timing_cuda_events = {}

        def start_cuda_phase():
            if not collect_timing:
                return None
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()
            return events

        def finish_cuda_phase(name, events):
            if events is not None:
                events[1].record()
                timing_cuda_events[name] = events

        batch_images = [example["image"] for example in examples]
        instructions = [self._select_prompt_instruction(example) for example in examples]
        actions = [example["action"] for example in examples]
        state_histories = [self._example_state_history(example) for example in examples]
        state_time_offsets = [self._example_state_time_offsets(example) for example in examples]

        action_tokens = self.action_token * self.chunk_len
        prompt_suffixes = [action_tokens for _ in instructions]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            prompt_suffixes=prompt_suffixes,
        )
        if collect_timing:
            input_timing = getattr(self.qwen_vl_interface, "_last_input_timing", None) or {}
            timing_host.update(input_timing.get("host", {}))
            timing_cuda_events.update(input_timing.get("cuda_events", {}))

        state_started = time.perf_counter() if collect_timing else None
        state_events = start_cuda_phase()
        qwen_inputs, input_ids_for_action = self._prepare_state_conditioned_inputs(
            qwen_inputs,
            state_histories,
            state_time_offsets=state_time_offsets,
        )
        finish_cuda_phase("timing/state_conditioning_gpu", state_events)
        if collect_timing:
            timing_host["timing/state_conditioning_host"] = time.perf_counter() - state_started

        vlm_started = time.perf_counter() if collect_timing else None
        vlm_events = start_cuda_phase()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
        finish_cuda_phase("timing/vlm_forward_gpu", vlm_events)
        if collect_timing:
            timing_host["timing/vlm_forward_host"] = time.perf_counter() - vlm_started

        action_started = time.perf_counter() if collect_timing else None
        action_events = start_cuda_phase()
        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._gather_action_token_embeddings(
                last_hidden,
                input_ids_for_action,
                action_token_id=self.action_token_id,
            )
            pred_actions = self.action_model.predict_action(action_queries)
            actions = torch.tensor(np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype)
            actions_target = actions[:, -self.action_horizon :, :]
            action_loss = self.l1_loss(pred_actions, actions_target)
        finish_cuda_phase("timing/action_head_gpu", action_events)
        if collect_timing:
            timing_host["timing/action_head_host"] = time.perf_counter() - action_started

        output = {"action_loss": action_loss}
        if collect_timing:
            output["_timing_host"] = timing_host
            output["_timing_cuda_events"] = timing_cuda_events
        return output

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs: str) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [self._select_prompt_instruction(example) for example in examples]
        return_vlm_text = bool(kwargs.get("return_vlm_text", False))
        vlm_text_instructions = [
            self._select_vlm_text_instruction(example, fallback=instructions[idx])
            for idx, example in enumerate(examples)
        ]
        state_histories = [self._example_state_history(example) for example in examples]
        state_time_offsets = [self._example_state_time_offsets(example) for example in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        action_tokens = self.action_token * self.chunk_len
        prompt_suffixes = [action_tokens for _ in instructions]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            prompt_suffixes=prompt_suffixes,
        )
        qwen_inputs, input_ids_for_action = self._prepare_state_conditioned_inputs(
            qwen_inputs,
            state_histories,
            state_time_offsets=state_time_offsets,
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._gather_action_token_embeddings(
                last_hidden,
                input_ids_for_action,
                action_token_id=self.action_token_id,
            )
            pred_actions = self.action_model.predict_action(action_queries)

        ret = {"normalized_actions": pred_actions.detach().cpu().numpy()}
        if return_vlm_text:
            ret["vlm_text"] = self._generate_vlm_text(
                batch_images,
                vlm_text_instructions,
                max_new_tokens=int(kwargs.get("vlm_text_max_new_tokens", 128)),
            )
        return ret
