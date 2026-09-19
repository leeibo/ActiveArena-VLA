# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""
QwenGR00TDual Framework

Dual-DiT variant of QwenGR00T for Astribot-style actions whose first 16
dimensions control the arms and final 2 dimensions control the camera.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class QwenGR00TDualDefaultConfig(QwenGR00TDefaultConfig):
    """QwenGR00T with separate arm and camera GR00T DiT heads."""

    name: str = "QwenGR00TDual"
    dual_action_model: dict = field(
        default_factory=lambda: {
            "arm_action_dim": 16,
            "camera_action_dim": 2,
            "arm_loss_weight": 1.0,
            "camera_loss_weight": 1.0,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenGR00TDual")
class Qwen_GR00T_Dual(baseframework):
    """GR00T framework with two independent flow-matching DiT heads."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenGR00TDualDefaultConfig, config)

        self.total_action_dim = int(self.config.framework.action_model.action_dim)
        self.arm_action_dim = int(self._dual_cfg_get("arm_action_dim", 16))
        self.camera_action_dim = int(self._dual_cfg_get("camera_action_dim", 2))
        self.arm_loss_weight = float(self._dual_cfg_get("arm_loss_weight", 1.0))
        self.camera_loss_weight = float(self._dual_cfg_get("camera_loss_weight", 1.0))
        self.loss_normalizer = (
            self.arm_loss_weight * self.arm_action_dim
            + self.camera_loss_weight * self.camera_action_dim
        )

        if self.arm_action_dim <= 0 or self.camera_action_dim <= 0:
            raise ValueError("QwenGR00TDual requires positive arm_action_dim and camera_action_dim.")
        if self.arm_loss_weight < 0 or self.camera_loss_weight < 0:
            raise ValueError("QwenGR00TDual requires non-negative loss weights.")
        if self.arm_action_dim + self.camera_action_dim != self.total_action_dim:
            raise ValueError(
                "QwenGR00TDual action split mismatch: "
                f"arm_action_dim({self.arm_action_dim}) + camera_action_dim({self.camera_action_dim}) "
                f"!= action_dim({self.total_action_dim})."
            )
        if self.loss_normalizer <= 0:
            raise ValueError("QwenGR00TDual requires a positive weighted loss normalizer.")

        self.config.framework.action_model.state_dim = 0
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = hidden_size
        self.action_model = nn.ModuleDict(
            {
                "arm": get_action_model(config=self._head_config(self.arm_action_dim, hidden_size)),
                "camera": get_action_model(config=self._head_config(self.camera_action_dim, hidden_size)),
            }
        )
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def _dual_cfg_get(self, key: str, default=None):
        dual_cfg = self.config.framework.get("dual_action_model", None)
        if dual_cfg is not None:
            value = dual_cfg.get(key, None) if hasattr(dual_cfg, "get") else getattr(dual_cfg, key, None)
            if value is not None:
                return value
        action_cfg = self.config.framework.action_model
        nested_cfg = action_cfg.get("dual_head", None) if hasattr(action_cfg, "get") else None
        if nested_cfg is not None:
            value = nested_cfg.get(key, None) if hasattr(nested_cfg, "get") else getattr(nested_cfg, key, None)
            if value is not None:
                return value
        return default

    def _head_config(self, action_dim: int, hidden_size: int):
        source_cfg = self.config.unwrap() if hasattr(self.config, "unwrap") else self.config
        head_cfg = OmegaConf.create(OmegaConf.to_container(source_cfg, resolve=True))
        head_cfg.framework.action_model.action_dim = int(action_dim)
        head_cfg.framework.action_model.state_dim = 0
        head_cfg.framework.action_model.diffusion_model_cfg.cross_attention_dim = int(hidden_size)
        return head_cfg

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

    def _encode_hidden(self, batch_images: List, instructions: List[str]) -> tuple[torch.Tensor, torch.Tensor | None]:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
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

        last_hidden, backbone_attention_mask = self._encode_hidden(batch_images, instructions)

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]
            if actions_target.shape[-1] != self.total_action_dim:
                raise ValueError(
                    f"Expected action dim {self.total_action_dim}, got {actions_target.shape[-1]}."
                )

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

            arm_actions = actions_target_repeated[:, :, : self.arm_action_dim]
            camera_actions = actions_target_repeated[:, :, self.arm_action_dim :]
            arm_loss = self.action_model["arm"](
                last_hidden_repeated,
                arm_actions,
                None,
                encoder_attention_mask=backbone_attention_mask,
            )
            camera_loss = self.action_model["camera"](
                last_hidden_repeated,
                camera_actions,
                None,
                encoder_attention_mask=backbone_attention_mask,
            )
            action_loss = (
                self.arm_loss_weight * self.arm_action_dim * arm_loss
                + self.camera_loss_weight * self.camera_action_dim * camera_loss
            ) / self.loss_normalizer

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs: str) -> dict:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [self._select_prompt_instruction(example) for example in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        last_hidden, backbone_attention_mask = self._encode_hidden(batch_images, instructions)
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)

        with torch.autocast("cuda", dtype=torch.float32):
            pred_arm = self.action_model["arm"].predict_action(
                last_hidden,
                None,
                encoder_attention_mask=backbone_attention_mask,
            )
            pred_camera = self.action_model["camera"].predict_action(
                last_hidden,
                None,
                encoder_attention_mask=backbone_attention_mask,
            )
            pred_actions = torch.cat([pred_arm, pred_camera], dim=-1)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}
