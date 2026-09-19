"""Qwen planner-only VLM framework.

This keeps planner training separate from OFT action heads.  It exposes only a
Qwen-VL interface, so VLM-only trainers can optimize language-generation loss
without allocating unused action modules.
"""

from dataclasses import dataclass, field
from typing import Optional

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class QwenPlannerVLMDefaultConfig:
    name: str = "QwenPlannerVLM"
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-2B-Instruct",
            "attn_implementation": "flash_attention_2",
        }
    )


@FRAMEWORK_REGISTRY.register("QwenPlannerVLM")
class QwenPlannerVLM(baseframework):
    """Planner-only Qwen-VL wrapper for supervised subtask/retrieval generation."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenPlannerVLMDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
