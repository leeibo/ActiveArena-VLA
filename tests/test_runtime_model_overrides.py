"""Regression tests for ActiveArena runtime model path overrides."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from starVLA.model.framework import base_framework


class RuntimeModelOverrideTests(unittest.TestCase):
    def test_build_framework_applies_base_vlm_override(self):
        cfg = SimpleNamespace(
            framework=SimpleNamespace(
                name="_activearena_test_framework",
                qwenvl=SimpleNamespace(base_vlm="checkpoint-relative-path"),
            )
        )
        seen = {}

        def fake_model(received_cfg):
            seen["base_vlm"] = received_cfg.framework.qwenvl.base_vlm
            return received_cfg

        registry = base_framework.FRAMEWORK_REGISTRY._registry
        previous = registry.get("_activearena_test_framework")
        registry["_activearena_test_framework"] = fake_model
        try:
            with patch.dict(os.environ, {"ACTIVEARENA_VLA_BASE_VLM": "/models/qwen3"}, clear=False), patch.object(
                base_framework, "_auto_import_framework_modules"
            ):
                base_framework.build_framework(cfg)
        finally:
            if previous is None:
                registry.pop("_activearena_test_framework", None)
            else:
                registry["_activearena_test_framework"] = previous

        self.assertEqual(seen["base_vlm"], "/models/qwen3")


if __name__ == "__main__":
    unittest.main()
