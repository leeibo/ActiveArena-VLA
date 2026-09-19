# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""Fast Action Tokenizer Adapter
"this file is adapted from https://huggingface.co/physical-intelligence/fast"

Overview:
    This module encapsulates a lightweight "action → language model-readable sequence" converter (Fast_Action_Tokenizer).
    Its core objective is to convert continuous/discrete raw robot actions (raw_actions) into
    pseudo-natural language token strings like <robot_action_12><robot_action_3><robot_action_87> ...
    This facilitates direct integration into multimodal large models (VLM/LLM) dialogue templates,
    leveraging their language modeling capabilities for action prediction.
"""

import json
import os
import importlib.util
from pathlib import Path

import numpy as np
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import PreTrainedTokenizerFast

DEFAULT_FAST_TOKENIZER_PATH = "./playground/Pretrained_models/fast"
FAST_HF_REPO_ID = "physical-intelligence/fast"


def _load_local_fast_processor(local_dir: str | os.PathLike):
    """Load the FAST UniversalActionProcessor with compatibility for transformers >= 5.x.

    transformers 5.x changed AutoProcessor internals which breaks the default
    loading path for the physical-intelligence/fast custom processor. This
    helper manually loads the custom class and its BPE tokenizer component.
    """
    local_dir = Path(local_dir).expanduser()
    required_files = ["processing_action_tokenizer.py", "tokenizer.json", "processor_config.json"]
    missing_files = [name for name in required_files if not (local_dir / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"FAST tokenizer dir {local_dir} is missing files: {missing_files}")

    spec = importlib.util.spec_from_file_location(
        "processing_action_tokenizer",
        local_dir / "processing_action_tokenizer.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load processing_action_tokenizer.py from {local_dir}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    UniversalActionProcessor = mod.UniversalActionProcessor

    bpe_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(local_dir / "tokenizer.json"),
        clean_up_tokenization_spaces=False,
    )

    with open(local_dir / "processor_config.json", "r") as f:
        cfg = json.load(f)

    processor = UniversalActionProcessor(
        bpe_tokenizer=bpe_tokenizer,
        scale=cfg.get("scale", 10),
        vocab_size=cfg.get("vocab_size", 2048),
        min_token=cfg.get("min_token", -354),
        action_dim=cfg.get("action_dim"),
        time_horizon=cfg.get("time_horizon"),
    )
    return processor


def _local_fast_tokenizer_candidates(pretrained_path: str):
    for env_name in ("STARVLA_FAST_TOKENIZER", "FAST_TOKENIZER_PATH"):
        env_value = os.getenv(env_name)
        if env_value:
            yield env_value

    yield pretrained_path

    if pretrained_path in {DEFAULT_FAST_TOKENIZER_PATH, FAST_HF_REPO_ID}:
        try:
            yield snapshot_download(FAST_HF_REPO_ID, local_files_only=True)
        except Exception:
            return


def _load_fast_processor(pretrained_path: str = DEFAULT_FAST_TOKENIZER_PATH):
    """Load FAST tokenizer from local files only.

    Batch jobs usually run without internet access, so the default path must be
    a local mirror or a locally cached Hugging Face snapshot.
    """
    for candidate in _local_fast_tokenizer_candidates(pretrained_path):
        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists():
            return _load_local_fast_processor(candidate_path)

    raise FileNotFoundError(
        "FAST tokenizer files were not found locally. Create "
        f"{DEFAULT_FAST_TOKENIZER_PATH} with physical-intelligence/fast files, "
        "or set framework.action_model.fast_tokenizer_name / STARVLA_FAST_TOKENIZER "
        "to a local directory containing processing_action_tokenizer.py, tokenizer.json, "
        "and processor_config.json. The training launcher will not download this tokenizer "
        "from Hugging Face during offline batch jobs."
    )


class Fast_Action_Tokenizer(nn.Module):
    """One MLP ResNet block with a residual connection."""

    def __init__(self, fast_tokenizer_name=DEFAULT_FAST_TOKENIZER_PATH):
        super().__init__()

        self.fast_tokenizer = _load_fast_processor(
            fast_tokenizer_name
        )  # load https://huggingface.co/physical-intelligence/fast

    def encoder_action2fastoken(self, raw_actions):
        # x: (batch_size, chunck, dim)
        batch_actions = np.stack(raw_actions, axis=0)  # (B, T, D)
        batch_fast_tokens = self.fast_tokenizer(batch_actions)

        return batch_fast_tokens  # List[str]

    def decoder_action(self, generated_ids):
        # api https://huggingface.co/physical-intelligence/fast
        # return: (batch_size, chunck, dim)
        pred_actions = self.fast_tokenizer.decode([generated_ids - self._ACTION_TOKEN_MIN])
        return pred_actions

    def fit_tokenizer_on_datasets(
        self,
        action_dataset,
        datasets_path="<your_local_path>",
    ):
        # If datasets_path exists, load directly
        if os.path.exists(datasets_path):

            self.fast_tokenizer = AutoProcessor.from_pretrained(datasets_path, trust_remote_code=True)
            return
        else:
            # If not found, Fit the tokenizer on the new dataset
            new_tokenizer = self.fast_tokenizer.tokenizer.fit(action_dataset)
            self.fast_tokenizer = new_tokenizer

            # Save the new tokenizer, optionally push it to the Hugging Face model hub
            self.fast_tokenizer.save_pretrained(datasets_path)


def get_action_model(config=None):
    """
    Factory: build ActionModel from global framework config.

    Args:
        config: Global config (expects config.framework.action_model namespace).
    Returns:
        ActionModel: Initialized diffusion action head.
    """
    action_cfg = config.framework.action_model if config is not None else {}
    fast_tokenizer_name = action_cfg.get(
        "fast_tokenizer_name",
        action_cfg.get("fast_tokenizer_path", DEFAULT_FAST_TOKENIZER_PATH),
    )
    action_model = Fast_Action_Tokenizer(fast_tokenizer_name=fast_tokenizer_name)

    return action_model


def start_debugpy_once():
    """start debugpy once"""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10094))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10094 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        start_debugpy_once()

    fast_tokenizer_name = "physical-intelligence/fast"
    fast_tokenizer = Fast_Action_Tokenizer(fast_tokenizer_name=fast_tokenizer_name)
    raw_actions = [np.random.randn(16, 7), np.random.randn(16, 7)]

    tokenizer = _load_fast_processor(fast_tokenizer_name)

    action_data = np.random.rand(2, 16, 7)
    tokens = tokenizer(action_data)
    decoded_actions = tokenizer.decode(tokens)

    # self func test
    vlm_tokens = fast_tokenizer.encoder_action2fastoken(raw_actions)
    print(vlm_tokens)
    pred_actions = fast_tokenizer.decoder_action(np.array([12, 3, 45, 87]))
    print(pred_actions)
