# ActiveArena-VLA reproducibility contract

ActiveArena-VLA is a self-contained release snapshot. The model implementation
is vendored under `starVLA/`; training and serving never clone or update an
upstream repository at runtime. Keep this source tree, the checkpoint bundle,
and the base-model snapshot together when reproducing results.

## Runtime names

The canonical policy environment is `activearena-vla`. Launchers use these
variables:

- `ACTIVEARENA_VLA_ROOT` — this source tree;
- `ACTIVEARENA_VLA_WEIGHTS_ROOT` — the checkpoint bundle;
- `ACTIVEARENA_VLA_PYTHON` — the policy Python executable;
- `ACTIVEARENA_VLA_ENV_NAME` — the conda environment name;
- `ACTIVEARENA_VLA_BASE_VLM` — the local Qwen3-VL snapshot.

The old `STARVLA_REPO_ROOT`, `STARVLA_WEIGHTS_ROOT`, `STARVLA_PYTHON`,
`CONDA_ENV_NAME`, and `STARVLA_BASE_VLM` variables remain accepted as
compatibility fallbacks. New launch scripts and documentation use the
ActiveArena names.

## Model assets

The released configs refer to these local snapshots:

```text
playground/Pretrained_models/ActiveArena/Qwen3-VL-2B-Instruct
playground/Pretrained_models/ActiveArena/Qwen3-VL-2B-Instruct-Action  # FAST recipes
```

The released OFT checkpoints use `Qwen/Qwen3-VL-2B-Instruct` at Hugging Face
commit `89644892e4d85e24eaac8bacfd4f463576704203` (2025-10-23). The same
revision is recorded in `ActiveArena-VLA-weights/manifest.json` and in the
three managed training configs. The FAST example requires its own action-token
snapshot; its model-host revision must be recorded before using that example.

The Qwen base models are not redistributed by this repository. Download each
model once, record the exact Hugging Face commit used by your release, and keep
`HF_HUB_OFFLINE=1` during evaluation. A local directory is preferred over a
mutable model identifier. If a checkpoint config still names the old
`playground/Pretrained_models/Qwen3-VL-*` path, the loader checks the canonical
ActiveArena path and vice versa.

Before publishing a model bundle, record its model-host repository and commit in
this file and add a SHA-256 manifest for the local model directory. The three
ActiveArena checkpoint bundles have their own manifest in
`../ActiveArena-VLA-weights/manifest.json`.

## Source and dependency freeze

The vendored model code was captured from `leeibo/starVLA-A@054a58e901c3be6d4fec32ae5fcea2fe5fe350ff`.
That commit is provenance only: the public checkout does not pull it during
installation or evaluation.

The `starVLA/` package name is intentionally retained for Python import and
checkpoint compatibility; it does not imply a runtime dependency on an
upstream checkout. Install the declared requirements through
`environment.yml`, and use the pinned `torch==2.6.0` /
`torchvision==0.21.0` pair. Flash-attention is optional for serving and must
be built against that pair when enabled for training. For a byte-for-byte
environment lock, export the resolved pip versions after environment creation
and keep that lock beside the release; the source tree itself never updates
dependencies automatically.
