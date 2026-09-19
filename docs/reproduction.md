# ActiveArena-VLA reproduction

ActiveArena-VLA contains the StarVLA-derived training and policy-server code
used by ActiveArena. The ActiveArena simulator snapshot and fixed-seed evaluation launcher
live in the sibling `ActiveArena` repository.

## Environment

Use Python 3.10 or newer with a CUDA-enabled PyTorch build, then install the
runtime dependencies:

```bash
conda env create -f environment.yml
conda activate activearena-vla
pip install -r requirements.txt
pip install -e .
```

Download the Qwen3-VL-2B-Instruct base model into
`playground/Pretrained_models/ActiveArena/Qwen3-VL-2B-Instruct`, or point
`ACTIVEARENA_VLA_BASE_VLM` (legacy: `STARVLA_BASE_VLM`) to an equivalent local directory. The three released OFT
bundles use continuous MLP action heads and do not require the FAST tokenizer.

## Released checkpoint bundles

`ActiveArena-VLA-weights` stores one complete bundle per model. Each bundle
contains `config.yaml`, `config.full.yaml`, `dataset_statistics.json`, and
`checkpoints/steps_100000_pytorch_model.pt`. Verify the download before use:

```bash
cd /path/to/ActiveArena-VLA-weights
./verify.sh
```

Start a policy server from this repository with a selected bundle:

```bash
MODEL=oft_subtask_action_12_ws
export ACTIVEARENA_VLA_ROOT=$PWD
export ACTIVEARENA_VLA_WEIGHTS_ROOT=/path/to/ActiveArena-VLA-weights
export ACTIVEARENA_VLA_PYTHON=$(command -v python)
bash examples/ActiveArena_Astribot/train_files/managed_runs/$MODEL/run_policy_server.sh
```

`POLICY_CKPT_PATH`, `POLICY_PORT`, `POLICY_GPU_ID`, `USE_BF16`, and
`IDLE_TIMEOUT` are supported overrides. The server speaks the websocket policy
protocol consumed by `ActiveArena/policy/activearena_astribot`.

## Training data and launch

Convert the collected HDF5 episodes with
[`tools/convert_astribot_to_lerobot.py`](../tools/convert_astribot_to_lerobot.py).
The full commands and field contract are in
[the data conversion guide](data-conversion.md). The output root defaults in the
managed YAML files to `playground/dataset/ActiveArena_Astribot_lerobot`; all 35 task
folders are required for the full `activearena_astribot` mixture.

```bash
pip install -r tools/requirements-data.txt
python tools/convert_astribot_to_lerobot.py \
  --raw-root ../ActiveArena/data \
  --output-root playground/dataset/ActiveArena_Astribot_lerobot \
  --config info_gathering_demo__info_gathering_demo
GPU_IDS=0,1,2,3 NUM_PROCESSES=4 \
  bash examples/ActiveArena_Astribot/train_files/managed_runs/oft_subtask_action_12_ws/run_train.sh
```

Outputs are stored under `results/Checkpoints/<model>` and can be served by
setting `RUN_OUTPUT` on the matching policy-server script. Conversion and
loader checks are documented in [data-conversion.md](data-conversion.md); the full 100,000-step training run and
GPU simulator evaluation have not been rerun as part of release preparation.
