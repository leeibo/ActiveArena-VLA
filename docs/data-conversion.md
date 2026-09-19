# Convert ActiveArena demonstrations for training

`tools/convert_astribot_to_lerobot.py` converts collected Astribot HDF5 episodes
and their language metadata into the LeRobot v2.1 image layout consumed by the
three released training recipes. This release tool follows the existing
Astribot dataset's frozen field contract and ActiveArena next-state targets. The original
Astribot conversion script was not present in the source checkouts; this is a
new, portable implementation validated on a real collected episode.

## Install the CPU conversion dependencies

Conversion needs no simulator, GPU, or LeRobot package. It may run in a separate
Python 3.10+ environment:

```bash
pip install -r tools/requirements-data.txt
```

The conversion smoke run used NumPy 1.26.4, h5py 3.13.0, PyArrow 20.0.0,
Pillow 11.2.1, and opencv-python-headless 4.11.0.86. The output was also prepared for
the training environment's PyArrow 14.0.1 reader.

The supplied presentation sample uses the same contract. Its exact collected
directory is `info_gathering_demo_ppt_samples__info_gathering_demo_ppt_external_views`;
pass that full name when converting it. The regular collector uses the shorter
`info_gathering_demo__info_gathering_demo` directory shown below.

## Input and output

The collector creates this layout:

```text
ActiveArena/data/
└── beat_block_hammer_rotate_view/
    └── info_gathering_demo__info_gathering_demo/
        ├── data/episode0.hdf5
        └── subtask_metadata/episode0.json
```

`--raw-root` is the directory containing task directories. `--config` is the
exact directory below each task, including the `__difficulty_tag` suffix.
`--tasks` restricts the selected tasks; by default every task directory under
the supplied root is selected. Existing output datasets are never overwritten.

From the ActiveArena-VLA repository, validate one episode without writing:

```bash
python tools/convert_astribot_to_lerobot.py \
  --raw-root ../ActiveArena/data \
  --output-root playground/dataset/ActiveArena_Astribot_lerobot \
  --tasks beat_block_hammer_rotate_view \
  --max-episodes 1 --dry-run
```

To validate the supplied presentation sample directly:

```bash
python tools/convert_astribot_to_lerobot.py \
  --raw-root ../ActiveArena/data_ppt_samples \
  --output-root /tmp/activearena-ppt-lerobot \
  --tasks beat_block_hammer_rotate_view \
  --config info_gathering_demo_ppt_samples__info_gathering_demo_ppt_external_views \
  --max-episodes 1 --dry-run
```

For the full ID training collection:

```bash
python tools/convert_astribot_to_lerobot.py \
  --raw-root ../ActiveArena/data \
  --output-root playground/dataset/ActiveArena_Astribot_lerobot \
  --config info_gathering_demo__info_gathering_demo \
  --fps 15
```

Do not mix ID training demonstrations with the fixed evaluation episodes. Use a
separate output root for randomized/evaluation collections. To convert a custom
collection, pass its full `--config` directory name. The converter sorts episode
numbers numerically and renumbers output episodes contiguously.

Each output task contains `data/chunk-000/episode_000000.parquet` and `meta/` with
`info.json`, `modality.json`, `episodes.jsonl`, `episodes_stats.jsonl`,
`tasks.jsonl`, `astribot_subtask_metadata.json`, and `conversion_manifest.json`.
The head-camera RGB image is embedded as lossless PNG bytes in each parquet row.
No wrist or observer camera is supplied to the policy.

## Action and annotation contract

The model has 18 state/action coordinates in this order:

```text
left_arm_0..6, left_gripper, right_arm_0..6, right_gripper, torso_yaw, head_2
```

ActiveArena's raw `joint_action/vector` has 19 coordinates and a different final
order. The converter reads the named joint streams, drops the fixed `head_1`
coordinate, then orders `torso_yaw` before `head_2`. It rejects an episode with a
moving `head_1`. The current joint vector is the state and the next raw joint
vector is the absolute action target. An N-frame raw trajectory produces N−1
training rows; the final image has no next-state target and is dropped.

`task_instruction` and `subtask_instruction_map` come from the paired JSON.
`subtask`, `subtask_instruction_idx`, and `stage` are required HDF5 fields and are
preserved as `subtask_index`, `subtask_instruction_index`, and `stage`. Optional
`subtask_keyframe` and `motion_keyframe` arrays are copied when present. The
released `action_keyframe` recipes use a causal fixed stride of 16 and up to 12
history frames; no separately computed keyframe column is required for them.

The default `--jpeg-encoding activearena` mirrors the repository's storage path:
SAPIEN RGB arrays are sent directly to `cv2.imencode`, and `cv2.imdecode` recovers
those arrays. The converter therefore does not swap channels again. For HDF5
from another producer that stores conventional JPEG RGB, explicitly use
`--jpeg-encoding standard-jpeg` and verify the result visually.

## Start training after conversion

The default 35-task registry expects all task folders. For a single hammer
smoke dataset, use its existing `activearena_astribot_task1` mixture instead:

```bash
NUM_PROCESSES=1 GPU_IDS=0 \
  bash examples/ActiveArena_Astribot/train_files/managed_runs/oft_subtask_action_12_ws/run_train.sh \
  datasets.vla_data.data_mix=activearena_astribot_task1 \
  trainer.max_train_steps=2
```

This command still requires the base Qwen3-VL model and the training environment
in [reproduction.md](reproduction.md). A successful conversion is not evidence
that a 100,000-step training job has been reproduced.

## Release verification

The converter was run on the real 80-frame `beat_block_hammer_rotate_view` demo
from the ActiveArena presentation sample. It produced 79 transitions. Every
18-dimensional state and next-state action matched the raw named streams; all
79 PNG images matched the raw camera's decoded pixels; every subtask index
resolved to text. No full training job or GPU simulator evaluation was run as
part of this release preparation.

The ActiveArena-VLA `LeRobotSingleDataset` loader was also exercised in the
existing Python 3.10 / PyArrow 14.0.1 training environment. It read all 79 rows
and returned a sample with action shape `(16, 18)`, state shape `(1, 18)`, three
224×224 history/current images, state history shape `(3, 18)`, and the expected
task/subtask text. The generic statistics scanner skipped image-valued columns
with its existing warning; state/action statistics and sample loading succeeded.

The reusable CPU regression checks cover next-state alignment, the 18-D joint
mapping, RGB channel preservation, subtask lookup, refusal to overwrite output,
missing language labels, and rejection of a moving omitted head joint:

```bash
python -m unittest discover -s tests -p test_astribot_converter.py
```

All three regression tests passed during release preparation.
