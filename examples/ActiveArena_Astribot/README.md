# ActiveArena Astribot

This example contains the three Astribot training configurations used by the
ActiveArena policy adapters. The directory, registry, mixtures, and dataset
root are frozen release names and do not depend on a moving simulator checkout:

- environment: `activearena-vla`
- robot type: `activearena_astribot`
- dataset root: `playground/dataset/ActiveArena_Astribot_lerobot`

- `train_files/managed_runs/oft_instruction_action_12_ws/`
- `train_files/managed_runs/oft_subtask_action_12_wos/`
- `train_files/managed_runs/oft_subtask_action_12_ws/`

Each run directory contains its training config, launch scripts, and policy
server script. The shared Astribot data registry is in
`train_files/data_registry/data_config.py`.

Run training from the repository root with, for example:

```bash
bash examples/ActiveArena_Astribot/train_files/managed_runs/oft_subtask_action_12_ws/run_train.sh
```

To launch the corresponding policy server:

```bash
bash examples/ActiveArena_Astribot/train_files/managed_runs/oft_subtask_action_12_ws/run_policy_server.sh
```
