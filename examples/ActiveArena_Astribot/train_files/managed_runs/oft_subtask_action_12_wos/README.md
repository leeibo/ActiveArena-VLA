# oft_subtask_action_12_wos

This managed run trains `QwenOFT` on Astribot LeRobot data with:

- VLA action prompt: task instruction (`lang`)
- VLM cotrain branch: enabled, with think output supervised by subtask instruction
- history: `action_keyframe`, max 12 history frames
- state input: disabled (`include_state: false`)

Launch:

```bash
bash examples/ActiveArena_Astribot/train_files/managed_runs/oft_subtask_action_12_wos/run_train.sh
```

Submit:

```bash
yhbatch examples/ActiveArena_Astribot/train_files/managed_runs/oft_subtask_action_12_wos/submit_yhbatch.sh
```

Inference server:

```bash
bash examples/ActiveArena_Astribot/train_files/managed_runs/oft_subtask_action_12_wos/run_policy_server.sh
```

Outputs are written to `results/Checkpoints/oft_subtask_action_12_wos`.
