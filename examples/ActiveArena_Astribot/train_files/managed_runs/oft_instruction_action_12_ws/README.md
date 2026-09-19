# oft_instruction_action_12_ws

This managed run trains `QwenOFTState` on Astribot LeRobot data with:

- VLA action prompt: task instruction (`lang`)
- VLM cotrain branch: enabled, with think output supervised by task instruction
- history: `action_keyframe`, max 12 history frames
- state input: enabled (`include_state: true`), one MLP-projected soft token per input image

Launch:

```bash
bash examples/ActiveArena_Astribot/train_files/managed_runs/oft_instruction_action_12_ws/run_train.sh
```

Inference server:

```bash
bash examples/ActiveArena_Astribot/train_files/managed_runs/oft_instruction_action_12_ws/run_policy_server.sh
```

Outputs are written to `results/Checkpoints/oft_instruction_action_12_ws`.
