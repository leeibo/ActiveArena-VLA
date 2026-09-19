import json
from pathlib import Path

import numpy as np

from examples.RoboTwin_Memory.tools.convert_test_data_to_lerobot import (
    align_state_action,
    discover_jobs,
    modality_metadata,
    resolve_episode_instruction,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_discovery_sorts_episode_numbers_and_keeps_domain_layout(tmp_path):
    raw_root = tmp_path / "test_data"
    data_dir = raw_root / "pick_item" / "demo_clean" / "data"
    data_dir.mkdir(parents=True)
    for name in ["episode10.hdf5", "episode2.hdf5", "episode1.hdf5"]:
        (data_dir / name).touch()
    _write_json(data_dir.parent / "scene_info.json", {})

    jobs = discover_jobs(raw_root, tmp_path / "output", ["clean"])

    assert len(jobs) == 1
    assert [path.name for path in jobs[0].hdf5_files] == [
        "episode1.hdf5",
        "episode2.hdf5",
        "episode10.hdf5",
    ]
    assert jobs[0].output_dir == tmp_path / "output" / "Clean" / "pick_item"


def test_instruction_resolution_is_deterministic_and_expands_scene_placeholders(tmp_path):
    robotwin_root = tmp_path / "RoboTwin"
    raw_domain_dir = tmp_path / "test_data" / "pick_item" / "demo_randomized"
    _write_json(
        robotwin_root / "description" / "task_instruction" / "pick_item.json",
        {"seen": ["Pick {A} with {a}.", "Use {a} to lift {A}."]},
    )
    _write_json(
        robotwin_root / "description" / "objects_description" / "items" / "red.json",
        {"seen": ["red object", "scarlet item"], "unseen": ["crimson object"]},
    )
    _write_json(
        raw_domain_dir / "scene_info.json",
        {"episode_7": {"info": {"{A}": "items/red", "{a}": "left"}}},
    )

    first = resolve_episode_instruction(
        "pick_item", "randomized", 7, raw_domain_dir, robotwin_root, seed=123
    )
    second = resolve_episode_instruction(
        "pick_item", "randomized", 7, raw_domain_dir, robotwin_root, seed=123
    )

    assert first == second
    assert "{" not in first
    assert "left arm" in first
    assert "object" in first or "item" in first


def test_state_action_alignment_uses_next_joint_vector_as_action():
    trajectory = np.arange(4 * 14, dtype=np.float64).reshape(4, 14)

    state, action = align_state_action(trajectory)

    np.testing.assert_array_equal(state, trajectory[:-1].astype(np.float32))
    np.testing.assert_array_equal(action, trajectory[1:].astype(np.float32))


def test_modality_metadata_matches_robotwin_registry_slices():
    metadata = modality_metadata()

    assert metadata["state"]["left_joints"] == {
        "start": 0,
        "end": 6,
        "original_key": "observation.state",
    }
    assert metadata["action"]["right_gripper"]["start"] == 13
    assert metadata["video"]["cam_left_wrist"]["original_key"] == "observation.images.cam_left_wrist"
