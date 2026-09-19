"""Run with python -m unittest discover -s tests -p test_astribot_converter.py.

Requires tools/requirements-data.txt, not model or simulator dependencies.
"""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from convert_astribot_to_lerobot import astribot_state, convert_task, discover_episodes, read_episode


class AstribotConverterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        config = self.raw / "test_task" / "demo"
        (config / "data").mkdir(parents=True)
        (config / "subtask_metadata").mkdir()
        self.source = config / "data/episode0.hdf5"
        self.expected = np.arange(4 * 18, dtype=np.float32).reshape(4, 18) / 100
        rgb = np.zeros((12, 16, 3), dtype=np.uint8)
        rgb[:] = [210, 40, 15]
        ok, jpeg = cv2.imencode(".jpg", rgb)
        assert ok
        self.decoded_rgb = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        with h5py.File(self.source, "w") as f:
            for key, values in {
                "left_arm": self.expected[:, :7], "left_gripper": self.expected[:, 7],
                "right_arm": self.expected[:, 8:15], "right_gripper": self.expected[:, 15],
                "torso": self.expected[:, 16:17], "head": np.column_stack([np.zeros(4), self.expected[:, 17]]),
            }.items():
                f.create_dataset(f"joint_action/{key}", data=values)
            f.create_dataset("observation/camera_head/rgb", data=np.array([jpeg.tobytes()] * 4, dtype=f'S{len(jpeg)}'))
            f.create_dataset("subtask", data=[1, 1, 2, 2])
            f.create_dataset("subtask_instruction_idx", data=[1, 1, 2, 2])
            f.create_dataset("stage", data=[1, 2, 3, 3])
        (config / "subtask_metadata/episode0.json").write_text(json.dumps({
            "task_instruction": "Find and pick the block",
            "subtask_instruction_map": {"1": "find the block", "2": "pick the block"},
        }))
        self.items = discover_episodes(self.raw, "demo", ["test_task"], None)

    def tearDown(self):
        self.temp.cleanup()

    def test_next_state_actions_camera_channels_and_subtask_text(self):
        output = self.root / "converted/test_task"
        convert_task(self.items, output, self.raw, 15, "robotwin")
        rows = pq.read_table(output / "data/chunk-000/episode_000000.parquet").to_pylist()
        self.assertEqual(len(rows), 3)
        np.testing.assert_array_equal([r["observation.state"] for r in rows], self.expected[:-1])
        np.testing.assert_array_equal([r["action"] for r in rows], self.expected[1:])
        for row in rows:
            image = Image.open(io.BytesIO(row["observation.images.camera_head"]["bytes"]))
            np.testing.assert_array_equal(np.asarray(image), self.decoded_rgb)
        metadata = json.loads((output / "meta/astribot_subtask_metadata.json").read_text())["episodes"][0]
        self.assertEqual(metadata["subtask_instruction_map"][str(rows[-1]["subtask_instruction_index"][0])], "pick the block")
        with self.assertRaises(FileExistsError):
            convert_task(self.items, output, self.raw, 15, "robotwin")

    def test_rejects_moving_omitted_joint(self):
        with h5py.File(self.source, "r+") as f:
            f["joint_action/head"][2, 0] = 0.5
            with self.assertRaisesRegex(ValueError, "head_1 moves"):
                astribot_state(f)

    def test_requires_complete_language_annotation(self):
        self.items[0].metadata.write_text(json.dumps({"task_instruction": "Pick block", "subtask_instruction_map": {"1": "find block"}}))
        with self.assertRaisesRegex(ValueError, "No subtask text"):
            read_episode(self.items[0], "robotwin")


if __name__ == "__main__":
    unittest.main()
