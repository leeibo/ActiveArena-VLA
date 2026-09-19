#!/usr/bin/env python3
"""Convert ActiveArena Astribot HDF5 demonstrations to LeRobot v2.1 image datasets.

This portable implementation follows the frozen ActiveArena Astribot data
contract and next-state supervision. It does not depend on the LeRobot package.
Use --dry-run to validate all selected episodes without writing output.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

STATE_NAMES = ([f"left_arm_{i}" for i in range(7)] + ["left_gripper"]
               + [f"right_arm_{i}" for i in range(7)] + ["right_gripper", "torso_yaw", "head_2"])
IMAGE_KEY = "observation.images.camera_head"
ANNOTATIONS = {"subtask_index": "subtask", "subtask_instruction_index": "subtask_instruction_idx", "stage": "stage"}
EPISODE_RE = re.compile(r"episode(\d+)\.hdf5$")


@dataclass(frozen=True)
class Episode:
    task: str
    source: Path
    metadata: Path


def discover_episodes(raw_root: Path, config: str, tasks: list[str] | None, max_episodes: int | None) -> list[Episode]:
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw data root does not exist: {raw_root}")
    selected = tasks or sorted(p.name for p in raw_root.iterdir() if p.is_dir() and not p.name.startswith("."))
    episodes = []
    for task in selected:
        task_dir = raw_root / task / config
        files = sorted((p for p in (task_dir / "data").glob("episode*.hdf5") if EPISODE_RE.fullmatch(p.name)),
                       key=lambda p: int(EPISODE_RE.fullmatch(p.name)[1]))
        if not files:
            raise FileNotFoundError(f"No episodes under {task_dir / 'data'}; --config must match the full directory name")
        for source in files[:max_episodes]:
            metadata = task_dir / "subtask_metadata" / f"{source.stem}.json"
            if not metadata.is_file():
                raise FileNotFoundError(f"Missing language/subtask metadata: {metadata}")
            episodes.append(Episode(task, source, metadata))
    if not episodes:
        raise ValueError("No tasks were selected")
    return episodes


def astribot_state(episode: h5py.File) -> np.ndarray:
    """Return the released 18-D order, never the raw 19-D vector order."""
    def read(name: str, width: int) -> np.ndarray:
        value = np.asarray(episode[f"joint_action/{name}"][:], dtype=np.float32)
        if value.ndim == 1:
            value = value[:, None]
        if value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"joint_action/{name}: expected (N, {width}), got {value.shape}")
        return value
    left, right = read("left_arm", 7), read("right_arm", 7)
    left_gripper, right_gripper = read("left_gripper", 1), read("right_gripper", 1)
    torso, head = read("torso", 1), read("head", 2)
    lengths = {len(x) for x in (left, right, left_gripper, right_gripper, torso, head)}
    if len(lengths) != 1 or len(left) < 2:
        raise ValueError(f"Joint streams must have equal lengths >= 2, got {lengths}")
    state = np.concatenate([left, left_gripper, right, right_gripper, torso, head[:, 1:2]], axis=1)
    if not np.isfinite(state).all():
        raise ValueError("Joint data contains NaN or infinite values")
    # The released action adapter holds head_1 fixed; silently dropping a moving
    # head_1 would produce a dataset that this embodiment cannot reproduce.
    if not np.allclose(head[:, 0], head[0, 0], atol=1e-5):
        raise ValueError("head_1 moves in this episode; the released 18-D policy assumes it is fixed")
    return state


def decode_rgb(encoded: object, encoding: str) -> np.ndarray:
    data = np.asarray(encoded)
    if data.ndim == 3:
        rgb = data.astype(np.uint8)
    else:
        rgb = cv2.imdecode(np.frombuffer(bytes(encoded), dtype=np.uint8), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError("Failed to decode head-camera JPEG")
        # ActiveArena stores SAPIEN RGB arrays through cv2.imencode. Matching
        # cv2.imdecode restores that original array; a second swap is incorrect.
        if encoding == "standard-jpeg":
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel camera frame, got {rgb.shape}")
    return rgb


def read_episode(item: Episode, encoding: str) -> tuple[np.ndarray, dict[str, np.ndarray], dict, str]:
    metadata = json.loads(item.metadata.read_text(encoding="utf-8"))
    instruction = metadata.get("task_instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"Missing task_instruction in {item.metadata}")
    instruction_map = {str(k): str(v) for k, v in metadata.get("subtask_instruction_map", {}).items()}
    with h5py.File(item.source, "r") as episode:
        state = astribot_state(episode)
        camera_key = next((key for key in ("observation/camera_head/rgb", "observation/head_camera/rgb") if key in episode), None)
        if camera_key is None:
            raise KeyError(f"No head camera in {item.source}")
        if len(episode[camera_key]) != len(state):
            raise ValueError(f"Head-camera/joint frame mismatch in {item.source}")
        decode_rgb(episode[camera_key][0], encoding)
        columns = {}
        for output, source in ANNOTATIONS.items():
            if source not in episode:
                raise KeyError(f"Missing required annotation {source} in {item.source}")
            value = np.asarray(episode[source][:], dtype=np.int64).reshape(-1)
            if len(value) != len(state):
                raise ValueError(f"Annotation length mismatch: {source} in {item.source}")
            columns[output] = value[:-1]
        missing = sorted({str(i) for i in columns['subtask_instruction_index']} - instruction_map.keys())
        if missing:
            raise ValueError(f"No subtask text for indices {missing} in {item.metadata}")
        for key in ("subtask_keyframe", "motion_keyframe"):
            if key in episode:
                value = np.asarray(episode[key][:], dtype=np.int64).reshape(-1)
                if len(value) != len(state):
                    raise ValueError(f"Annotation length mismatch: {key}")
                columns[key] = value[:-1]
    return state, columns, metadata, camera_key


def statistics(values: np.ndarray) -> dict:
    return {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(),
            "q01": np.quantile(values, .01, axis=0).tolist(), "q99": np.quantile(values, .99, axis=0).tolist(),
            "count": [len(values)]}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf-8")


def convert_task(items: list[Episode], output: Path, raw_root: Path, fps: int, encoding: str) -> dict:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new --output-root to avoid overwriting data.")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    episode_rows, episode_stats, subtask_rows, task_rows = [], [], [], []
    task_indices, total, shape, optional = {}, 0, None, None
    try:
        (partial / "meta").mkdir()
        for index, item in enumerate(items):
            state, annotations, metadata, camera_key = read_episode(item, encoding)
            count = len(state) - 1
            instruction = metadata["task_instruction"]
            if instruction not in task_indices:
                task_indices[instruction] = len(task_indices)
                task_rows.append({"task_index": task_indices[instruction], "task": instruction})
            image_entries = []
            with h5py.File(item.source, "r") as episode:
                for frame in episode[camera_key][:-1]:
                    rgb = decode_rgb(frame, encoding)
                    current_shape = [3, rgb.shape[0], rgb.shape[1]]
                    if shape is not None and current_shape != shape:
                        raise ValueError(f"Camera resolution changes within {item.task}: {shape} -> {current_shape}")
                    shape = current_shape
                    buffer = io.BytesIO()
                    Image.fromarray(rgb).save(buffer, format="PNG")
                    image_entries.append({"bytes": buffer.getvalue(), "path": None})
            if optional is not None and set(annotations) != optional:
                raise ValueError(f"Optional keyframe fields differ between episodes in {item.task}")
            optional = set(annotations)
            fields = {
                "observation.state": pa.array(state[:-1].tolist(), type=pa.list_(pa.float32(), 18)),
                "action": pa.array(state[1:].tolist(), type=pa.list_(pa.float32(), 18)),
                IMAGE_KEY: pa.array(image_entries, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                **{key: pa.array(values[:, None].tolist(), type=pa.list_(pa.int64(), 1)) for key, values in annotations.items()},
                "timestamp": pa.array(np.arange(count, dtype=np.float32) / fps),
                "frame_index": pa.array(np.arange(count, dtype=np.int64)),
                "episode_index": pa.array(np.full(count, index, dtype=np.int64)),
                "index": pa.array(np.arange(total, total + count, dtype=np.int64)),
                "task_index": pa.array(np.full(count, task_indices[instruction], dtype=np.int64)),
            }
            destination = partial / "data" / f"chunk-{index // 1000:03d}" / f"episode_{index:06d}.parquet"
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table(fields), destination, compression="zstd")
            episode_rows.append({"episode_index": index, "tasks": [instruction], "length": count})
            episode_stats.append({"episode_index": index, "stats": {
                "observation.state": statistics(state[:-1]), "action": statistics(state[1:]),
                **{key: statistics(value[:, None]) for key, value in annotations.items()},
            }})
            subtask_rows.append({"lerobot_episode_index": index, "num_frames": count,
                                 "source_episode": str(item.source.relative_to(raw_root)),
                                 "task_name": item.task, "task_instruction": instruction,
                                 "subtask_instruction_map": metadata["subtask_instruction_map"],
                                 "subtask_defs": metadata.get("subtask_defs", [])})
            total += count
            print(f"{item.task}: episode {index + 1}/{len(items)}, {count} transitions", flush=True)
        # State/action feature names are a flat list in the LeRobot v2.1
        # schema. Images use named axes separately below.
        features = {key: {"dtype": "float32", "shape": [18], "names": STATE_NAMES} for key in ("observation.state", "action")}
        features[IMAGE_KEY] = {"dtype": "image", "shape": shape, "names": ["channels", "height", "width"]}
        features.update({key: {"dtype": "int64", "shape": [1], "names": [key]} for key in optional})
        features.update({key: {"dtype": "float32" if key == "timestamp" else "int64", "shape": [1], "names": None}
                         for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")})
        write_json(partial / "meta/info.json", {
            "codebase_version": "v2.1", "robot_type": "astribot", "total_episodes": len(items), "total_frames": total,
            "total_tasks": len(task_rows), "total_videos": 0, "total_chunks": (len(items) + 999) // 1000,
            "chunks_size": 1000, "fps": fps, "splits": {"train": f"0:{len(items)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet", "video_path": None, "features": features,
        })
        write_json(partial / "meta/modality.json", {
            "state": {"astribot": {"original_key": "observation.state", "start": 0, "end": 18}},
            "action": {"astribot": {"original_key": "action", "start": 0, "end": 18}},
            "video": {"camera_head": {"original_key": IMAGE_KEY}},
            "annotation": {"human.action.task_description": {"original_key": "task_index"}},
        })
        write_jsonl(partial / "meta/episodes.jsonl", episode_rows)
        write_jsonl(partial / "meta/episodes_stats.jsonl", episode_stats)
        write_jsonl(partial / "meta/tasks.jsonl", task_rows)
        write_json(partial / "meta/astribot_subtask_metadata.json", {"episodes": subtask_rows})
        write_json(partial / "meta/conversion_manifest.json", {
            "converter": "ActiveArena-VLA/tools/convert_astribot_to_lerobot.py", "version": 1,
            "state_action_alignment": "state[t] = joints[t], action[t] = joints[t+1]; final observation dropped",
            "state_order": STATE_NAMES, "jpeg_encoding": encoding, "fps": fps,
            "episodes": len(items), "frames": total, "head_1": "held fixed; omitted from policy representation",
        })
        partial.replace(output)
    except BaseException:
        shutil.rmtree(partial)
        raise
    return {"task": items[0].task, "episodes": len(items), "frames": total, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="ActiveArena data root containing task directories")
    parser.add_argument("--output-root", type=Path, required=True, help="LeRobot root, one dataset directory per task")
    parser.add_argument("--config", default="info_gathering_demo__info_gathering_demo", help="Exact collected config directory below each task")
    parser.add_argument("--tasks", nargs="+", help="Task names; default: all task directories in raw root")
    parser.add_argument("--max-episodes", type=int, help="Convert first N episodes per task in numeric order")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--jpeg-encoding",
        choices=("activearena", "standard-jpeg", "robotwin"),
        default="activearena",
        help="activearena uses the frozen cv2.imencode path; robotwin is a deprecated compatibility alias; standard-jpeg is for other producers",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate joints, metadata, annotation lengths, and first image; write nothing")
    args = parser.parse_args()
    if args.fps <= 0 or (args.max_episodes is not None and args.max_episodes <= 0):
        parser.error("--fps and --max-episodes must be positive")
    episodes = discover_episodes(args.raw_root, args.config, args.tasks, args.max_episodes)
    if args.dry_run:
        transitions = sum(len(read_episode(item, args.jpeg_encoding)[0]) - 1 for item in episodes)
        print(json.dumps({"valid_episodes": len(episodes), "transitions": transitions, "dry_run": True}))
        return
    by_task = {}
    for item in episodes:
        by_task.setdefault(item.task, []).append(item)
    for task, items in by_task.items():
        print(json.dumps(convert_task(items, args.output_root / task, args.raw_root, args.fps, args.jpeg_encoding)))


if __name__ == "__main__":
    main()
