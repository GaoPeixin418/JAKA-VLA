"""Convert a local LeRobot 3.0 Zu7 dataset to the OpenPI-pinned LeRobot layout."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


CAMERAS = {
    "observation.images.camera1": "image",
    "observation.images.camera2": "left_wrist_image",
    "observation.images.camera3": "right_wrist_image",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(source):
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text())
    if not str(info.get("codebase_version", "")).startswith("v3") and not str(info.get("codebase_version", "")).startswith("3"):
        raise ValueError("Expected LeRobot 3.0 source")
    if info["fps"] != 25:
        raise ValueError("Expected 25 Hz source")
    expected = {"observation.state": [14], "action": [14]}
    for key, shape in expected.items():
        if info["features"][key]["shape"] != shape:
            raise ValueError(f"Unexpected {key} shape")
    for key in CAMERAS:
        if info["features"][key]["shape"] != [240, 320, 3]:
            raise ValueError(f"Unexpected camera shape: {key}")
    meta_files = sorted((source / "meta/episodes").rglob("*.parquet"))
    if not meta_files:
        raise FileNotFoundError("Episode metadata missing")
    episodes = [row for path in meta_files for row in pq.read_table(path).to_pylist()]
    if len(episodes) != info["total_episodes"]:
        raise ValueError("Episode count mismatch")
    return info, episodes, info_path


def open_video(source, info, key, episode, video_starts):
    chunk = episode[f"videos/{key}/chunk_index"]
    file_index = episode[f"videos/{key}/file_index"]
    relative = info["video_path"].format(video_key=key, chunk_index=chunk, file_index=file_index)
    path = source / relative
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frame_offset = round((float(episode[f"videos/{key}/from_timestamp"]) - video_starts[path]) * 25)
    if frame_offset < 0:
        raise ValueError(f"Negative frame offset: {path}")
    return capture, path, frame_offset


def convert(source, output, max_episodes):
    info, episodes, info_path = load_source(source)
    if output.exists():
        raise FileExistsError(output)
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive")
    selected = episodes[:max_episodes]
    source_files = {info_path}
    video_starts = {}
    for episode in selected:
        for key in CAMERAS:
            chunk = episode[f"videos/{key}/chunk_index"]
            file_index = episode[f"videos/{key}/file_index"]
            relative = info["video_path"].format(video_key=key, chunk_index=chunk, file_index=file_index)
            path = source / relative
            start = float(episode[f"videos/{key}/from_timestamp"])
            video_starts[path] = min(video_starts.get(path, start), start)
    features = {
        "state": {"dtype": "float32", "shape": (14,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (14,), "names": ["actions"]},
    }
    for target in CAMERAS.values():
        features[target] = {"dtype": "image", "shape": (240, 320, 3), "names": ["height", "width", "channel"]}

    dataset = LeRobotDataset.create(
        repo_id=output.name,
        root=output,
        robot_type="jaka_zu7_proxy",
        fps=25,
        features=features,
        use_videos=False,
        image_writer_threads=4,
    )
    checks = []
    for output_episode, episode in enumerate(selected):
        source_episode = int(episode["episode_index"])
        if source_episode != output_episode:
            raise ValueError("Expected consecutive source episode indices from zero")
        chunk = episode["data/chunk_index"]
        file_index = episode["data/file_index"]
        data_path = source / info["data_path"].format(chunk_index=chunk, file_index=file_index)
        source_files.add(data_path)
        rows = pq.read_table(data_path).slice(episode["dataset_from_index"], episode["length"]).to_pylist()
        if len(rows) != episode["length"]:
            raise ValueError("Episode length mismatch")
        timestamps = np.asarray([float(row["timestamp"]) for row in rows])
        if len(rows) > 1 and not np.allclose(np.diff(timestamps), 1 / 25, atol=0.005):
            raise ValueError(f"Nonuniform timestamps in episode {source_episode}")
        captures = {}
        try:
            for key in CAMERAS:
                capture, video_path, frame_offset = open_video(source, info, key, episode, video_starts)
                source_files.add(video_path)
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_offset)
                captures[key] = capture
            for frame_index, row in enumerate(rows):
                if int(row["episode_index"]) != source_episode or int(row["frame_index"]) != frame_index:
                    raise ValueError(f"Episode/frame index mismatch in episode {source_episode}")
                state = np.asarray(row["observation.state"], dtype=np.float32)
                actions = np.asarray(row["action"], dtype=np.float32)
                if state.shape != (14,) or actions.shape != (14,) or not np.isfinite(state).all() or not np.isfinite(actions).all():
                    raise ValueError(f"Invalid state/action in episode {source_episode}, frame {frame_index}")
                frame = {"state": state, "actions": actions, "task": episode["tasks"][0]}
                for key, target in CAMERAS.items():
                    ok, bgr = captures[key].read()
                    if not ok or bgr.shape != (240, 320, 3):
                        raise ValueError(f"Cannot decode {key}, episode {source_episode}, frame {frame_index}")
                    frame[target] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                dataset.add_frame(frame)
            dataset.save_episode()
            checks.append({
                "source_episode": source_episode,
                "output_episode": output_episode,
                "frames": len(rows),
                "task": episode["tasks"][0],
                "first_state": np.asarray(rows[0]["observation.state"], dtype=np.float32).tolist(),
                "last_state": np.asarray(rows[-1]["observation.state"], dtype=np.float32).tolist(),
                "first_action": np.asarray(rows[0]["action"], dtype=np.float32).tolist(),
                "last_action": np.asarray(rows[-1]["action"], dtype=np.float32).tolist(),
                "timestamp_max_step_error_s": float(np.max(np.abs(np.diff(timestamps) - 0.04))) if len(rows) > 1 else 0.0,
            })
            print(f"Converted episode {source_episode}: {len(rows)} frames", flush=True)
        finally:
            for capture in captures.values():
                capture.release()
    manifest = {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "source_format": info["codebase_version"],
        "output_purpose": "OpenPI-pinned LeRobot local compatibility; no Hub upload",
        "image_semantics": "Decoded source MP4 RGB frames stored as PNG; no resizing, but source MP4 was already lossy",
        "state_semantics": "measured_qpos_and_gripper_aperture_14d",
        "action_semantics": "next_sample_absolute_drive_targets_14d",
        "fps": 25,
        "camera_mapping": CAMERAS,
        "source_files_sha256": {str(path.relative_to(source)): sha256(path) for path in sorted(source_files)},
        "episodes": checks,
    }
    (output / "jaka_conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    manifest = convert(args.source, args.output, args.max_episodes)
    print(json.dumps({"episodes": len(manifest["episodes"]), "frames": sum(x["frames"] for x in manifest["episodes"])}, indent=2))


if __name__ == "__main__":
    main()
