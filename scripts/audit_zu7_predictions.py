"""Teacher-forced SmolVLA prediction audit on the saved Zu7 expert demonstrations.

This is offline inference only: it does not train or run the simulator.
It measures action-label alignment, trajectory coverage and errors by coarse
task phase derived from gripper command transitions. Those phase labels are
heuristic and must not be treated as simulator ground truth.
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

ROOT = Path("/data/jaka-vla")
JOINTS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
CAMERAS = ["observation.images.camera1", "observation.images.camera2", "observation.images.camera3"]
PACK_KEYS = ["left_arm_joint_states", "left_ee_joint_states", "right_arm_joint_states", "right_ee_joint_states"]


def pack(hf, root, prefix):
    return np.concatenate([hf[f"{root}/{key}"][:] for key in PACK_KEYS], axis=1).astype(np.float32)


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"count": 0}
    return {"count": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "p95": float(np.quantile(x, .95)), "max": float(x.max())}


def first_below(values, start, threshold):
    found = np.flatnonzero(values[start:] < threshold)
    return int(start + found[0]) if len(found) else len(values) - 1


def first_above(values, start, threshold):
    found = np.flatnonzero(values[start:] > threshold)
    return int(start + found[0]) if len(found) else len(values) - 1


def phase_boundaries(actions):
    left, right = actions[:, 6], actions[:, 13]
    left_close = first_below(left, 1, .98)
    left_closed = first_below(left, left_close, .05)
    left_release = first_above(left, left_closed + 1, .05)
    left_opened = first_above(left, left_release, .95)
    right_close = first_below(right, 1, .98)
    right_closed = first_below(right, right_close, .05)
    right_release = first_above(right, right_closed + 1, .05)
    phases = [
        ("left_approach", 0, left_close),
        ("left_close_and_transfer_approach", left_close, right_close),
        ("handover", right_close, max(left_opened, right_closed)),
        ("right_transport_and_place", max(left_opened, right_closed), right_release),
    ]
    return phases, {"left_close_start": left_close, "left_close_complete": left_closed,
                    "left_release_start": left_release, "left_release_complete": left_opened,
                    "right_close_start": right_close, "right_close_complete": right_closed,
                    "right_release_start": right_release}


def select_frames(phases, length, chunk):
    selected = {}
    for name, start, end in phases:
        start = max(0, min(int(start), length - chunk))
        end = max(start, min(int(end), length - chunk))
        if end == start:
            positions = [start]
        else:
            positions = [round(start + (end - start) * q) for q in (.12, .50, .88)]
        selected[name] = sorted(set(int(x) for x in positions))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/lerobot/zu7_collision_smoke")
    parser.add_argument("--repo-id", default="jaka-local/zu7-collision-smoke")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/smolvla_zu7_refine3000_seed42/checkpoints/last/pretrained_model")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/zu7_collision_smoke/handover_block/jaka_zu7_proxy_jaka_zu7_proxy/data")
    parser.add_argument("--output", type=Path, default=ROOT / "results/zu7_offline_audit_20261106")
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[0, 1, 2, 42])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.perf_counter()

    dataset = LeRobotDataset(args.repo_id, root=args.dataset, video_backend="pyav")
    checkpoint_identity = json.loads((args.checkpoint / "project_identity.json").read_text())
    data_identity = json.loads((args.dataset / "meta/project_identity.json").read_text())
    source_identity = json.loads((args.raw.parent / "robot_identity.json").read_text())
    if not (checkpoint_identity["robot_id"] == data_identity["robot_id"] == source_identity["robot_id"] == "dual_jaka_zu7"):
        raise ValueError("Robot identity mismatch across checkpoint/data/source")
    if not (checkpoint_identity["urdf_sha256"] == data_identity["urdf_sha256"] == source_identity["urdf_sha256"]):
        raise ValueError("Checkpoint, data and raw source use different Zu7 geometry")
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to("cuda").eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}})
    chunk = int(policy.config.chunk_size)
    if chunk != 50 or policy.config.input_features["observation.state"].shape != (14,):
        raise ValueError("Unexpected checkpoint shape/chunk; expected Zu7 state 14 and chunk 50")
    episodes = list(dataset.meta.episodes)
    if len(episodes) != 3:
        raise ValueError(f"This bounded audit expects the three saved Zu7 demonstrations; found {len(episodes)}")

    source = []
    for ep in episodes:
        i = int(ep["episode_index"])
        path = args.raw / f"episode_{i:07d}.hdf5"
        with h5py.File(path) as hf:
            state = pack(hf, "state", "")
            action = pack(hf, "action", "")
            command = pack(hf, "commanded_state", "")
            timing = hf["observation_timestamps"][:]
            if hf.attrs.get("robot_id") != "dual_jaka_zu7":
                raise ValueError(f"Wrong robot identity: {path}")
            align = np.abs(action[:-1] - command[1:])
            state_command = np.abs(state - command)
            phases, events = phase_boundaries(action)
            if len(state) != int(ep["length"]):
                raise ValueError(f"Raw and LeRobot episode length mismatch: {path}")
            source.append({"episode": i, "path": path, "length": len(state), "state": state, "action": action,
                           "command": command, "timestamps": timing, "phases": phases, "events": events,
                           "alignment": align, "state_command": state_command})

    # Validate how a global LeRobot index maps to an episode/frame before inference.
    offsets = [int(ep["dataset_from_index"]) for ep in episodes]
    selected = []
    for ep in source:
        frame_map = select_frames(ep["phases"], ep["length"], chunk)
        ep["selected_frames"] = frame_map
        for phase, frames in frame_map.items():
            selected.extend((ep["episode"], frame, phase) for frame in frames)
    for ep_i, frame_i, _ in selected:
        item = dataset[offsets[ep_i] + frame_i]
        if int(item["episode_index"]) != ep_i or int(item["frame_index"]) != frame_i:
            raise ValueError("LeRobot global index does not map to the expected episode/frame")

    per_observation = []
    per_phase = {}
    predictions_saved = []
    calls = 0
    for ep_i, frame_i, phase in selected:
        item = dataset[offsets[ep_i] + frame_i]
        observation = {k: v for k, v in item.items() if k.startswith("observation.") or k == "task"}
        target = np.stack([source[ep_i]["action"][frame_i:frame_i + chunk] for _ in [0]])[0]
        outputs = []
        for seed in args.noise_seeds:
            policy.reset()
            generator = torch.Generator(device="cuda").manual_seed(int(seed))
            noise = torch.randn((1, chunk, policy.config.max_action_dim), generator=generator, device="cuda")
            with torch.inference_mode():
                predicted = post(policy.predict_action_chunk(pre(observation), noise=noise))[0].float().cpu().numpy()
            outputs.append(predicted)
            calls += 1
        pred = np.stack(outputs)
        error = np.abs(pred - target[None])
        hmetrics = {}
        for horizon in (1, 8, 16, chunk):
            e = error[:, :horizon]
            hmetrics[str(horizon)] = {
                "joint_mae_deg": float(np.rad2deg(e[:, :, JOINTS]).mean()),
                "joint_p95_deg": float(np.rad2deg(np.quantile(e[:, :, JOINTS], .95))),
                "gripper_mae": float(e[:, :, [6, 13]].mean()),
                "gripper_binary_accuracy": float((((pred[:, :horizon, [6, 13]] > .5) == (target[None, :horizon, [6, 13]] > .5)).mean())),
            }
        row = {"episode": ep_i, "frame": frame_i, "time_s": float(source[ep_i]["timestamps"][frame_i]),
               "phase": phase, "task": str(item["task"]), "noise_seeds": args.noise_seeds,
               "error_by_horizon": hmetrics,
               "first_step_joint_mae_deg_by_noise_seed": np.rad2deg(np.abs(pred[:, 0, JOINTS] - target[0, JOINTS]).mean(axis=1)).tolist(),
               "first_step_gripper_pred_mean": pred[:, 0, [6, 13]].mean(axis=0).tolist(),
               "first_step_gripper_target": target[0, [6, 13]].tolist()}
        per_observation.append(row)
        per_phase.setdefault(phase, []).append(row)
        predictions_saved.append({"episode": ep_i, "frame": frame_i, "phase": phase, "predictions": pred, "target": target})
        print(f"AUDITED episode={ep_i} frame={frame_i} phase={phase}", flush=True)

    phase_summary = {}
    for phase, rows in per_phase.items():
        phase_summary[phase] = {"observations": len(rows), "horizons": {h: {
            "joint_mae_deg_mean": float(np.mean([r["error_by_horizon"][h]["joint_mae_deg"] for r in rows])),
            "joint_mae_deg_p95_over_observations": float(np.quantile([r["error_by_horizon"][h]["joint_mae_deg"] for r in rows], .95)),
            "gripper_mae_mean": float(np.mean([r["error_by_horizon"][h]["gripper_mae"] for r in rows])),
            "gripper_binary_accuracy_mean": float(np.mean([r["error_by_horizon"][h]["gripper_binary_accuracy"] for r in rows])),
        } for h in ("1", "8", "16", str(chunk))}}
    initial_states = np.stack([x["state"][0] for x in source])
    coverage = {"episodes": len(source), "frames": int(sum(x["length"] for x in source)),
        "same_three_scene_seed_reuse": True,
        "initial_joint_state_range_rad": np.ptp(initial_states[:, JOINTS], axis=0).tolist(),
        "joint_state_min_rad": np.min(np.concatenate([x["state"][:, JOINTS] for x in source], axis=0), axis=0).tolist(),
        "joint_state_max_rad": np.max(np.concatenate([x["state"][:, JOINTS] for x in source], axis=0), axis=0).tolist(),
        "note": "Three trajectories use seeds 0,1,2; the second dataset reuses the same initial scenes and does not add independent scene coverage."}
    alignment = {str(x["episode"]): {"max_action_to_next_command_abs": float(x["alignment"].max()),
        "p95_action_to_next_command_abs": float(np.quantile(x["alignment"], .95)),
        "median_state_to_command_abs": float(np.median(x["state_command"])),
        "p95_state_to_command_abs": float(np.quantile(x["state_command"], .95)),
        "duration_s": float(x["timestamps"][-1] - x["timestamps"][0]), "event_frames": x["events"],
        "heuristic_phase_ranges": {name: [a, b] for name, a, b in x["phases"]}}
        for x in source}
    report = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scope": "Teacher-forced offline inference on the three saved training trajectories; no training or simulator rollout.",
        "checkpoint": str(args.checkpoint), "chunk_size": chunk, "sampled_noise_seeds": args.noise_seeds,
        "prediction_calls": calls, "elapsed_seconds": time.perf_counter() - started,
        "dataset_coverage": coverage, "label_and_tracking_alignment": alignment,
        "phase_summary": phase_summary, "observations": per_observation,
        "limitations": ["Only three training scenes; no generalization claim.",
            "Teacher-forced errors do not predict closed-loop success by themselves.",
            "Phase boundaries are inferred from gripper target thresholds, not semantic simulator labels.",
            "Noise-seed variation is a diagnostic and differs from the default deterministic fixed-noise rollout."]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    summary = ["# Zu7离线动作预测审计", "", report["scope"], "", "## 数据/标签核查", "",
        f"- {coverage['episodes']}条示教、{coverage['frames']}帧，分属同3个场景；对齐误差按回合列在下表。",
        "- 阶段边界由左右夹爪开闭动作估计，仅用于误差分组。", "",
        "| 回合 | 帧数 | 时长(s) | action(t)与command(t+1)最大差 | 状态-命令绝对误差p95 | 关键事件帧 |",
        "|---:|---:|---:|---:|---:|---|"]
    for key, val in alignment.items():
        summary.append(f"| {key} | {source[int(key)]['length']} | {val['duration_s']:.2f} | {val['max_action_to_next_command_abs']:.6g} | {val['p95_state_to_command_abs']:.5f} | {val['event_frames']} |")
    summary += ["", "## 分阶段预测误差", "", "误差以关节角度MAE（度）和夹爪误差报告。H表示比较预测块前H帧。", "",
        "| 阶段 | 观测数 | H=1关节MAE° | H=8关节MAE° | H=16关节MAE° | H=50关节MAE° | H=1夹爪MAE | H=1夹爪阈值准确率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for phase, val in phase_summary.items():
        h = val["horizons"]
        summary.append(f"| {phase} | {val['observations']} | {h['1']['joint_mae_deg_mean']:.2f} | {h['8']['joint_mae_deg_mean']:.2f} | {h['16']['joint_mae_deg_mean']:.2f} | {h['50']['joint_mae_deg_mean']:.2f} | {h['1']['gripper_mae_mean']:.3f} | {h['1']['gripper_binary_accuracy_mean']:.1%} |")
    summary += ["", "## 边界", "", *[f"- {x}" for x in report["limitations"]], ""]
    (args.output / "report.md").write_text("\n".join(summary), encoding="utf-8")
    arrays = {"episode": [], "frame": [], "phase": [], "predictions": [], "targets": []}
    for x in predictions_saved:
        arrays["episode"].append(x["episode"]); arrays["frame"].append(x["frame"]); arrays["phase"].append(x["phase"])
        arrays["predictions"].append(x["predictions"]); arrays["targets"].append(x["target"])
    np.savez_compressed(args.output / "predictions.npz", episode=arrays["episode"], frame=arrays["frame"],
                        phase=np.asarray(arrays["phase"]), predictions=np.stack(arrays["predictions"]), targets=np.stack(arrays["targets"]))
    print(json.dumps({"output": str(args.output), "calls": calls, "elapsed_seconds": report["elapsed_seconds"],
                      "phase_summary": phase_summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
