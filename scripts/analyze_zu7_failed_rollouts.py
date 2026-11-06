"""Align failed SmolVLA closed-loop traces against their same-seed expert demos."""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/data/jaka-vla")
KEYS = ["left_arm_joint_states", "left_ee_joint_states", "right_arm_joint_states", "right_ee_joint_states"]
JOINT_DIMS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
JOINT_NAMES = [f"LJ{i}" for i in range(1, 7)] + [f"RJ{i}" for i in range(1, 7)]


def packed(hf, prefix):
    return np.concatenate([hf[f"{prefix}/{key}"][:] for key in KEYS], axis=1).astype(np.float32)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def first_crossing(values, limit):
    hits = np.flatnonzero(values > limit)
    return int(hits[0]) if len(hits) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expert", type=Path, default=ROOT / "data/raw/zu7_collision_smoke/handover_block/jaka_zu7_proxy_jaka_zu7_proxy/data")
    p.add_argument("--results", type=Path, default=ROOT / "results/zu7_learned_seen6000")
    p.add_argument("--output", type=Path, default=ROOT / "results/zu7_offline_audit_20261106")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    for name in ('closed_loop_comparison.json', 'closed_loop_comparison.md', 'closed_loop_state_drift.png'):
        if (a.output / name).exists():
            raise FileExistsError(a.output / name)
    arrays = []
    report = {"scope": "Offline alignment of recorded closed-loop policy and same-seed saved expert trajectories; no training or rollout.", "episodes": {}}
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    for seed in range(3):
        expert_path = a.expert / f"episode_{seed:07d}.hdf5"
        with h5py.File(expert_path) as hf:
            expert_state = packed(hf, "state")
            expert_action = packed(hf, "action")
            expert_command = packed(hf, "commanded_state")
        trace_path = a.results / f"seed_{seed}_trace.npz"
        with np.load(trace_path) as z:
            trace = z["control_trace"]
            chunks = z["raw_chunks"] if "raw_chunks" in z.files else None
            blended_chunks = z["blended_chunks"] if "blended_chunks" in z.files else None
            chunk_steps = z["chunk_steps"] if "chunk_steps" in z.files else None
        steps = trace[:, 0].astype(int)
        measured = trace[:, 1:15]
        target_raw = trace[:, 29:43]
        target_clipped = trace[:, 43:57]
        valid = steps < len(expert_state)
        steps, measured, target_raw, target_clipped = steps[valid], measured[valid], target_raw[valid], target_clipped[valid]
        expert_same = expert_state[steps]
        qerr = np.rad2deg(np.abs(measured[:, JOINT_DIMS] - expert_same[:, JOINT_DIMS]))
        arm_errors = {"left": qerr[:, :6].mean(axis=1), "right": qerr[:, 6:].mean(axis=1)}
        joint_by_name = {name: stats(qerr[:, i]) for i, name in enumerate(JOINT_NAMES)}
        target_action_error = np.abs(target_raw - expert_action[steps])
        target_action_joint_deg = np.rad2deg(target_action_error[:, JOINT_DIMS])
        target_target_next = np.abs(target_clipped - expert_command[steps + 1]) if len(steps) and steps[-1] + 1 < len(expert_command) else np.empty(0)
        pred_rows = []
        boundary_jumps = []
        if chunks is not None:
            for i, t in enumerate(chunk_steps.astype(int)):
                if t >= len(expert_action):
                    continue
                valid_chunk = min(len(expert_action) - t, chunks.shape[1])
                pred_rows.append({"step": int(t), "joint_first_action_mae_deg": float(np.rad2deg(np.abs(chunks[i, 0, JOINT_DIMS] - expert_action[t, JOINT_DIMS])).mean()),
                    "gripper_first_action_mae": float(np.abs(chunks[i, 0, [6, 13]] - expert_action[t, [6, 13]]).mean()),
                    "joint_executed_8_action_mae_deg": float(np.rad2deg(np.abs(chunks[i, :min(8, valid_chunk)][:, JOINT_DIMS] - expert_action[t:t + min(8, valid_chunk)][:, JOINT_DIMS])).mean())})
                if i + 1 < len(chunks) and int(chunk_steps[i + 1]) == int(t + 8):
                    boundary_jumps.append(float(np.rad2deg(np.abs(chunks[i, 8][JOINT_DIMS] - chunks[i + 1, 0][JOINT_DIMS])).mean()))
        event_times = []
        contacts_path = a.results / f"seed_{seed}_contacts.json"
        if contacts_path.exists():
            for item in json.loads(contacts_path.read_text()):
                event_times.append({"first_time_s": item.get("first_time_s"), "peak_impulse_ns": item.get("peak_impulse_ns"), "pair": item.get("pair")})
        row = {"failure": json.loads((a.results / "results.json").read_text())[seed].get("failure_reason"),
            "logged_steps": len(steps), "duration_s": float(steps[-1] / 25) if len(steps) else 0,
            "state_error_by_arm_deg": {arm: stats(e) for arm, e in arm_errors.items()},
            "first_state_divergence_over_2_5_10deg_control_steps": {str(threshold): {arm: first_crossing(err, threshold) for arm, err in arm_errors.items()} for threshold in (2, 5, 10)},
            "largest_joint_state_errors_deg": joint_by_name,
            "raw_target_vs_expert_action_joint_mae_deg": stats(target_action_joint_deg) if len(target_action_joint_deg) else {},
            "clipped_controller_command_vs_next_expert_command_joint_mae_deg": stats(np.rad2deg(target_target_next[:, JOINT_DIMS])) if len(target_target_next) else {},
            "predicted_chunks_vs_expert_at_inference_time": pred_rows,
            "replan_boundary_jump_mean_deg": float(np.mean(boundary_jumps)) if boundary_jumps else None,
            "replan_boundary_jump_p95_deg": float(np.quantile(boundary_jumps, .95)) if boundary_jumps else None,
            "recorded_contact_pairs": event_times}
        report["episodes"][str(seed)] = row
        arrays.append({"seed": seed, "steps": steps, "left": arm_errors["left"], "right": arm_errors["right"]})
        axes[seed].plot(steps / 25, arm_errors["left"], label="Left arm", linewidth=1.6)
        axes[seed].plot(steps / 25, arm_errors["right"], label="Right arm", linewidth=1.6)
        axes[seed].set_ylabel(f"seed {seed}\nMAE (deg)")
        axes[seed].grid(alpha=.25)
    axes[0].legend(loc="upper left", ncols=2)
    axes[-1].set_xlabel("Simulation time (s)")
    fig.suptitle("SmolVLA executed-state drift from same-seed expert trajectory")
    fig.savefig(a.output / "closed_loop_state_drift.png", dpi=180)
    plt.close(fig)
    report["limitations"] = ["Expert and policy run use the same RNG seeds, but contact dynamics and images can diverge after small state differences.",
        "Same-timestep state difference is diagnostic, not a causal decomposition; post-divergence expert-state comparisons are increasingly off-manifold.",
        "Joint angle differences use absolute per-joint position error; these summaries are not task success metrics."]
    (a.output / "closed_loop_comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = ["# 闭环失败与同种子专家轨迹对齐", "", report["scope"], "", "| seed | 失败点 | 首次左臂偏差>2° | >5° | >10° | 末段左臂MAE | 末段右臂MAE | 重规划边界跳变均值 |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for seed, row in report["episodes"].items():
        thresholds = row["first_state_divergence_over_2_5_10deg_control_steps"]
        left, right = row["state_error_by_arm_deg"]["left"], row["state_error_by_arm_deg"]["right"]
        fmt = lambda value: "—" if value is None else str(value)
        lines.append(f"| {seed} | {row['failure']} | {fmt(thresholds['2']['left'])} | {fmt(thresholds['5']['left'])} | {fmt(thresholds['10']['left'])} | {left['mean']:.2f}° | {right['mean']:.2f}° | {fmt(row['replan_boundary_jump_mean_deg'])}° |")
    lines += ["", "## 判读限制", "", *[f"- {x}" for x in report["limitations"]], ""]
    (a.output / "closed_loop_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({seed: {"failure": row["failure"], "divergence": row["first_state_divergence_over_2_5_10deg_control_steps"],
        "left": row["state_error_by_arm_deg"]["left"], "right": row["state_error_by_arm_deg"]["right"],
        "replan_jump_mean_deg": row["replan_boundary_jump_mean_deg"]} for seed, row in report["episodes"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
