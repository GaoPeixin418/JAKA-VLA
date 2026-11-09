"""Classify saved matched grasp rollouts by observable offline event order."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_episode(success, first_close_step, displacement_before_close_m, error_at_close_mm):
    if success:
        return "verified_grasp"
    if first_close_step is None:
        return "no_close_before_budget"
    if displacement_before_close_m >= 0.02:
        return "object_displaced_before_close"
    if error_at_close_mm is not None and error_at_close_mm >= 40.0:
        return "off_target_at_close"
    return "failed_close_near_target"


def event_time(control, predicate):
    indices = np.flatnonzero(predicate)
    if not len(indices):
        return None
    return float(control[indices[0], 0] / 25.0)


def analyze_row(root, run_name, row):
    seed = int(row["seed"])
    directory = root / "results" / run_name
    trace_path = directory / f"seed_{seed}_trace.npz"
    if file_sha256(trace_path) != row["trace_sha256"]:
        raise ValueError(f"Trace SHA256 mismatch: {trace_path}")
    with np.load(trace_path, allow_pickle=False) as trace:
        control = trace["control_trace"]
        evidence = trace["grasp_evidence"]
    if control.ndim != 2 or control.shape[1] != 74 or evidence.ndim != 2 or evidence.shape[1] != 6:
        raise ValueError(f"Unexpected trace schema: {trace_path}")
    if not np.all(np.diff(control[:, 0]) == 1):
        raise ValueError(f"Nonconsecutive control steps: {trace_path}")

    close_indices = np.flatnonzero(control[:, 35] < 0.15)
    close_index = int(close_indices[0]) if len(close_indices) else None
    before_close = control[:close_index + 1] if close_index is not None else control
    object_xy = control[:, 57:59]
    displacement = np.linalg.norm(object_xy - object_xy[0], axis=1)
    preclose_max = float(displacement[:len(before_close)].max())
    contact_path = directory / f"seed_{seed}_contacts.json"
    contacts = json.loads(contact_path.read_text())
    block_contact = [float(c["first_time_s"]) for c in contacts if "scene:box" in c["pair"]]
    valid_hold = evidence[:, 5] > 0.5
    first_hold = np.flatnonzero(valid_hold)
    longest_hold_ticks = 0
    current_hold_ticks = 0
    for held in valid_hold:
        current_hold_ticks = current_hold_ticks + 1 if held else 0
        longest_hold_ticks = max(longest_hold_ticks, current_hold_ticks)
    error = row["target_horizontal_error_at_close_mm"]
    label = classify_episode(bool(row["grasp_success"]), row["first_close_step"], preclose_max, error)
    if close_index is None and row["first_close_step"] is not None:
        raise ValueError(f"Close event absent from trace: {trace_path}")
    if close_index is not None and int(control[close_index, 0]) != row["first_close_step"]:
        raise ValueError(f"Close step mismatch: {trace_path}")
    if (longest_hold_ticks * 0.004 >= 0.6) != bool(row["grasp_success"]):
        raise ValueError(f"Verified hold mismatch: {trace_path}")
    return {
        "seed": seed,
        "label": label,
        "grasp_success": bool(row["grasp_success"]),
        "first_block_contact_s": min(block_contact) if block_contact else None,
        "first_20mm_horizontal_object_displacement_s": event_time(control, displacement >= 0.02),
        "first_close_request_s": float(control[close_index, 0] / 25.0) if close_index is not None else None,
        "first_valid_hold_tick_s": float(evidence[first_hold[0], 0]) if len(first_hold) else None,
        "longest_valid_hold_s": round(longest_hold_ticks * 0.004, 3),
        "max_horizontal_object_displacement_through_close_mm": round(1000 * preclose_max, 3),
        "raw_tcp_horizontal_error_at_close_or_end_mm": error,
        "block_contact_before_close": bool(block_contact and (close_index is None or min(block_contact) < control[close_index, 0] / 25.0)),
        "trace_sha256": row["trace_sha256"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/jaka-vla"))
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence_path = args.evidence or args.root / "results/zu7_matched_checkpoint_evidence_20261109.json"
    evidence = json.loads(evidence_path.read_text())
    results = {}
    for checkpoint in evidence["checkpoints"]:
        name = checkpoint["checkpoint"]
        run_name = f"zu7_matched_{name}_seed203_207"
        rows = [analyze_row(args.root, run_name, row) for row in checkpoint["episodes_detail"]]
        if len(rows) != 5 or len({r["seed"] for r in rows}) != 5:
            raise ValueError(f"Expected five distinct seeds for {name}")
        if sum(r["grasp_success"] for r in rows) != checkpoint["successes"]:
            raise ValueError(f"Success count mismatch for {name}")
        results[name] = {"counts": dict(sorted(Counter(r["label"] for r in rows).items())), "rows": rows}
    result = {
        "scope": "Offline descriptive event labels for the matched 15 rollout development set; simulator contact and object pose are not policy inputs.",
        "thresholds": {"horizontal_object_displacement_m": 0.02, "raw_horizontal_error_mm": 40.0, "close_request_aperture": 0.15},
        "source_evidence_sha256": file_sha256(evidence_path),
        "checkpoints": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v["counts"] for k, v in results.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
