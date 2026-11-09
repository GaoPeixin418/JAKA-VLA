"""Recompute grasp evidence and motion metrics from saved evaluation traces.

Reads completed rollouts only. Simulator truth is used for offline evaluation;
this script never provides observations or actions to a policy.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    groups = {}
    joint_indices = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
    for name in json.loads(args.manifest.read_text()):
        directory = args.manifest.parent / name
        for report in json.loads((directory / 'results.json').read_text()):
            if report['complete_handover_tested'] or report['mode'] == 'expert_replay':
                raise ValueError('Expected learned-policy grasp subtask only')
            trace_path = directory / f"seed_{report['seed']}_trace.npz"
            with np.load(trace_path) as trace:
                physics = trace['physics_trace']
                evidence = trace['grasp_evidence']
                if len(physics) != len(evidence) or len(physics) < 2:
                    raise ValueError('Missing per-physics-step grasp evidence')
                dt = float(np.median(np.diff(physics[:, 0])))
                np.testing.assert_allclose(np.diff(physics[:, 0]), dt, atol=1e-8, rtol=1e-6)
                np.testing.assert_allclose(evidence[:, 0], physics[:, 0], atol=1e-8)
                valid = ((evidence[:, 1] > .05) & (evidence[:, 2] > .15)
                         & (evidence[:, 2] < .85) & (evidence[:, 3] == 1)
                         & (evidence[:, 4] == 0))
                np.testing.assert_array_equal(valid, evidence[:, 5].astype(bool))
                ticks = longest = 0
                first_success = None
                for index, held in enumerate(valid):
                    ticks = ticks + 1 if held else 0
                    longest = max(longest, ticks)
                    if ticks * dt >= .6 and first_success is None:
                        first_success = float(evidence[index, 0])
                if bool(report['grasp_success']) != (first_success is not None):
                    raise ValueError('Reported grasp success disagrees with saved evidence')
                np.testing.assert_allclose(longest * dt, report['longest_verified_left_grasp_s'], atol=1e-7)
                if first_success is not None:
                    np.testing.assert_allclose(first_success, report['grasp_success_time_s'], atol=1e-7)
                command_v = physics[:, 15:29][:, joint_indices]
                measured_v = physics[:, 43:57][:, joint_indices]
                tracking = abs(physics[:, 1:15][:, joint_indices] - physics[:, 29:43][:, joint_indices])
                condition = report['initial_condition']
                if condition.get('scope') == 'intervened_recovery_start_not_executed_prefix':
                    group = 'intervened_recovery_start'
                elif ('new_seed200_202' in name or '_matched_' in name
                      or any(abs(float(x)) > 1e-8
                             for x in condition.get('left_joint_offset_rad', []))):
                    group = 'new_scenes_with_initial_offset'
                else:
                    group = 'seen_scenes_nominal_start'
                row = {
                    'experiment': name, 'seed': report['seed'],
                    'grasp_success': first_success is not None,
                    'verified_hold_s': longest * dt, 'first_success_time_s': first_success,
                    'max_lift_m_per_physics_step': float(evidence[:, 1].max()),
                    'max_command_speed_rad_s': float(abs(command_v).max()),
                    'max_command_acceleration_rad_s2': float(abs(np.diff(command_v, axis=0) / dt).max()),
                    'max_measured_speed_rad_s': float(abs(measured_v).max()),
                    'max_tracking_error_rad': float(tracking.max()),
                    'p99_measured_speed_rad_s': float(np.quantile(abs(measured_v), .99)),
                    'normalized_state_abs_max': max(x['normalized_state_abs_max'] for x in report['normalization_checks']),
                    'failure_reason': report['failure_reason'],
                    'initial_condition': condition, 'trace_sha256': sha256(trace_path),
                    'video': str(directory / f"seed_{report['seed']}_multiview.mp4"),
                }
                groups.setdefault(group, []).append(row)
    result = {
        'scope': 'Offline verification; frozen video frames excluded; no full handover tested',
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha256(args.checkpoint / 'model.safetensors'),
        'groups': {key: {'successes': sum(r['grasp_success'] for r in rows),
                         'episodes': len(rows), 'rows': rows} for key, rows in groups.items()},
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: {k: v for k, v in value.items() if k != 'rows'}
                      for key, value in result['groups'].items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
