"""Create a paired, offline summary from saved Zu7 checkpoint rollouts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--geometry-report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    protocol = json.loads((args.root/'zu7_matched_checkpoint_comparison_20261109.json').read_text())
    geometry = json.loads(args.geometry_report.read_text())
    geometry_by_trace = {Path(row['trace']).parent.name+'/'+Path(row['trace']).name: row
                         for row in geometry['reports']}
    summaries = []
    conditions_by_seed = {}
    all_offsets_bounded = True
    for run in protocol['runs']:
        experiment = run['name']
        tag = experiment.removeprefix('zu7_matched_').removesuffix('_seed203_207')
        directory = args.root/experiment
        audit_path = args.root/f'zu7_matched_{tag}_audit_20261109.json'
        audit = json.loads(audit_path.read_text())
        audit_rows = {int(row['seed']): row
                      for group in audit['groups'].values() for row in group['rows']}
        result_rows = {int(row['seed']): row
                       for row in json.loads((directory/'results.json').read_text())}
        metrics = {int(item['file'].split('_')[1]): item for item in
                   json.loads((directory/'control_metrics.json').read_text())}
        episodes = []
        for row in run['details']:
            seed = int(row['seed'])
            condition = result_rows[seed]['initial_condition']
            previous = conditions_by_seed.setdefault(seed, condition)
            if previous != condition:
                raise ValueError(f'Paired initial condition differs for seed {seed}')
            offsets = condition.get('left_joint_offset_rad', [])
            if len(offsets) != 6 or max(abs(float(x)) for x in offsets) > protocol['protocol']['initial_offset_rad']+1e-8:
                all_offsets_bounded = False
            trace_path = directory/f'seed_{seed}_trace.npz'
            geometry_key = experiment+'/'+trace_path.name
            geometry_row = geometry_by_trace.get(geometry_key)
            if geometry_row is None:
                raise ValueError(f'Missing geometry result for {geometry_key}')
            geometry_path = args.geometry_report.parent/(directory.name+'_'+trace_path.stem+'_geometry.npz')
            with np.load(trace_path) as trace, np.load(geometry_path) as fk:
                control = trace['control_trace']
                close = np.flatnonzero(control[:, 35] < .15)
                first_close = int(close[0]) if len(close) else len(control)-1
                object_xy = control[:, 57:59]
                displacement = np.linalg.norm(object_xy[:first_close+1]-object_xy[0], axis=1)
                measured_tcp = fk['measured_tcp'][:first_close+1, :2]
                raw_tcp = fk['raw_target_tcp'][:first_close+1, :2]
                target_xy = object_xy[:first_close+1]
                measured_error = np.linalg.norm(measured_tcp-target_xy, axis=1)
                raw_error = np.linalg.norm(raw_tcp-target_xy, axis=1)
                max_preclose_move = float(displacement.max())
                min_measured_error = float(measured_error.min())
                min_command_error = float(raw_error.min())
            eval_row = audit_rows[seed]
            episode = {
                'seed': seed,
                'grasp_success': bool(row['grasp_success']),
                'failure_type': (row['failure_reason'] or {}).get('type'),
                'max_lift_m': float(row['lift_m']),
                'verified_hold_s': float(eval_row['verified_hold_s']),
                'first_close_step': None if not len(close) else first_close,
                'target_horizontal_error_at_close_mm': (
                    None if not geometry_row['close_events'] else
                    float(geometry_row['close_events'][0]['raw_target_horizontal_error_mm'])),
                'measured_aperture_at_close': (
                    None if not geometry_row['close_events'] else
                    float(geometry_row['close_events'][0]['measured_aperture'])),
                'object_max_displacement_before_close_m': max_preclose_move,
                'best_measured_tcp_horizontal_error_before_close_mm': min_measured_error*1000,
                'best_command_tcp_horizontal_error_before_close_mm': min_command_error*1000,
                'fk_measured_max_error_m': float(geometry_row['fk_measured_max_error_m']),
                'max_measured_joint_speed_rad_s': metrics[seed]['max_measured_joint_speed_rad_s'],
                'p99_measured_joint_speed_rad_s': metrics[seed]['p99_measured_joint_speed_rad_s'],
                'max_command_tracking_error_rad': eval_row['max_tracking_error_rad'],
                'initial_condition': condition,
                'trace_sha256': sha256(trace_path),
            }
            episodes.append(episode)
        summaries.append({
            'checkpoint': tag,
            'checkpoint_sha256': audit['checkpoint_sha256'],
            'successes': sum(ep['grasp_success'] for ep in episodes),
            'episodes': len(episodes),
            'episodes_detail': episodes,
        })

    if not all_offsets_bounded:
        raise ValueError('At least one tested initial offset is missing or exceeds the declared bound')
    result = {
        'scope': 'Offline paired development diagnostic; one red block, left grasp-lift-hold only; no full handover, language-choice test, or real robot.',
        'protocol': protocol['protocol'],
        'paired_initial_conditions_identical_across_checkpoints': True,
        'all_left_joint_offsets_within_declared_bound': True,
        'checkpoints': summaries,
        'interpretation': [
            'Five episodes per checkpoint are too few for significance or a generalization claim.',
            'Training datasets and update counts differ, so the checkpoint comparison is not a causal ablation.',
            'Object displacement and TCP-to-object errors use simulator truth offline and do not represent policy inputs.',
        ],
        'geometry_report_sha256': sha256(args.geometry_report),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({s['checkpoint']: f"{s['successes']}/{s['episodes']}" for s in summaries}, ensure_ascii=False))


if __name__ == '__main__':
    main()
