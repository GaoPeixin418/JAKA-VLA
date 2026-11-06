"""Nine bounded development rollouts, separating ordinary and recovery starts."""
from datetime import datetime
import json
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path('/data/jaka-vla')


def main():
    checkpoint = ROOT/'checkpoints/smolvla_zu7_recovery4000_seed42/checkpoints/last/pretrained_model'
    profile = json.loads((checkpoint.parent.parent.parent/'resource_profile.json').read_text())
    if not profile['completed']:
        raise ValueError('Training did not complete')
    runs = [('zu7_recovery4000_seen_seed0_1', 0, 2, []),
            ('zu7_recovery4000_seen_seed6', 6, 1, []),
            ('zu7_recovery4000_new_seed200_202', 200, 3, ['--initial-offset', '.04'])]
    for seed, step in [(0,50),(6,35),(101,45)]:
        probe = ROOT/f'results/zu7_recovery_collect_seed{seed}_step{step}'
        source = json.loads((probe/'raw/robot_identity.json').read_text())
        condition = ROOT/f'configs/recovery_start_seed{seed}_step{step}.json'
        if condition.exists():
            raise FileExistsError(condition)
        condition.write_text(json.dumps(source['initial_condition'], indent=2)+'\n')
        runs.append((f'zu7_recovery4000_intervened_seed{seed}_step{step}',seed,1,
                     ['--initial-condition-file',str(condition)]))
    completed = []
    for name, seed, episodes, extra in runs:
        now = datetime.now(ZoneInfo('Asia/Shanghai'))
        if (now.hour, now.minute) >= (18,20):
            raise TimeoutError('Stop launching rollouts at backup cutoff')
        command = ['python3','scripts/run_zu7_eval.py','--name',name,'--checkpoint',str(checkpoint),
                   '--seed-start',str(seed),'--episodes',str(episodes),'--k','8','--max-steps','250',
                   '--review-video','--task-scope','left_grasp_lift_hold',
                   '--instruction','Use the left arm to grasp the red block, lift it, and hold it.',*extra]
        subprocess.run(command,cwd=ROOT,check=True,timeout=300)
        completed.append(name)
        (ROOT/'results/zu7_recovery4000_evaluation_manifest.json').write_text(json.dumps(completed,indent=2)+'\n')


if __name__=='__main__':main()
