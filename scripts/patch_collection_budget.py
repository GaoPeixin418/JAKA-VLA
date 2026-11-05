"""Make expert seed search honor an explicit attempt budget when configured."""
from pathlib import Path
p=Path('/data/jaka-vla/repos/RoboTwin/scripts/collect_data.py')
s=p.read_text()
if 'max_seed_attempts' not in s:
    old='        while suc_num < args["episode_num"]:\n'
    new='''        attempt_start_seed = epid
        while suc_num < args["episode_num"]:
            max_attempts = args.get("max_seed_attempts")
            if max_attempts is not None and epid - attempt_start_seed >= max_attempts:
                raise RuntimeError(f"Expert seed budget exhausted: {suc_num}/{args['episode_num']} successes in {max_attempts} attempts")
'''
    assert old in s;s=s.replace(old,new,1);p.write_text(s)
print('Collection attempt budget installed.')
