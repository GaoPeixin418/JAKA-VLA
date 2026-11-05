from pathlib import Path
import yaml

root = Path('/data/jaka-vla/repos/RoboTwin')
config = yaml.safe_load((root / 'env_cfg/task_config/demo_clean.yml').read_text())
config.update(episode_num=1, save_freq=10, language_num=5, jaka_uniform_sampling=True,
              save_path='/data/jaka-vla/data/raw', clear_cache_freq=1)
for name, count in [('jaka_smoke', 1), ('jaka_pilot', 50)]:
    config['episode_num'] = count
    (root / f'env_cfg/task_config/{name}.yml').write_text(yaml.safe_dump(config, sort_keys=False))
    Path(f'/data/jaka-vla/configs/{name}.yml').write_text(yaml.safe_dump(config, sort_keys=False))
print('Created one-episode smoke and 50-episode pilot configs.')
