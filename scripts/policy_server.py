"""Trusted local Unix-socket inference bridge between isolated Python environments.

The containing directory is private (0700). Only connect local project processes.
No data or model is sent to an external service.
"""
import argparse
import os
import time
from multiprocessing.connection import Listener
from pathlib import Path
import numpy as np
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

ROOT = Path('/data/jaka-vla')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    args = p.parse_args()
    run = ROOT / 'run'
    run.mkdir(mode=0o700, exist_ok=True)
    run.chmod(0o700)
    address = run / 'policy.sock'
    if address.exists():
        raise FileExistsError(f'Remove stale socket only after checking no server is running: {address}')
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to('cuda').eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(args.checkpoint),
        preprocessor_overrides={'device_processor': {'device': 'cuda'}})
    with Listener(str(address), family='AF_UNIX') as listener:
        print(f'READY {address}', flush=True)
        with listener.accept() as conn:
            while True:
                request = conn.recv()
                if request['op'] == 'close':
                    conn.send({'closed': True})
                    break
                if request['op'] == 'reset':
                    torch.manual_seed(request['seed'])
                    policy.reset()
                    conn.send({'reset': True})
                    continue
                obs = request['observation']
                frame = {'observation.state': torch.from_numpy(np.asarray(obs['state'], np.float32)),
                         'task': obs['instruction']}
                for i, rgb in enumerate(obs['images'], 1):
                    frame[f'observation.images.camera{i}'] = torch.from_numpy(
                        np.asarray(rgb).copy()).permute(2, 0, 1).float() / 255.0
                with torch.inference_mode():
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    batch = pre(frame)
                    actions = post(policy.predict_action_chunk(batch))
                    torch.cuda.synchronize()
                    duration = time.perf_counter() - start
                array = actions[0].float().cpu().numpy()
                if array.shape != (policy.config.chunk_size, 14) or not np.isfinite(array).all():
                    raise RuntimeError(f'Invalid action chunk: {array.shape}')
                conn.send({'actions': array, 'inference_seconds': duration})


if __name__ == '__main__':
    main()
