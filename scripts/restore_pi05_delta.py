"""Restore an exact pi0.5 checkpoint from a verified base and parameter delta."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from flax import traverse_util
import orbax.checkpoint as ocp
from openpi.models.model import restore_params


def digest(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    manifest=json.loads((args.bundle/'delta_manifest.json').read_text())
    if manifest['format']!='pi05_exact_parameter_delta_v1':raise ValueError('Unsupported format')
    values=traverse_util.flatten_dict(restore_params(args.reference/'params',restore_type=np.ndarray))
    if set(values)!=set(tuple(r['path']) for r in manifest['leaves']):raise ValueError('Parameter structure differs')
    with np.load(args.bundle/'parameter_delta.npz',allow_pickle=False) as delta:
        for row in manifest['leaves']:
            key=tuple(row['path'])
            if digest(values[key])!=row['reference_sha256']:raise ValueError(f'Reference mismatch: {key}')
            if 'delta_array' in row:values[key]=delta[row['delta_array']].view(np.dtype(row['dtype'])).reshape(row['shape'])
            if digest(values[key])!=row['sha256']:raise ValueError(f'Reconstructed parameter mismatch: {key}')
    args.output.mkdir(parents=True)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save((args.output/'params').resolve(),{'params':traverse_util.unflatten_dict(values)})
    del values
    reloaded=traverse_util.flatten_dict(restore_params(args.output/'params',restore_type=np.ndarray))
    for row in manifest['leaves']:
        if digest(reloaded[tuple(row['path'])])!=row['sha256']:raise AssertionError('Saved checkpoint differs')
    for name in ('assets','train_state','project_identity.json','reload_verification.json','reload_first_actions.npy','_CHECKPOINT_METADATA'):
        source=args.bundle/name
        if source.is_dir():shutil.copytree(source,args.output/name)
        elif source.exists():shutil.copy2(source,args.output/name)
    (args.output/'delta_restore_verification.json').write_text(json.dumps({'passed':True,'all_parameter_arrays':len(reloaded),'source':str(args.bundle)},indent=2)+'\n')
    print('Verified reconstructed checkpoint:',args.output,flush=True)

if __name__=='__main__':main()
