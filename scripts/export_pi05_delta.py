"""Make an exact parameter delta against a separately backed-up qualification model.

This is a backup encoding, not quantization. All reconstructed leaf bytes must match.
The base checkpoint remains a required, explicitly hashed dependency.
"""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
from flax import traverse_util
from openpi.models.model import restore_params


def digest(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    reference=traverse_util.flatten_dict(restore_params(args.reference/'params',restore_type=np.ndarray))
    target=traverse_util.flatten_dict(restore_params(args.checkpoint/'params',restore_type=np.ndarray))
    if set(reference)!=set(target):raise ValueError('Parameter structure differs')
    leaves=[];arrays={};reference_hashes={};changed_elements=0
    for i,(key,value) in enumerate(target.items()):
        value=np.asarray(value);base=np.asarray(reference[key]);sha=digest(value);base_sha=digest(base)
        if value.shape!=base.shape or value.dtype!=base.dtype:raise ValueError(f'Layout differs: {key}')
        record={'path':list(key),'shape':list(value.shape),'dtype':str(value.dtype),'sha256':sha,'reference_sha256':base_sha}
        if sha!=base_sha:
            name=f'array_{i:04d}';arrays[name]=np.ascontiguousarray(value).view(np.uint8).reshape(-1)
            record['delta_array']=name;changed_elements+=value.size
        leaves.append(record)
    if changed_elements>52_153_376:raise ValueError('Changes exceed the audited trainable parameter budget')
    np.savez_compressed(args.output/'parameter_delta.npz',**arrays)
    # Verify the exact bytes that were saved, not only the in-memory dictionary.
    with np.load(args.output/'parameter_delta.npz',allow_pickle=False) as saved:
        for row in leaves:
            value=reference[tuple(row['path'])]
            if 'delta_array' in row:value=saved[row['delta_array']].view(np.dtype(row['dtype'])).reshape(row['shape'])
            if digest(value)!=row['sha256']:raise AssertionError(row['path'])
    for name in ('assets','train_state','project_identity.json','reload_verification.json','reload_first_actions.npy','_CHECKPOINT_METADATA'):
        source=args.checkpoint/name
        if source.is_dir():shutil.copytree(source,args.output/name)
        elif source.exists():shutil.copy2(source,args.output/name)
    report={'format':'pi05_exact_parameter_delta_v1','checkpoint':str(args.checkpoint),'reference_checkpoint':str(args.reference),
            'reference_local_archive':'outputs/20261111/pi05/qualification_checkpoint_and_results.tar',
            'reference_archive_sha256':'9438701e4e2007d936fa61361fff665c9477e04bfcd8fa8c9a3dc4a7f3f266a7',
            'changed_arrays':len(arrays),'changed_elements':int(changed_elements),'all_parameter_bytes_reconstructed':True,'leaves':leaves,
            'scope':'exact model parameters, optimizer, normalization and identity; requires separately verified reference archive'}
    (args.output/'delta_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='leaves'}),flush=True)

if __name__=='__main__':main()
