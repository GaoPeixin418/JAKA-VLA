"""Save changed safetensors tensors relative to a separately verified model backup."""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    b, t = a.base.read_bytes(), a.target.read_bytes()
    bl, tl = struct.unpack('<Q', b[:8])[0], struct.unpack('<Q', t[:8])[0]
    bh, th = json.loads(b[8:8+bl]), json.loads(t[8:8+tl])
    (a.output/'header.bin').write_bytes(t[:8+tl])
    changes = []
    with (a.output/'changed-tensors.bin').open('xb') as f:
        for key, info in th.items():
            if key == '__metadata__':
                continue
            ts, te = info['data_offsets']
            value = t[8+tl+ts:8+tl+te]
            old = bh.get(key)
            same = False
            if old and old['dtype'] == info['dtype'] and old['shape'] == info['shape']:
                bs, be = old['data_offsets']
                same = value == b[8+bl+bs:8+bl+be]
            if not same:
                changes.append({'key': key, 'offset': f.tell(), 'length': len(value)})
                f.write(value)
    m = {'base_project_path': str(a.base), 'base_sha256': hashlib.sha256(b).hexdigest(),
         'target_project_path': str(a.target), 'target_sha256': hashlib.sha256(t).hexdigest(),
         'changes': changes, 'changed_bytes': (a.output/'changed-tensors.bin').stat().st_size}
    (a.output/'manifest.json').write_text(json.dumps(m, indent=2)+'\n')
    print(json.dumps({'changed_bytes': m['changed_bytes'], 'tensor_count': len(changes), 'target_sha256': m['target_sha256']}))


if __name__ == '__main__':
    main()
