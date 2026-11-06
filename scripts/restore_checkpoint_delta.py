"""Reconstruct an exact safetensors file from a verified earlier model and delta."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--base', type=Path, required=True)
p.add_argument('--delta', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
m = json.loads((a.delta/'manifest.json').read_text())
base = a.base.read_bytes()
assert hashlib.sha256(base).hexdigest() == m['base_sha256'], 'Wrong base checkpoint'
bl = struct.unpack('<Q', base[:8])[0]
bh = json.loads(base[8:8+bl])
header = (a.delta/'header.bin').read_bytes()
th = json.loads(header[8:])
changed = (a.delta/'changed-tensors.bin').read_bytes()
changes = {c['key']: c for c in m['changes']}
size = max(v['data_offsets'][1] for k, v in th.items() if k != '__metadata__')
result = bytearray(header) + bytearray(size)
for key, info in th.items():
    if key == '__metadata__':
        continue
    start, end = info['data_offsets']
    if key in changes:
        c = changes[key]
        raw = changed[c['offset']:c['offset']+c['length']]
    else:
        old = bh[key]
        assert old['shape'] == info['shape'] and old['dtype'] == info['dtype']
        bs, be = old['data_offsets']
        raw = base[8+bl+bs:8+bl+be]
    assert len(raw) == end-start
    result[len(header)+start:len(header)+end] = raw
assert hashlib.sha256(result).hexdigest() == m['target_sha256'], 'Reconstruction SHA256 mismatch'
with a.output.open('xb') as f:
    f.write(result)
print('Exact model SHA256 verified:', m['target_sha256'])
