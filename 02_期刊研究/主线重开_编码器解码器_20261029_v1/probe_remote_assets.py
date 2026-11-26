"""Finite CPU asset check. Never enables unsafe checkpoint unpickling."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import tarfile
import torch

ROOT = Path(__file__).resolve().parent
assets = ROOT / 'assets'
out = ROOT / 'receipts'
out.mkdir(exist_ok=True)
manifest = json.loads((assets / 'probe_package_manifest.json').read_text())
for entry in [manifest['source_archive'], manifest['weight']]:
    p = assets / entry['name']
    assert p.stat().st_size == entry['bytes']
    assert hashlib.sha256(p.read_bytes()).hexdigest() == entry['sha256'], p.name
with tarfile.open(assets / manifest['source_archive']['name']) as tf:
    members = tf.getmembers()
    expected = {'source/' + row['path'] for row in manifest['files']}
    assert {m.name for m in members} == expected and len(members) == len(expected)
    for m in members:
        assert m.isfile() and not m.issym() and not m.islnk()
        parts = PurePosixPath(m.name)
        assert not parts.is_absolute() and '..' not in parts.parts
        p = ROOT.joinpath(*parts.parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = tf.extractfile(m).read()
        if p.exists():
            assert p.read_bytes() == data
        else:
            p.write_bytes(data)
for row in manifest['files']:
    data = (ROOT / 'source' / row['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == row['sha256']
    assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == row['git_blob_sha']
ckpt = torch.load(assets / manifest['weight']['name'], map_location='cpu', weights_only=True)
assert isinstance(ckpt, dict)
states = {}
def inspect_state(name, value):
    if isinstance(value, dict):
        tensors = {k: {'shape': list(v.shape), 'dtype': str(v.dtype)}
                   for k, v in value.items() if isinstance(v, torch.Tensor)}
        if len(tensors) > 50:
            states[name] = tensors
        else:
            for k, v in value.items():
                if isinstance(v, dict):
                    inspect_state(name + '.' + str(k), v)
for key, value in ckpt.items():
    inspect_state(str(key), value)
result = {'status': 'SOURCE_AND_WEIGHT_BYTES_VERIFIED_SAFE_CPU_CHECKPOINT_READ',
          'source_commit': manifest['source_commit'], 'source_files': len(manifest['files']),
          'weight_sha256': manifest['weight']['sha256'],
          'checkpoint_top_keys': [str(x) for x in ckpt],
          'state_candidates': states,
          'simple_metadata': {str(k): v for k, v in ckpt.items() if isinstance(v, (str, int, float, bool, type(None)))},
          'torch': torch.__version__, 'cuda_model_forwards': 0, 'training_updates': 0}
(out / 'asset_checkpoint_check.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != 'state_candidates'}))
for name, values in states.items():
    print(json.dumps({'state': name, 'tensors': len(values), 'classification_heads':
          {k: v for k, v in values.items() if 'enc_score_head' in k or 'dec_score_head' in k}}))
