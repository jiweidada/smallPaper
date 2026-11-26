"""Preserve readonly-reader key error; consume raw observer keys, not display aliases."""
from pathlib import Path
import hashlib,json
ROOT=Path.cwd();old=ROOT/'outputs/review_p2_subspace_diagnostic12_20261011_v1.py'
out=ROOT/'outputs/review_p2_subspace_diagnostic12_20261011_v2.py'
source=old.read_text(encoding='utf8')
assert "d['source_ok']" in source
source=source.replace("d['source_ok']","d['source_verified']")
source=source.replace("    with p.open('xb') as f:f.write(decoded)","    if p.exists():assert p.read_bytes()==decoded,'Existing byte-exact artifact changed'\n    else:\n        with p.open('xb') as f:f.write(decoded)")
manifest="write(dest/'manifest.json',dict(files={p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()}))\n"
assert source.count(manifest)==1
source=source.replace(manifest,'')+"\n"+manifest
with out.open('x',encoding='utf8') as f:f.write(source)
failure=dict(status='READONLY_LOCAL_REVIEW_V1_KEY_ERROR_PRESERVED',source_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),exception="KeyError: 'source_ok'",line=58,
    cause='Display adapter source_ok differs from byte-exact remote observation source_verified. v2 uses the actual schema, checks already-written artifact bytes and resumes only local bookkeeping.',
    actual_CPU_worker_repeated=False,new_GPU_dispatches=0,new_training_epochs=0,new_BASE_forwards=0,science_changed=False)
with (ROOT/'outputs/p2_subspace_diagnostic12_20261011_v1/local_review_failure_v1.json').open('x',encoding='utf8') as f:json.dump(failure,f,indent=2)
print(json.dumps(dict(status='V2_READONLY_READER_PREPARED_ORIGINAL_FAILURE_PRESERVED',source_sha256=hashlib.sha256(out.read_bytes()).hexdigest())))
