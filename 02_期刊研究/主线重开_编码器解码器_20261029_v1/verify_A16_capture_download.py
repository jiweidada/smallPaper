"""Verify and extract the completed capture; no detector/sampling-core replay."""
from pathlib import Path,PurePosixPath
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,tarfile,zipfile
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'A16_实际特征空间变化审计_v1'
assert not (OUT/'capture_download_verification.json').exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
archive=OUT/'backup'/'A16_native_patches_and_source_checks.tar.gz'
assert archive.stat().st_size==80280884
assert sha(archive)=='b18062de4d682eeb067fee0c385c2bdc0a50f8eca6d2882ceefbf15eddf96b2a'
with tarfile.open(archive,'r:gz') as tar:
    members=tar.getmembers();assert len(members)==53
    assert len({m.name for m in members})==53
    for member in members:
        rel=PurePosixPath(member.name);assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
        path=OUT.joinpath(*rel.parts);assert path.resolve().is_relative_to(OUT.resolve())
        data=tar.extractfile(member).read()
        if path.exists():assert path.read_bytes()==data,member.name
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
manifest=json.loads((OUT/'capture_completed.json').read_text(encoding='utf-8'))
for f in manifest['files']:
    path=OUT/f['file'];assert path.stat().st_size==f['bytes'] and sha(path)==f['sha256']
for f in json.loads((OUT/'freeze_manifest.json').read_text(encoding='utf-8'))['files']:
    assert sha(OUT/f['file'])==f['sha256']
for f in (OUT/'new_patches').glob('*.npz'):
    with zipfile.ZipFile(f) as z:assert z.testzip() is None
capture=json.loads((OUT/'capture_summary.json').read_text(encoding='utf-8'))
assert capture['state_digest_before']==capture['state_digest_after']=='82c3ba49d69370a3d0105e47fc0ab5d715a9e7b1ad15feb275b1b497b5b0ccce'
assert capture['new_full_model_forwards']==26 and capture['main_model_updates']==0
calls=[json.loads(s) for s in (OUT/'full_forward_calls.jsonl').read_text().splitlines()]
assert len(calls)==26 and [x['call'] for x in calls]==list(range(1,27))
assert Counter(x['kind'] for x in calls)=={'PLAIN_IDENTITY':1,'NATIVE_PATCH_CAPTURE':25}
assert [x['case'] for x in calls if x['kind']=='NATIVE_PATCH_CAPTURE']==list(range(25))
operators=Counter(json.loads(s)['kind'] for s in (OUT/'source_operator_calls.jsonl').read_text().splitlines())
assert operators=={'AUTHOR_CPU_DOUBLE_SAMPLING_CORE':25,'CPU_DOUBLE_OFFSET_LINEAR':5,'CPU_DOUBLE_ATTENTION_LINEAR':5}
exit_receipt=json.loads((OUT/'process'/'a16_spatial_response_v1_exit.json').read_text())
assert exit_receipt['exit_code']==0 and len(exit_receipt['stages'])==3
assert all(s['exit_code']==0 for s in exit_receipt['stages'])
receipt={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_DOWNLOADED_CAPTURE_HASHES_AND_BOUNDED_TERMINAL_EXECUTION',
    'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'archive_members':53,
    'manifest_files_verified':len(manifest['files']),'new_complete_model_forwards':26,
    'cumulative_complete_model_forwards':252,'new_suffix_forwards':0,
    'source_core_calls':25,'separate_linear_calls':10,'actual_exit_code':0,'main_model_updates':0,
    'new_reexecution_in_verifier':0,'response_metrics_computed':False,
    'scope':'Integrity,lineage and budget verification;not a fresh detector execution or result replication.'}
(OUT/'capture_download_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
