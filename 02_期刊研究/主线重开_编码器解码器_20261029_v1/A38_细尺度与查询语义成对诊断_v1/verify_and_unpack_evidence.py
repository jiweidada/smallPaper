from pathlib import Path
import hashlib,json,sys,zipfile
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent
log=Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文\前期调研_码率受限航拍小目标_20261028_v1\远端验证_20261029_v1\ssh_operation_receipts.jsonl')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[]
for line in log.open(encoding='utf-8'):
    try:r=json.loads(line)
    except ValueError:continue
    if r.get('id')=='A38_package_completed_native_evidence_25' and r.get('action')=='exec':rows.append(r)
assert len(rows)==1 and rows[0]['exit_code']==0
receipt=json.loads(rows[0]['stdout'])
archive=D/'native_semantic_evidence.zip'
assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
with (D/'remote_archive_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
out=D/'downloaded_evidence';out.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:
    manifest=json.loads(z.read('evidence_manifest.json'))
    assert set(z.namelist())==set(manifest['files_sha256'])|{'evidence_manifest.json'}
    assert len(z.namelist())==receipt['files']==154
    for name in z.namelist():
        target=(out/name).resolve();assert target.is_relative_to(out.resolve())
        raw=z.read(name)
        if name!='evidence_manifest.json':assert hashlib.sha256(raw).hexdigest()==manifest['files_sha256'][name],name
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(raw)
for name in ['execution_inputs_manifest.json','export_recovery_execution_manifest.json']:
    assert sha(out/name)==sha(D/name),name
    for n,h in json.loads((out/name).read_text(encoding='utf-8'))['files_sha256'].items():
        assert sha(D/n)==h and sha(out/n)==h,n
collection=json.loads((out/'collection_result.json').read_text())
old=json.loads((out/'worker_exit.json').read_text());new=json.loads((out/'recovery_worker_exit.json').read_text())
assert old['worker_exit_code']==1 and old['author_entries']==old['author_completed']==2
assert new['worker_exit_code']==0 and new['combined_author_entries']==new['combined_author_completed']==114
assert collection['entries']==collection['completed']==114 and collection['state_unchanged']
entries=[];completed=[];exported=[]
for name in ['forward_entries.jsonl','recovery_forward_entries.jsonl']:
    for line in (out/name).read_text().splitlines():
        r=json.loads(line)
        if r['event']=='author_detector_entry':entries.append(r['entry'])
        if r['event']=='author_detector_completed':completed.append(r['entry'])
        if r['event']=='case_exported':exported.append(r['case'])
assert entries==completed==list(range(1,115))
assert exported==list(range(112))
assert len(list((out/'captures').glob('*.npz')))==112
result=dict(status='PASS_ARCHIVE_AND_COUNTERS_WITH_DECLARED_EXPORT_DEVIATION',archive_files=154,
    manifest_files_verified=153,whole_detector_entries=114,whole_detector_completed=114,exported_cases=112,
    original_failed_exit=1,recovery_exit=0,source_and_frozen_analysis_files_unchanged=True,
    full_query_feature_repeat_available=False,detector_optimizer_updates=0,
    archive_sha256=receipt['sha256'],peak_cuda_allocated_bytes=collection['peak_cuda_allocated_bytes'],
    peak_scope='Recovery process only; original failed two-call process did not export its peak statistic.')
with (D/'archive_verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
