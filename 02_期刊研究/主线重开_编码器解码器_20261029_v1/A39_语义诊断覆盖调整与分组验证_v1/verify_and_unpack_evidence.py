from pathlib import Path
import hashlib,json,sys,zipfile
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
log=Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文\前期调研_码率受限航拍小目标_20261028_v1\远端验证_20261029_v1\ssh_operation_receipts.jsonl')
rows=[]
for line in log.open(encoding='utf-8'):
    try:r=json.loads(line)
    except ValueError:continue
    if r.get('id')=='A39_package_native_evidence_11' and r.get('action')=='exec':rows.append(r)
assert len(rows)==1 and rows[0]['exit_code']==0
receipt=json.loads(rows[0]['stdout']);archive=D/'native_semantic_evidence.zip'
assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
with (D/'remote_archive_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
out=D/'downloaded_evidence';out.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:
    manifest=json.loads(z.read('evidence_manifest.json'))
    assert set(z.namelist())==set(manifest['files_sha256'])|{'evidence_manifest.json'}
    assert len(z.namelist())==receipt['files']==137
    for name in z.namelist():
        target=(out/name).resolve();assert target.is_relative_to(out.resolve());raw=z.read(name)
        if name!='evidence_manifest.json':assert hashlib.sha256(raw).hexdigest()==manifest['files_sha256'][name],name
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(raw)
assert sha(D/'execution_inputs_manifest.json')==sha(out/'execution_inputs_manifest.json')
original=json.loads((D/'execution_inputs_manifest.json').read_text(encoding='utf-8'))
for name,h in original['files_sha256'].items():assert sha(D/name)==sha(out/name)==h,name
cases=json.loads((D/'combined_cases_FROZEN.json').read_text(encoding='utf-8'))['cases']
for row in cases:
    if row['capture_phase']=='A38':assert sha(row['capture_path'])==row['capture_sha256']
collection=json.loads((out/'collection_result.json').read_text())
worker=json.loads((out/'worker_exit.json').read_text())
assert worker['worker_exit_code']==0 and worker['author_entries']==worker['author_completed']==113
assert collection['entries']==collection['completed']==113 and collection['state_unchanged']
assert collection['all_first_new_case_arrays_bitwise_repeat_equal'] and collection['full_query_feature_repeat_available']
events=[json.loads(x) for x in (out/'forward_entries.jsonl').read_text().splitlines()]
assert [x['entry'] for x in events if x['event']=='author_detector_entry']==list(range(1,114))
assert [x['entry'] for x in events if x['event']=='author_detector_completed']==list(range(1,114))
inputs=json.loads((D/'collector_inputs.json').read_text(encoding='utf-8'))
assert [x['case'] for x in events if x['event']=='case_exported']==[r['index'] for r in inputs['cases']]
first=dict(np.load(out/('captures/case_%03d.npz'%inputs['cases'][0]['index']),allow_pickle=False))
repeat=dict(np.load(out/'repeat_first_new.npz',allow_pickle=False))
assert set(first)==set(repeat)
for key in first:np.testing.assert_array_equal(first[key],repeat[key])
result=dict(status='PASS_A39_ARCHIVE_SOURCE_COUNTER_AND_FULL_NEW_QUERY_REPEAT',archive_files=137,manifest_files=136,
    new_native_entries=113,new_unique_cases=112,reused_A38_caches_verified=112,combined_unique_cases=224,
    repeat_arrays_bitwise_equal=len(first),state_unchanged=True,detector_optimizer_updates=0,
    archive_sha256=receipt['sha256'],old_A38_cap_closed=True,
    inherited_limit='OldA38queryfullrepetitionunavailable;newA39firstcasefullrepeatverified.')
with (D/'archive_verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
