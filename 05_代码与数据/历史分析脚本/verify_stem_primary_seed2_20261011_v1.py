"""Verify and unpack the actual primary12 artifacts; no replay-gate substitution."""
from pathlib import Path,PurePosixPath
import base64,hashlib,importlib.util,json,stat,sys,zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stem_primary_collection_20261011_v1/seed2_original_stem_v1'
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
p=json.loads("{\"base64\":\"eyJzdGF0dXMiOiAiQ09MTEVDVEVEX1NUT1BQRURfRVhBQ1QxMl9QUklNQVJZX0FSVElGQUNUUyIsICJzZWVkIjogMiwgImFybSI6ICJvcmlnaW5hbF9zdGVtIiwgImFyY2hpdmUiOiAiLzI0MDg1NDAzMDE0L0RFSU0vZXhwZXJpbWVudHMvc3RlbV9wcmltYXJ5X2NvbGxlY3Rpb25fMjAyNjEwMTFfdjEvc2VlZDJfb3JpZ2luYWxfc3RlbV92MS9wcmltYXJ5X2FydGlmYWN0c192MS56aXAiLCAiYnl0ZXMiOiAxODYzNDk1NjAsICJzaGEyNTYiOiAiMmUyY2FhYzRkZGRhN2NjN2NjY2MwMTgzMDhlMWYwZTJjMTk3ZmQ5ODhjNzg2ZGY0ZjEyMTU0YWQ5ZDEwYTZkZSIsICJtYW5pZmVzdF9zaGEyNTYiOiAiZTBjODcxNDVhMTgxZGY1NzRhNjU1NWZiMmViNzczNzczYTUzMjkzNTJkZTQyMzNhNGMzM2Q4ZTk1MzRmMTc1OSIsICJhcmNoaXZlX2VudHJpZXMiOiA2MDYsICJsYXN0X2NoZWNrcG9pbnRfc2hhMjU2IjogIjNhN2I5NGVhZTJiOGRmMzQ5MGViMDZmYzJmNzcyN2ZkMTliYTc1NTk4MDcyMmUyMDZjMGE0NmM4NzMzMmI5NGEiLCAibmF0aXZlX3N0YXRzIjogWzAuMDc5MDgxMzcyOTA3NTg1NzgsIDAuMTQ4NTU3MDg3MDQ4NDM5ODQsIDAuMDczOTczMTUzNDQ2ODQ1OTQsIDAuMDUxNjA2MTQ3MzcyNzE5MzcsIDAuMTEyOTcwMDY1MTkzMDM2NCwgMC4xNDIyNjMyNTQ3OTc2MDkxLCAwLjA0ODI3OTU4MjAwNjM4MDY5NiwgMC4xNDYxMzY3MTAxMzU5ODE0LCAwLjIxMTk0MDQ1MjE2NjAwMjU3LCAwLjE0ODMzNDM0NzA1NjU3MDc1LCAwLjMwNTMwODQ3NjYwODExNjU1LCAwLjMwMjM3MDg5ODMyMzUzMjE0XSwgIm5hdGl2ZV9nYXRlX3Bhc3NlZCI6IGZhbHNlLCAic2F2ZWRfRU1BX2luZGVwZW5kZW50X3JlcGxheV9pbmNsdWRlZCI6IGZhbHNlLCAiY29sbGVjdGVkX3JlbW90ZV9VVEMiOiAiMjAyNi0xMC0xMVQxMjoxNzowMC4xMTgyNDMrMDA6MDAiLCAiY29sbGVjdG9yX3NoYTI1NiI6ICJlYjZkYzYxNGMyOWE4MGMyZWY4NjViYjZiOTRmMzA3OTk0YTVhMmEzYmViNTVkNDBjZmQ0ZjAyMTU4MDI1NWY1IiwgIm5ld19tb2RlbF9mb3J3YXJkcyI6IDAsICJuZXdfZXZhbHVhdG9yX2NhbGxzIjogMCwgIm5ld190cmFpbmluZ19lcG9jaHMiOiAwfQ==\",\"sha256\":\"39ae744ad4f7e29906b75e6f68e39d8ee2d0d618235690882e59695a321c798a\",\"bytes\":1093}");b=base64.b64decode(p['base64'],validate=True)
assert len(b)==p['bytes'] and hashlib.sha256(b).hexdigest()==p['sha256']
with (OUT/'remote_collection_metadata_v1.json').open('xb') as f:f.write(b)
meta=json.loads(b);assert meta['status']=='COLLECTED_STOPPED_EXACT12_PRIMARY_ARTIFACTS' and meta['seed']==2 and meta['arm']=='original_stem'
assert not meta['saved_EMA_independent_replay_included']
assert meta['new_model_forwards']==meta['new_evaluator_calls']==meta['new_training_epochs']==0
assert meta['collector_sha256']==sha(ROOT/'英文小论文/code/stem_primary_collection_20261011_v1/collect.py')
archive=OUT/'primary_artifacts_v1.zip'
assert archive.stat().st_size==meta['bytes'] and sha(archive)==meta['sha256']
destination=OUT/'extracted';assert destination.resolve().is_relative_to(ROOT.resolve())
with zipfile.ZipFile(archive) as z:
    names=z.namelist()
    assert len(names)==len(set(names))==len({n.casefold() for n in names})==meta['archive_entries']
    for info in z.infolist():
        n=info.filename;path=PurePosixPath(n)
        assert not info.is_dir() and stat.S_ISREG(info.external_attr>>16)
        assert not path.is_absolute() and '..' not in path.parts and '\\' not in n and ':' not in n and path.as_posix()==n
    raw=z.read('primary_manifest.json');assert hashlib.sha256(raw).hexdigest()==meta['manifest_sha256']
    manifest=json.loads(raw);assert manifest['schema']=='exact12_primary_artifact_collection_v1'
    assert set(names)==set(manifest['files'])|{'primary_manifest.json'}
    identity=manifest['identity'];assert identity['arm']=='original_stem' and identity['seed']==2
    assert manifest['completed_epochs']==12 and manifest['epoch_zero_based']==11 and manifest['evaluated_weights']=='EMA'
    assert manifest['stage_images']==548 and manifest['new_model_forwards']==manifest['new_training_epochs']==0
    assert not manifest['saved_EMA_independent_replay_included']
    allowed=read(ROOT/'英文小论文/code/native_selection_diagnostic12_20261011_v1/diagnostic_protocol.json')['allowed_native_roots']['original_stem12_20261011_v1']['files']
    assert {n[5:]:m['sha256'] for n,m in manifest['files'].items() if n.startswith('code/')}==allowed
    assert manifest['last_checkpoint_sha256']==meta['last_checkpoint_sha256']==manifest['files']['run/last.pth']['sha256']
    assert manifest['native_stats']==meta['native_stats']
    destination.mkdir(exist_ok=False)
    for n in names:
        target=destination/n;assert target.resolve().is_relative_to(destination.resolve())
        target.parent.mkdir(parents=True,exist_ok=True)
        h=hashlib.sha256();length=0
        with z.open(n) as reader,target.open('xb') as writer:
            while True:
                chunk=reader.read(8*1024**2)
                if not chunk:break
                writer.write(chunk);h.update(chunk);length+=len(chunk)
        if n=='primary_manifest.json':assert h.hexdigest()==meta['manifest_sha256']
        else:
            expected=manifest['files'][n];assert length==expected['bytes'] and h.hexdigest()==expected['sha256'],n
run=destination/'run'
assert sha(run/'training_completion.json')==manifest['source_training_completion_sha256']
done=read(run/'training_completion.json')
assert done['status']=='NATIVE_FIT_FINISHED_EXACT12' and done['entered_epochs']==list(range(12)) and done['completed_epochs']==12
assert done['identity']==identity and done['last_checkpoint_sha256']==sha(run/'last.pth')==meta['last_checkpoint_sha256']
assert sha(run/'stages_epoch12/manifest.json')==done['stage_manifest_sha256']
assert sha(run/'endpoint_epoch12/manifest.json')==done['endpoint_manifest_sha256']
assert read(destination/'screen/candidate_completion.json')['exit_code']==0
assert read(destination/'screen/controller_completion.json')['status']=='COMPLETE_FIXED12'
assert read(run/'first100_parity.json')['all100_rows_exact']
sys.path.insert(0,str(ROOT/'英文小论文/code/original_stem12_20261011_v1'))
from stage_trace import verify_trace
from results import load_endpoint
stage=verify_trace(run/'stages_epoch12',identity)
endpoint,flags,_=load_endpoint(run/'endpoint_epoch12')
assert endpoint['identity']==identity and endpoint['native_stats']==meta['native_stats']
rows=[json.loads(line) for line in (run/'log.txt').read_text().splitlines() if line]
assert [r['epoch'] for r in rows]==list(range(12)) and rows[-1]['test_coco_eval_bbox']==endpoint['native_stats']
assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
result=dict(status='PASS_DOWNLOADED_EXACT12_PRIMARY_ARTIFACTS_ONLY',identity=identity,
    downloaded_packet=str(destination),archive_sha256=meta['sha256'],primary_manifest_sha256=meta['manifest_sha256'],
    last_checkpoint_sha256=sha(run/'last.pth'),native_stats=endpoint['native_stats'],
    actual_stage_images=548,all605_collected_files_verified=True,training_epochs=12,native_gate_passed=done['native_gate_passed'],
    saved_EMA_independent_replay_verified=False,formal_terminal_reader_passed=False,
    verification_script_sha256=sha(__file__),new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0,
    scope=manifest['scope'])
write(OUT/'primary_verification_v1.json',result)
print(json.dumps(dict(status=result['status'],AP=100*endpoint['native_stats'][0],APs=100*endpoint['native_stats'][3],
    actual_stage_images=548,weights_sha256=result['last_checkpoint_sha256'],saved_EMA_replay_pending=True),ensure_ascii=False))

