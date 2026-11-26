"""Verify and unpack the actual primary12 artifacts; no replay-gate substitution."""
from pathlib import Path,PurePosixPath
import base64,hashlib,importlib.util,json,stat,sys,zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stem_primary_collection_20261011_v1/seed1_original_stem_v1'
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
p={'base64': 'eyJzdGF0dXMiOiAiQ09MTEVDVEVEX1NUT1BQRURfRVhBQ1QxMl9QUklNQVJZX0FSVElGQUNUUyIsICJzZWVkIjogMSwgImFybSI6ICJvcmlnaW5hbF9zdGVtIiwgImFyY2hpdmUiOiAiLzI1MDg1NDAwMTIwL0RFSU0vZXhwZXJpbWVudHMvc3RlbV9wcmltYXJ5X2NvbGxlY3Rpb25fMjAyNjEwMTFfdjEvc2VlZDFfb3JpZ2luYWxfc3RlbV92MS9wcmltYXJ5X2FydGlmYWN0c192MS56aXAiLCAiYnl0ZXMiOiAxODg1MTM5OTMsICJzaGEyNTYiOiAiMTBlZWJkOTQyYjZkMmM0Y2VjNzQ2NGJhYTFkMzk2MWJhMGJiZjUzMDdkYjgzMzU5ZjM5MWZmMjVlODc1NGNmZCIsICJtYW5pZmVzdF9zaGEyNTYiOiAiZWVlYjg2MGRhNDA0ZTAzMTQ0MDNmMmUzYjFiNGEzYmMyNGI1MTY3YTIzMDA4ZmNkOTc0NjVlMDc1NmE2MTg5ZiIsICJhcmNoaXZlX2VudHJpZXMiOiA2MDYsICJsYXN0X2NoZWNrcG9pbnRfc2hhMjU2IjogIjM2ZDMwMTJlZWU4ODg4YTRiYzAzMDI0Y2MyMzE3MmY3ODcwODllYzU0MDJkMmY4MDQxMjk1NDNlNzIyZjk0MTciLCAibmF0aXZlX3N0YXRzIjogWzAuMDYxMjg0OTYyNTU1MTkwNDc1LCAwLjExNzA1OTY1MjY5OTI3MzA3LCAwLjA1NzQwNzY3ODc3MTY4MTgxLCAwLjA0MDIyNDMzNjM5OTMyMjA1LCAwLjA4OTAxNjM5NTU3OTc0MiwgMC4xMzA1Njk2NjExODMyMjMwMiwgMC4wNDAxMTYyMTkwMzI1MTYxODYsIDAuMTIwMzA3MzY2MDk4NjE0NTksIDAuMTc4NDUwMzQ4NjQ0OTc3MjcsIDAuMTI0ODMwNzMxMDA0MTU4NjksIDAuMjY1Njg2MDc0MDg5NjkwMjUsIDAuMjI0ODgzODkwMzMxNzkxNzRdLCAibmF0aXZlX2dhdGVfcGFzc2VkIjogZmFsc2UsICJzYXZlZF9FTUFfaW5kZXBlbmRlbnRfcmVwbGF5X2luY2x1ZGVkIjogZmFsc2UsICJjb2xsZWN0ZWRfcmVtb3RlX1VUQyI6ICIyMDI2LTEwLTExVDEyOjQ1OjM2LjQ4ODgwMSswMDowMCIsICJjb2xsZWN0b3Jfc2hhMjU2IjogImViNmRjNjE0YzI5YTgwYzJlZjg2NWJiNmI5NGYzMDc5OTRhNWEyYTNiZWI1NWQ0MGNmZDRmMDIxNTgwMjU1ZjUiLCAibmV3X21vZGVsX2ZvcndhcmRzIjogMCwgIm5ld19ldmFsdWF0b3JfY2FsbHMiOiAwLCAibmV3X3RyYWluaW5nX2Vwb2NocyI6IDB9', 'sha256': '084853a43ef8ca0e2d69529b8f95717426ef357fb01577df31195bbe840e715b', 'bytes': 1095};b=base64.b64decode(p['base64'],validate=True)
assert len(b)==p['bytes'] and hashlib.sha256(b).hexdigest()==p['sha256']
with (OUT/'remote_collection_metadata_v1.json').open('xb') as f:f.write(b)
meta=json.loads(b);assert meta['status']=='COLLECTED_STOPPED_EXACT12_PRIMARY_ARTIFACTS' and meta['seed']==1 and meta['arm']=='original_stem'
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
    identity=manifest['identity'];assert identity['arm']=='original_stem' and identity['seed']==1
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

