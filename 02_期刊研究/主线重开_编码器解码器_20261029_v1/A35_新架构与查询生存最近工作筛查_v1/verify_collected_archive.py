from pathlib import Path,PurePosixPath
import hashlib,io,json,sys,zipfile
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=read(D/'native_prefix_protocol_FROZEN.json');complete=read(D/'remote_complete.json')
archive=D/'native_evidence.zip'
assert archive.stat().st_size==complete['archive_bytes'] and sha(archive)==complete['archive_sha256']
E=D/'downloaded_evidence';E.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:
    assert len(z.infolist())==complete['archived_files']
    for info in z.infolist():
        p=PurePosixPath(info.filename);assert not p.is_absolute() and '..' not in p.parts
        target=E.joinpath(*p.parts).resolve();assert E.resolve() in target.parents
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(z.read(info))
manifest=read(E/'file_manifest.json')['files_sha256']
for path,h in manifest.items():assert sha(E/path)==h,path
assert len(manifest)+1==complete['archived_files']
for name,h in read(D/'execution_inputs_manifest.json')['remote_files_sha256'].items():
    assert sha(E/name)==h and sha(D/name)==h,name
assert sha(E/'execution_inputs_manifest.json')==sha(D/'execution_inputs_manifest.json')
assert complete['counts']==dict(decoder_entries=37,decoder_completed=37,complete_three_layer_calls=13,
    partial_prefix_calls=24,decoder_layer_entries=75,query_initializers=12,
    whole_detector_forwards=0,image_encoder_forwards=0,optimizer_updates=0)
reference_arrays=prefix_arrays=rank_arrays=0
with zipfile.ZipFile(D/'native500_reference.zip') as refs:
    for c in cfg['cases']:
        with np.load(E/'captures'/('case%02d.npz'%c['index']),allow_pickle=False) as z:
            with np.load(io.BytesIO(refs.read(c['reference_entry'])),allow_pickle=False) as r:
                for k in ['boxes','logits']:
                    assert np.array_equal(z['depth3_'+k],r[k]);reference_arrays+=1
            for depth in [1,2]:
                assert np.array_equal(z[f'depth{depth}_features'],z[f'full_call_layer{depth}_features']);prefix_arrays+=1
            for depth in [1,2,3]:
                prob=z[f'depth{depth}_probabilities'][0,:,1:11].max(1)
                assert np.array_equal(z[f'depth{depth}_rank_valid'],np.argsort(-prob,kind='stable'));rank_arrays+=1
                for field in ['boxes','logits','features','probabilities']:assert np.isfinite(z[f'depth{depth}_{field}']).all()
with np.load(E/'captures/case00.npz',allow_pickle=False) as first,np.load(E/'repeat_first_full.npz',allow_pickle=False) as repeated:
    for k in repeated.files:assert np.array_equal(repeated[k],first['depth3_'+k])
events=[json.loads(line) for line in (E/'calls.jsonl').read_text(encoding='utf-8').splitlines()]
assert sum(x['kind']=='PREFIX_ENTER' for x in events)==37
assert sum(x['kind']=='PREFIX_COMPLETE' for x in events)==37
fidelity=read(E/'fidelity.json')
assert fidelity['weights_unchanged'] and fidelity['native_repeat_bitwise_equal'] and not fidelity['cuda_initialized']
out=dict(status='PASS_ALL_ARCHIVE_HASHES_AND_NATIVE_PREFIX_FIDELITY',archive_sha256=sha(archive),
         archive_files=complete['archived_files'],manifest_files_verified=len(manifest),
         reference_head_arrays_compared=reference_arrays,native_prefix_feature_arrays_compared=prefix_arrays,
         ranking_arrays_compared=rank_arrays,repeat_arrays_compared=len(repeated.files),counts=complete['counts'],
         scope='Archive and native execution fidelity; no pruning effect,novelty or new algorithm claim.')
with (D/'archive_verification.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out))
