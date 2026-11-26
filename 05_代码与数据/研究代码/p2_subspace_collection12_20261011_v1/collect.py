"""Archive completed runtime diagnostics once; no model or evaluator imports."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,zipfile
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,o):
    with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,indent=2,allow_nan=False)
def live(r):
    p=Path('/proc/'+str(r['pid'])+'/stat')
    return p.exists() and p.read_text().split()[21]==str(r['start_ticks']) and p.read_text().split()[2]!='Z'
def main(prefix,seed):
    assert prefix=={1:'/25085400120',2:'/24085403014'}[seed]
    src=Path(prefix)/'DEIM/experiments/p2_subspace_diagnostic12_20261011_v1';spec=read(src/'code/protocol.json')
    assert {n:sha(src/'code'/n) for n in spec['remote_sources']}==spec['remote_sources']
    report=read(src/f'seed{seed}_GPU/report.json');completion=read(src/f'seed{seed}_GPU/completion.json');wait=read(src/f'seed{seed}_wait_completion.json')
    assert report['status']=='TWO_FIXED12_P2_RUNTIME_SUBSPACE_INTERVENTIONS_NATIVE_EVALUATED' and completion['exit_code']==wait['exit_code']==0
    assert not live(completion['receipt']) and not live(wait['receipt'])
    assert completion['report_sha256']==sha(src/f'seed{seed}_GPU/report.json') and wait['completion_sha256']==sha(src/f'seed{seed}_GPU/completion.json')
    assert report['candidate_model_forwards']==274 and report['new_native_evaluator_runs']==2
    assert report['new_BASE_forwards']==report['new_training_epochs']==report['new_optimizer_updates']==0
    assert report['checkpoint_unchanged'] and report['original28_training_sources_unchanged'] and not report['training_efficacy_proven']
    assert [r['mode'] for r in report['results']]==spec['interventions']
    user=Path(prefix)/f'DEIM/experiments/user_backbone12_20261011_v1/seed{seed}_p2_skip'
    assert sha(user/'run/last.pth')==report['parent_checkpoint_sha256'] and sha(user/'candidate_reload_v1/verification.json')==report['parent_replay_verification_sha256']
    pairs={}
    for sub in ('code',f'seed{seed}_CPU',f'seed{seed}_GPU'):
        for p in (src/sub).rglob('*'):
            if p.is_file():pairs[p.relative_to(src).as_posix()]=p
    for p in src.glob(f'seed{seed}_*.json'):pairs[p.name]=p
    pairs['parent_saved_EMA_verification.json']=user/'candidate_reload_v1/verification.json'
    pairs['parent_training_completion.json']=user/'run/training_completion.json'
    pairs['collection_source.py']=Path(__file__)
    root=Path(__file__).resolve().parent.parent;out=root/f'seed{seed}';out.mkdir(exist_ok=False)
    files={n:dict(bytes=p.stat().st_size,sha256=sha(p)) for n,p in sorted(pairs.items())}
    manifest=dict(schema='P2_runtime_subspace_complete_collection_v1',seed=seed,files=files,parent_checkpoint_sha256=report['parent_checkpoint_sha256'],diagnostic_source_sha256={p.name:sha(p) for p in (src/'code').iterdir() if p.is_file()},collector_sha256=sha(__file__),new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0)
    write(out/'manifest.json',manifest);archive=out/'runtime_artifacts_v1.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for n,p in sorted(pairs.items()):z.write(p,n)
        z.write(out/'manifest.json','manifest.json')
    metadata=dict(status='COMPLETED_P2_RUNTIME_SUBSPACE_ARTIFACTS_COLLECTED_NO_NEW_FORWARD',seed=seed,archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),manifest_sha256=sha(out/'manifest.json'),archive_entries=len(pairs)+1,collector_sha256=sha(__file__),parent_checkpoint_sha256=report['parent_checkpoint_sha256'],new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(out/'metadata.json',metadata);print(json.dumps(metadata),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,choices=(1,2),required=True);a=p.parse_args();main(a.prefix,a.seed)
