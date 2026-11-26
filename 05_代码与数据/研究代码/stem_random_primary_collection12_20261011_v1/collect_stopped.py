"""Package already-stopped native random-Stem exact12 runs; never loads a model."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,os,sys,traceback

COLLECTOR_SHA='eb6dc614c29a80c2ef865bb6b94f307994a5a2a3beb55d40cfd4f021580255f5'
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def main(prefix,seed):
    assert seed in (1,2) and prefix=={1:'/25085400120',2:'/24085403014'}[seed]
    root=Path(prefix)/'DEIM/experiments/stem_primary_collection_20261011_v1'
    source=root/'code/collect.py';assert sha(source)==COLLECTOR_SHA
    output=root/f'seed{seed}_native_stem_scratch_v1'
    assert not output.exists(),'A prior package/attempt exists; inspect it instead of retrying.'
    start=root/f'scratch_seed{seed}_collect_v1_start.json'
    completion=root/f'scratch_seed{seed}_collect_v1_completion.json'
    failure=root/f'scratch_seed{seed}_collect_v1_failure.json'
    assert not any(p.exists() for p in (start,completion,failure))
    receipt=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
        seed=seed,arm='native_stem_scratch',wrapper_sha256=sha(__file__),collector_sha256=sha(source),
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),automatic_retry=False,
        new_model_forwards=0,new_training_epochs=0,new_optimizer_updates=0,new_evaluator_calls=0)
    write(start,receipt)
    try:
        spec=importlib.util.spec_from_file_location('native_primary_collector',source)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        metadata=m.collect(prefix,seed,'native_stem_scratch')
        assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
        result=dict(status='STOPPED_RANDOM_NATIVE_STEM_PRIMARY12_PACKAGE_COLLECTED',exit_code=0,metadata=metadata,
            receipt=receipt,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            new_model_forwards=0,new_training_epochs=0,new_optimizer_updates=0,new_evaluator_calls=0,
            saved_EMA_independent_replay_included=False)
        write(completion,result)
    except BaseException:
        write(failure,dict(status='READ_ONLY_PACKAGE_FAILURE_NO_RETRY',traceback=traceback.format_exc(),receipt=receipt))
        raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=(1,2),required=True)
    p.add_argument('--prefix',required=True);a=p.parse_args();main(a.prefix,a.seed)
