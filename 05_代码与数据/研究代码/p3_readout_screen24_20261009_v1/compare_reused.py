"""Reuse only the frozen completed baseline; never execute it or change its data."""
from pathlib import Path
import hashlib,json
from runtime import HERE,sha
from results import read,load_endpoint,pair
from evaluator_binding import same_result_evaluator

def verify_baseline(seed):
    binding=read(HERE.parent/'reused_baseline.json')
    if binding['seed']!=seed:raise RuntimeError('Wrong baseline seed')
    for path,digest in binding['files'].items():
        if sha(path)!=digest:raise RuntimeError('Reused baseline changed: '+path)
    base=Path(binding['run'])
    rows=[json.loads(l) for l in (base/'log.txt').read_text().splitlines() if l.strip()]
    if [r['epoch'] for r in rows]!=list(range(24)):raise RuntimeError('Baseline is not completed 24 epochs')
    endpoint=read(base/'endpoint_epoch24/training_completion.json')
    if endpoint['status']!='NATIVE_FIT_RETURNED_AND_LAST_CHECKPOINT_PRESENT':raise RuntimeError('Baseline incomplete')
    if sha(base/'last.pth')!=endpoint['last_checkpoint_sha256']:raise RuntimeError('Baseline checkpoint changed')
    return base,binding

def compare(candidate,seed):
    import numpy as np
    base,binding=verify_baseline(seed)
    left,lh,_=load_endpoint(base/'endpoint_epoch24')
    right,rh,_=load_endpoint(Path(candidate)/'endpoint_epoch24')
    same_result_evaluator(left,right)
    if left['protocol']!=right['protocol'] or left['environment']!=right['environment']:
        raise RuntimeError('Historical baseline runtime or evaluation protocol differs')
    for key in ('seed','common_init_sha256','training_binding_sha256','train_annotation_sha256','val_annotation_sha256','native_profiling'):
        if left['identity'][key]!=right['identity'][key]:raise RuntimeError('Baseline lineage differs: '+key)
    streams=[[json.loads(l) for l in (d/'first100_training_inputs.jsonl').read_text().splitlines()] for d in (base,Path(candidate))]
    aligned=len(streams[0])==len(streams[1])==100 and streams[0]==streams[1]
    metrics=pair(left,right,lh,rh)
    arrays=[]
    for d in (base,Path(candidate)):
        with np.load(d/'endpoint_epoch24/precision_recall.npz',allow_pickle=False) as v:
            p=v['precision'];v75=p[5,:,:,1,2];arrays.append(float(v75[v75>-1].mean()*100))
    metrics['APs75_percent_baseline_spatial']=arrays
    metrics['delta_APs75_pp']=arrays[1]-arrays[0]
    result={'seed':seed,'fixed_epoch':24,'first100_inputs_and_rng_equal':aligned,
       'historical_baseline':str(base),'differences_spatial_minus_baseline':metrics,
       'status':'OBSERVED_24_EPOCH_SCREEN_NOT_FULL132',
       'claim_boundary':'No CENTER arm: spatial information versus extra capacity is not isolated'}
    with (HERE.parent/f'paired_seed{seed}.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    if not aligned:raise RuntimeError('First100 inputs/RNG mismatch; results retained')
    return result
