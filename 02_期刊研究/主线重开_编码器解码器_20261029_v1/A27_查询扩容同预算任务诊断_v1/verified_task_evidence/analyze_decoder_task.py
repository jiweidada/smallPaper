"""Prespecified exploratory task analysis. Constructs no detector or encoder."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from o27_metrics import (ARMS,prepare_cases,postprocess_fixed,coverage_counts,
                         coco_artifacts,metrics_from_blocks,decision)

ROOT=Path(__file__).resolve().parent
START=time.monotonic()

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(name,value):
    p=ROOT/name;t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(t,p)
def differences(metrics,harm):
    o=metrics['ONEWAY500'];n=metrics['NATIVE500'];t=metrics['TWOWAY500'];b=metrics['BASE250']
    return {'harm_rate':float(harm),'APsmall_O_minus_N':o['APsmall']-n['APsmall'],
      'AP_O_minus_N':o['AP']-n['AP'],'APsmall_O_minus_T':o['APsmall']-t['APsmall'],
      'APsmall_O_minus_B':o['APsmall']-b['APsmall']}

def main():
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    p=json.loads((ROOT/'task_protocol_FROZEN.json').read_text(encoding='utf-8'))
    assert p['status']=='FROZEN_EXPLORATORY_SAME_BUDGET_TASK'
    assert p['arms']==ARMS
    for row in p['files']:assert sha(ROOT/row['path'])==row['sha256']
    for row in p['dependencies']:assert sha(row['path'])==row['sha256']
    assert sha(ROOT/'raw_annotations_verified.json')==p['annotations_sha256']
    fd=os.open(str(ROOT/'analysis_one_shot.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.write(fd,json.dumps({'pid':os.getpid(),'utc':utc()}).encode());os.close(fd)
    atomic('analysis_started.json',{'pid':os.getpid(),'start_ticks':int(Path('/proc/self/stat').read_text().split()[21]),
      'utc':utc(),'protocol_sha256':sha(ROOT/'task_protocol_FROZEN.json')})
    decoded=json.loads((ROOT/'decoder_result.json').read_text(encoding='utf-8'))
    assert decoded['status']=='EXPLORATORY_DECODER_TASK_OUTPUTS_COMPLETE'
    assert decoded['protocol_sha256']==sha(ROOT/'task_protocol_FROZEN.json')
    assert decoded['decoder_completed']==45 and decoded['layer_entries']==135
    for row in decoded['outputs']:assert sha(ROOT/row['path'])==row['sha256']
    cases=prepare_cases(json.loads((ROOT/'raw_annotations_verified.json').read_text(encoding='utf-8'))['cases'])
    assert len(cases)==12 and len({c['prefix'] for c in cases})==12
    assert [c['file'] for c in cases]==[c['file'] for c in p['cases']]
    filtered={arm:[] for arm in ARMS};records={arm:[] for arm in ARMS};coverage=[]
    for i,case in enumerate(cases):
        raw_coverage={}
        for arm in ARMS:
            with np.load(ROOT/'predictions'/('case%02d_%s.npz'%(i,arm)),allow_pickle=False) as z:
                boxes=z['boxes'];logits=z['logits']
                dt=postprocess_fixed(boxes,logits,case,cap=500)
                if arm in ['BASE250','NATIVE500','ONEWAY500']:
                    raw_coverage[arm]=postprocess_fixed(boxes,logits,case,argmax_only=True)
            filtered[arm].append(dt)
            records[arm].append({'case_index':i,'file':case['file'],**{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in dt.items()}})
        coverage.append({'case_index':i,'file':case['file'],'prefix':case['prefix'],
          **coverage_counts(case,raw_coverage['BASE250'],raw_coverage['NATIVE500'],raw_coverage['ONEWAY500'])})
    atomic('postprocessed_detections.json',{'policy':p['metrics']['postprocess'],'arms':records})
    atomic('coverage_counts_observed.json',{'policy':p['metrics']['harm'],'cases':coverage})
    blocks={};reference={};point_metrics={};block_arrays={};block_meta=[]
    for arm in ARMS:
        blocks[arm],reference[arm]=coco_artifacts(cases,filtered[arm])
        point_metrics[arm]=metrics_from_blocks(blocks[arm],np.ones(12,dtype=int))
        for key in ['AP','APsmall','AP50']:assert abs(point_metrics[arm][key]-reference[arm][key])<1e-12
        for j,block in enumerate(blocks[arm]):
            prefix=arm+'_'+str(j)
            block_meta.append({'prefix':prefix,'arm':arm,'area_index':block['area_index'],'cat':block['cat']})
            for name in ['tp','fp','image_indices','ngt','presence']:block_arrays[prefix+'_'+name]=block[name]
    np.savez_compressed(ROOT/'coco_matching_blocks.npz',**block_arrays)
    atomic('coco_matching_blocks_metadata.json',{'rows':block_meta,'COCO_reference':reference,
           'fixed_score_tie_order':True,'conditional_bootstrap_contract':p['statistics']})
    denom=np.asarray([c['base_correct_small'] for c in coverage],dtype=np.int64)
    lost=np.asarray([c['native_uncompensated_small_losses'] for c in coverage],dtype=np.int64)
    assert np.all((lost>=0)&(lost<=denom))
    totals={key:sum(c[key] for c in coverage) for key in ['base_correct_small','native_uncompensated_small_losses','oneway_uncompensated_small_losses']}
    totals['prefixes_with_base_small_TP']=int((denom>0).sum())
    point=differences(point_metrics,float(lost.sum()/denom.sum()) if denom.sum() else 0.0)
    atomic('task_point_observed.json',{'metrics':point_metrics,'point':point,'coverage':totals,
          'GT':{'all':sum(len(c['gt_labels']) for c in cases),'small_original_area_le1024':sum(int(c['gt_small'].sum()) for c in cases)}})
    rng=np.random.default_rng(p['statistics']['seed']);n=p['statistics']['paired_bootstrap_replicates']
    weights=rng.multinomial(12,np.ones(12)/12,size=n)
    np.savez_compressed(ROOT/'bootstrap_weights.npz',weights=weights)
    samples={key:np.empty(n,dtype=np.float64) for key in point}
    zero_denominators=0
    for i,w in enumerate(weights):
        if i%100==0:
            assert time.monotonic()-START<p['statistics']['analysis_wall_seconds']
            atomic('task_progress.json',{'status':'PAIRED_BOOTSTRAP_RUNNING','completed':i,'planned':n,'seconds':time.monotonic()-START})
        metric={arm:metrics_from_blocks(blocks[arm],w) for arm in ARMS}
        d=int(denom@w)
        if not d:zero_denominators+=1
        value=differences(metric,float((lost@w)/d) if d else 0.0)
        for key in samples:samples[key][i]=value[key]
    intervals={key:np.quantile(value,[.025,.975],method='linear').tolist() for key,value in samples.items()}
    np.savez_compressed(ROOT/'paired_bootstrap_samples.npz',**samples)
    judged=decision(point,intervals,totals)
    result={'utc':utc(),'status':'EXPLORATORY_TASK_ANALYSIS_COMPLETE','protocol_sha256':sha(ROOT/'task_protocol_FROZEN.json'),
      'metrics':point_metrics,'point':point,'paired_95pct_percentile_intervals':intervals,'coverage':totals,'decision':judged,
      'bootstrap_replicates':n,'bootstrap_zero_exposure_denominators':zero_denominators,
      'zero_exposure_draw_policy':'harm rate0; retain draw, no selective dropping',
      'new_complete_three_layer_decoder_calls':45,'A26_decoder_outputs_reused':3,'contexts':12,
      'whole_detector_calls':0,'optimizer_updates':0,'model_constructed_by_analysis':False,
      'CUDA_initialized':torch.cuda.is_initialized(),'seconds':time.monotonic()-START,
      'GT':{'all':sum(len(c['gt_labels']) for c in cases),'small_original_area_le1024':sum(int(c['gt_small'].sum()) for c in cases)},
      'interpretation':'Reused12-image exploratory cohort, one frozen checkpoint, fixedTopK initializer bypassing PAQI. Not independent confirmation, complete detector reproduction, benchmark score or novelty proof.',
      'next':'No automatic training, resume, new mask threshold, seed, candidate or followup experiment.'}
    assert not torch.cuda.is_initialized()
    atomic('task_result.json',result);atomic('task_progress.json',{'status':result['status'],'completed':n,'planned':n})
    print(json.dumps({key:result[key] for key in ['status','point','paired_95pct_percentile_intervals','coverage','decision','seconds']}))

if __name__=='__main__':
    try:main()
    except BaseException:
        atomic('task_failure.json',{'status':'STOPPED_NO_RETRY','seconds':time.monotonic()-START,'traceback':traceback.format_exc()})
        raise
