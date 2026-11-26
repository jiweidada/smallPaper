"""No-GT frozen proxy decisions, then cached counterfactual evaluation. No policy fit."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,multiprocessing as mp,os,sys,time,traceback
import cv2,numpy as np,torch
from PIL import Image
ROOT=Path('/root/c10_region_restore_20261029_v1');OUT=ROOT/'local_proxy_v1';REG=ROOT/'restoration_probe_v1'
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(REG));sys.path.insert(0,str(ROOT/'simple_codec_gate_v1'))
from study import json_write,state_hash
from evaluate_fixed import prepare_pixels,detection_records,ground_truth_input
from run_probe import detector,weights,blend,sha
from run_gate import metrics,matches
START=time.monotonic()
METHODS=['regional_pseudograd','regional_confidence','global_pseudograd','global_confidence']
def event(stage,**kw):
    row=dict(stage=stage,pid=os.getpid(),elapsed_seconds=time.monotonic()-START,**kw)
    json_write(OUT/'live.json',row);print(json.dumps(row),flush=True)
def by_image(path):
    out=defaultdict(list)
    for row in json.loads(path.read_text()):out[row['image_id']].append(row)
    return out
def pseudo_batch(pred):
    valid=[p for p in pred if p['score']>=.25 and p['bbox'][2]>0 and p['bbox'][3]>0]
    boxes=[]
    for p in valid:
        x,y,w,h=p['bbox'];boxes.append([(x+w/2)/640,(y+h/2)/640,w/640,h/640])
    return dict(batch_idx=torch.zeros(len(valid),device='cuda'),cls=torch.tensor([p['category_id'] for p in valid],device='cuda').float().view(-1,1),
                bboxes=torch.tensor(boxes,device='cuda').float().reshape(-1,4)),len(valid)
def confidence_regions(pred,meta):
    sums=np.zeros(4);r=meta['geometry']['ratio'];l,t,_,_=meta['geometry']['pad_ltrb']
    for p in pred:
        if p['score']<.25:continue
        x,y,w,h=p['bbox'];cx=(x+w/2-l)/r;cy=(y+h/2-t)/r
        index=2*int(cy>=meta['height']/2)+int(cx>=meta['width']/2)
        sums[index]+=p['score']
    return sums
def binary_mask(values):return sum(1<<i for i,x in enumerate(values) if x>0)
def nms_records(decoded,meta):
    det=nms.non_max_suppression(decoded.detach(),conf_thres=.001,iou_thres=.7,multi_label=True,agnostic=False,max_det=500,nc=10,end2end=False)[0]
    return detection_records(det,meta)[0]
from ultralytics.utils import nms
def compare_records(current,expected):
    assert len(current)==len(expected)
    if not current:return 0.
    a=np.asarray([p['bbox']+[p['score'],p['category_id']] for p in current]);b=np.asarray([p['bbox']+[p['score'],p['category_id']] for p in expected])
    delta=float(np.max(np.abs(a-b)));assert delta<1e-4,delta
    return delta

def decide():
    # Decision phase deliberately never opens dataset/*gt*, regional losses, choices, or utility files.
    obs=json.loads((ROOT/'receipts/diagnostic_observations.json').read_text())['images']
    cache={m['image_id']:m for m in json.loads((REG/'restoration_cache.json').read_text())['images']}
    decisions={};forwards=0;backwards=0;checks=[]
    for arm in ['D0','DA']:
        model,criterion=detector(arm);before=state_hash(model)
        jpeg_preds=by_image(REG/'predictions'/arm/'subset_00.json');restored_preds=by_image(REG/'predictions'/arm/'subset_15.json')
        rows=[]
        for i,m in enumerate(obs):
            rid=m['image_id'];assert sha(m['q20'])==m['jpeg_sha256'] and sha(cache[rid]['file'])==cache[rid]['sha256']
            with Image.open(m['q20']) as im:a=np.asarray(im.convert('RGB'))
            with Image.open(cache[rid]['file']) as im:b=np.asarray(im.convert('RGB'))
            ws=weights(*a.shape[:2]);base=prepare_pixels(a,m['geometry']).detach().requires_grad_(True)
            batch,count=pseudo_batch(restored_preds[rid])
            with torch.enable_grad():
                decoded,raw=model(base);forwards+=1
                losses,_=criterion.loss(raw['one2many'] if 'one2many' in raw else raw,batch)
                assert torch.isfinite(losses).all()
                gradient=torch.autograd.grad(losses.sum(),base)[0].detach();backwards+=1
            assert torch.isfinite(gradient).all() and all(p.grad is None for p in model.parameters())
            with torch.no_grad():
                # A gradient forward must reproduce the previously frozen ordinary forward.
                delta=compare_records(nms_records(decoded,m),jpeg_preds[rid])
                scores=[]
                for k in range(4):
                    single=prepare_pixels(blend(a,b,ws,1<<k),m['geometry'])
                    scores.append(float(-(gradient*(single-base.detach())).sum()))
                full=prepare_pixels(b,m['geometry']);global_score=float(-(gradient*(full-base.detach())).sum())
                response=confidence_regions(restored_preds[rid],m)-confidence_regions(jpeg_preds[rid],m)
            choices=dict(regional_pseudograd=binary_mask(scores),regional_confidence=binary_mask(response),
                         global_pseudograd=15 if global_score>0 else 0,global_confidence=15 if response.sum()>0 else 0)
            if i<2:
                # Only chosen outputs are fetched here, after the two input-only choices are computed.
                for method in ['regional_pseudograd','regional_confidence']:
                    chosen=choices[method];expected=by_image(REG/'predictions'/arm/('subset_%02d.json'%chosen))[rid]
                    with torch.no_grad():fresh=model(prepare_pixels(blend(a,b,ws,chosen),m['geometry']))[0];forwards+=1
                    discrepancy=compare_records(nms_records(fresh,m),expected)
                    checks.append(dict(arm=arm,image_id=rid,method=method,subset=chosen,max_abs_delta=discrepancy))
            rows.append(dict(image_id=rid,file_name=m['file_name'],choices=choices,regional_pseudograd_scores=scores,
                             global_pseudograd_score=global_score,confidence_region_deltas=response.tolist(),pseudo_label_count=count,
                             pseudo_loss=float(losses.sum().detach()),baseline_max_abs_delta=delta))
            del base,decoded,raw,gradient,losses,full,single
            if (i+1)%20==0:event('NO_GT_DECISIONS',arm=arm,images=i+1,detector_forwards=forwards,input_backward_calls=backwards)
        assert state_hash(model)==before;decisions[arm]=rows
        del model,criterion;torch.cuda.empty_cache()
    assert forwards==208 and backwards==200
    json_write(OUT/'decisions.json',decisions)
    freeze=dict(decisions_sha256=sha(OUT/'decisions.json'),GT_utility_or_choice_tables_read=False,
                detector_forwards=forwards,input_backward_calls=backwards,training_updates=0,restorer_forwards=0,checks=checks)
    json_write(OUT/'decisions_frozen.json',freeze);return obs,decisions,freeze

def score_one(job):
    arm,method=job;gt=json.loads((ROOT/'dataset/diagnostic_gt_orig.json').read_text());obs=json.loads((ROOT/'receipts/diagnostic_observations.json').read_text())['images']
    inp_gt=ground_truth_input(gt,obs);pred=json.loads((OUT/'predictions'/arm/(method+'.json')).read_text());ev,mi=metrics(inp_gt,pred,500)
    original=[];mapping={m['image_id']:m for m in obs}
    for p in pred:
        m=mapping[p['image_id']];r=m['geometry']['ratio'];l,t,_,_=m['geometry']['pad_ltrb'];x,y,w,h=p['bbox']
        original.append(dict(image_id=p['image_id'],category_id=p['category_id'],score=p['score'],bbox=[(x-l)/r,(y-t)/r,w/r,h/r]))
    _,mo=metrics(gt,original,500);assert abs(mi['AP']-mo['AP'])<1e-10
    result=dict(AP=mo['AP'],AP50=mo['AP50'],AP_small_original=mo['AP_small'],AP_small_input=mi['AP_small'],
                matches=[matches(ev,inp_gt,c,k) for c in [.05,.1,.25,.5] for k in [.5,.75]])
    json_write(OUT/'metrics'/arm/(method+'.json'),result);return arm,method,result
def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/'one_shot.lock').open('x') as f:f.write(str(os.getpid()))
    torch.set_num_threads(4);cv2.setNumThreads(0);torch.cuda.reset_peak_memory_stats()
    json_write(OUT/'frozen.json',dict(protocol_sha256=sha(OUT/'protocol.json'),script_sha256=sha(__file__),
        reused_regional_script_sha256=sha(REG/'run_probe.py'),training_updates=0,pid=os.getpid()))
    obs,decisions,decision_freeze=decide();event('DECISIONS_FROZEN_NO_GT',decisions_sha256=decision_freeze['decisions_sha256'])
    # Ground-truth and utility records are accessible only after all decisions have been frozen.
    result=defaultdict(dict);diagnostics={}
    for arm in ['D0','DA']:
        allpred={k:by_image(REG/'predictions'/arm/('subset_%02d.json'%k)) for k in range(16)}
        for method in METHODS:
            pred=[]
            for row in decisions[arm]:pred.extend(allpred[row['choices'][method]][row['image_id']])
            json_write(OUT/'predictions'/arm/(method+'.json'),pred)
        truth={r['image_id']:r for r in json.loads((REG/(arm+'_loss_diagnostics.json')).read_text())}
        labels=np.asarray([truth[r['image_id']]['dynamic']['singles'] for r in decisions[arm]])
        pred=np.asarray([r['regional_pseudograd_scores'] for r in decisions[arm]])
        diagnostics[arm]=dict(region_rows=400,source_images=100,regional_pseudograd_sign_agreement=float(np.mean((pred>0)==(labels>0))),
            pearson=float(np.corrcoef(pred.ravel(),labels.ravel())[0,1]),
            choices_equal_GT_independent={m:sum(r['choices'][m]==truth[r['image_id']]['dynamic']['choices']['independent'] for r in decisions[arm]) for m in METHODS},
            selection_counts={m:{str(k):sum(r['choices'][m]==k for r in decisions[arm]) for k in range(16)} for m in METHODS})
    jobs=[(a,m) for a in ['D0','DA'] for m in METHODS]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        for arm,method,scores in pool.map(score_one,jobs):result[arm][method]=scores;event('SCORING',arm=arm,method=method)
    assert sha(OUT/'decisions.json')==decision_freeze['decisions_sha256']
    prior=json.loads((REG/'completed.json').read_text());interpret={}
    for arm in ['D0','DA']:
        best=max(METHODS,key=lambda m:result[arm][m]['AP_small_input']);oracle=prior['metrics'][arm]['independent']['AP_small_input']
        interpret[arm]=dict(best_observed_simple_proxy=best,best_proxy_small_AP_pct=100*result[arm][best]['AP_small_input'],
            gap_from_GT_independent_pp=100*(oracle-result[arm][best]['AP_small_input']),
            jpeg_gain_pp=100*(result[arm][best]['AP_small_input']-prior['metrics'][arm]['subset_00']['AP_small_input']),
            new_algorithm_necessity_weakens_point_screen=100*(oracle-result[arm][best]['AP_small_input'])<=.25)
    json_write(OUT/'completed.json',dict(status='COMPLETED',metrics=result,diagnostics=diagnostics,interpretation=interpret,
        elapsed_seconds=time.monotonic()-START,decision_freeze=decision_freeze,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cached_prediction_evaluation=True,not_deployment_latency_benchmark=True,training_updates=0,restorer_forwards=0))
    event('COMPLETED',interpretation=interpret)
if __name__=='__main__':
    try:main()
    except Exception as e:json_write(OUT/'failed.json',dict(error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-START));raise
