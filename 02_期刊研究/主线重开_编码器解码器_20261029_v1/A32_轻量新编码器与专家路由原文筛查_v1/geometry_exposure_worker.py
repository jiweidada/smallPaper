"""Frozen annotation/cache association audit. No model imports, forwards, or training."""
import ast
import collections
import hashlib
import json
import math
import pathlib
import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
D=pathlib.Path(__file__).resolve().parent

def read(p): return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(p,z):
    with pathlib.Path(p).open('x',encoding='utf-8') as f:
        json.dump(z,f,ensure_ascii=False,indent=2);f.write('\n')

def scalar_iou(a,b):
    x,y,w,h=a;xx,yy,ww,hh=b
    inter=max(0,min(x+w,xx+ww)-max(x,xx))*max(0,min(y+h,yy+hh)-max(y,yy))
    return inter/(w*h+ww*hh-inter) if inter else 0.

def run():
    start=time.monotonic();p=read(D/'geometry_exposure_plan_FROZEN.json')
    assert sha(__file__)==p['inputs']['worker']['sha256']
    for row in p['inputs'].values(): assert sha(row['path'])==row['sha256'],row['path']
    save(D/'geometry_one_shot.lock',dict(plan_sha256=sha(D/'geometry_exposure_plan_FROZEN.json'),pid=__import__('os').getpid()))
    gt=read(p['inputs']['GT']['path'])
    geometry={x['image_id']:x['geometry'] for x in read(p['inputs']['raw_annotations']['path'])['cases']}
    by_image=collections.defaultdict(list)
    for g in gt['annotations']: by_image[g['image_id']].append(g)
    images=sorted(x['id'] for x in gt['images']);cats=list(range(10))
    # Reuse only the already independently verified, authored matcher definitions.
    helper=pathlib.Path(p['inputs']['matcher']['path']).read_text(encoding='utf-8')
    module=ast.parse(helper)
    nodes=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in ['matrix_iou','match_image']]
    scope=dict(np=np,THRESHOLDS=np.arange(.5,.96,.05))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<audited-matcher-definitions>','exec'),scope)
    labels={};endpoint_checks={}
    for arm in p['arms']:
        pred=read(p['inputs'][arm]['path']);group=collections.defaultdict(list)
        for i,x in enumerate(pred):
            x['_id']=i+1;group[(x['image_id'],x['category_id'])].append(x)
        threshold=p['thresholds'][arm];matched=set();fp=0;tp=0
        for im in images:
            for c in cats:
                result=scope['match_image'](group[(im,c)],[g for g in by_image[im] if g['category_id']==c],[0,1e10])
                keep=result['scores']>=threshold
                ids=result['matches'][0,keep];matched.update(int(v) for v in ids if v)
                tp+=int(np.count_nonzero(ids));fp+=int(np.count_nonzero(ids==0))
        small={g['id'] for g in gt['annotations'] if g['_area_input']<1024}
        got=dict(FP=fp,TP=tp,small_TP=len(matched&small),small_FN=len(small-matched),threshold=threshold)
        assert got['FP']==926 and got['small_TP']==p['expected_small_TP'][arm],got
        labels[arm]=matched;endpoint_checks[arm]=got
    keys=['same16','same16_different8','same16_different8_different_class','same16_different8_IoU_lt_01']
    rows=[];independent_pair_checks=0
    for im in images:
        z=geometry[im];s=z['ratio'];left,top=z['pad_ltrb'][:2]
        small=[g for g in by_image[im] if g['_area_input']<1024]
        centers={g['id']:((g['bbox'][0]+g['bbox'][2]/2)*s+left,(g['bbox'][1]+g['bbox'][3]/2)*s+top) for g in small}
        for phase_x,phase_y in p['grid_phases']:
            bins=collections.defaultdict(list)
            for g in small:
                x,y=centers[g['id']];bins[(math.floor((x-phase_x)/16),math.floor((y-phase_y)/16))].append(g)
            for g in small:
                x,y=centers[g['id']];coarse=(math.floor((x-phase_x)/16),math.floor((y-phase_y)/16))
                fine=(math.floor((x-phase_x)/8),math.floor((y-phase_y)/8))
                neighbors=[h for h in bins[coarse] if h['id']!=g['id']]
                split=[]
                for h in neighbors:
                    xx,yy=centers[h['id']]
                    if (math.floor((xx-phase_x)/8),math.floor((yy-phase_y)/8))!=fine: split.append(h)
                values=[bool(neighbors),bool(split),any(h['category_id']!=g['category_id'] for h in split),any(scalar_iou(g['bbox'],h['bbox'])<.1 for h in split)]
                # Independent pairwise interval-membership verification, no bucket lookup.
                second=[False]*4
                for h in small:
                    if h['id']==g['id']:continue
                    xx,yy=centers[h['id']]
                    same=(phase_x+16*coarse[0]<=xx<phase_x+16*(coarse[0]+1) and phase_y+16*coarse[1]<=yy<phase_y+16*(coarse[1]+1))
                    split_flag=not(phase_x+8*fine[0]<=xx<phase_x+8*(fine[0]+1) and phase_y+8*fine[1]<=yy<phase_y+8*(fine[1]+1))
                    if same:
                        second[0]=True
                        if split_flag:
                            second[1]=True
                            if h['category_id']!=g['category_id']:second[2]=True
                            if scalar_iou(g['bbox'],h['bbox'])<.1:second[3]=True
                    independent_pair_checks+=1
                assert values==second,(im,g['id'],phase_x,phase_y,values,second)
                row=dict(image_id=im,GT_id=g['id'],category_id=g['category_id'],area_input=g['_area_input'],size_bin=int(g['_area_input']>=64)+int(g['_area_input']>=256),phase=[phase_x,phase_y],center_input=[x,y],**dict(zip(keys,values)))
                row['miss']={arm:g['id'] not in labels[arm] for arm in p['arms']};rows.append(row)
    assert len(rows)==4127*4
    summaries={}
    for phase in p['grid_phases']:
        rr=[r for r in rows if r['phase']==phase];sub={}
        for key in keys:
            exposed=[r for r in rr if r[key]];unexposed=[r for r in rr if not r[key]]
            item=dict(exposed_GT=len(exposed),exposed_images=len({r['image_id'] for r in exposed}),unexposed_GT=len(unexposed),exposure_fraction=len(exposed)/len(rr),arms={})
            for arm in p['arms']:
                a=sum(r['miss'][arm] for r in exposed);b=sum(r['miss'][arm] for r in unexposed)
                er=a/len(exposed) if exposed else None;ur=b/len(unexposed) if unexposed else None
                item['arms'][arm]=dict(exposed_misses=a,unexposed_misses=b,exposed_miss_rate=er,unexposed_miss_rate=ur,raw_risk_difference=None if er is None or ur is None else er-ur)
            sub[key]=item
        summaries[str(phase)]=sub
    # Standardized association in shared image/category/input-area strata, primary phase only.
    rr=[r for r in rows if r['phase']==[0,0]];strata=collections.defaultdict(list)
    for r in rr:strata[(r['image_id'],r['category_id'],r['size_bin'])].append(r)
    eligible=0;strata_count=0;contributions={a:np.zeros((100,2),dtype=float) for a in p['arms']}
    imindex={im:i for i,im in enumerate(images)}
    for (im,_,_),ss in strata.items():
        e=[r for r in ss if r['same16_different8']];u=[r for r in ss if not r['same16_different8']]
        if not e or not u:continue
        eligible+=len(ss);strata_count+=1;w=len(e)*len(u)/(len(e)+len(u))
        for arm in p['arms']:
            rd=sum(r['miss'][arm] for r in e)/len(e)-sum(r['miss'][arm] for r in u)/len(u)
            contributions[arm][imindex[im]]+=[w*rd,w]
    rng=np.random.default_rng(p['bootstrap_seed']);draws=rng.integers(0,100,size=(1000,100));adjusted={};boots={}
    for arm,values in contributions.items():
        total=values.sum(0);samples=values[draws].sum(1)
        boot=np.divide(samples[:,0],samples[:,1],out=np.full(1000,np.nan),where=samples[:,1]>0);boots[arm]=boot
        adjusted[arm]=dict(weighted_risk_difference=None if total[1]==0 else total[0]/total[1],paired_image_bootstrap95=np.nanquantile(boot,[.025,.975]).tolist(),shared_strata=strata_count,eligible_GT=eligible,scope='Descriptive size/category/image-stratified association,not causal or a deployment gain.')
    np.savez_compressed(D/'geometry_bootstrap.npz',draws=draws,**boots)
    save(D/'geometry_per_GT.json',dict(rows=rows))
    out=dict(status='COMPLETED_EXPLORATORY_GEOMETRY_ASSOCIATION',images=100,small_GT=4127,primary_phase=[0,0],endpoint_checks=endpoint_checks,exposure=summaries,primary_stratified=adjusted,independent_pair_checks=independent_pair_checks,independently_verified_GT_phase_rows=len(rows),new_model_calls=0,new_optimizer_updates=0,new_algorithm_validated=False,scope=p['interpretation_limits'],elapsed_seconds=time.monotonic()-start)
    save(D/'geometry_result.json',out)
    print(json.dumps(dict(status=out['status'],primary_exposure=summaries['[0, 0]']['same16_different8'],primary_stratified=adjusted,independent_checks=independent_pair_checks,elapsed_seconds=out['elapsed_seconds']),ensure_ascii=False))

if __name__=='__main__': run()
