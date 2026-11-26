"""Synthetic metric checks only. No model, cached predictions or real GT effects."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torchvision.ops import box_convert
from o27_metrics import (matlab_round,ignore_integral,region_keep,prepare_cases,
    postprocess_fixed,coco_artifacts,metrics_from_blocks,decision)

ROOT=Path(__file__).resolve().parent

def main():
    assert not (ROOT/'metric_preflight_result.json').exists()
    checks={}
    assert matlab_round([0.5,1.5,8.5,-0.5,-1.5]).tolist()==[1,2,9,-1,-2]
    checks['MATLAB_half_away_rounding']=True
    # The native inclusive bitmap covers columns1..5. Its integral formula
    # counts pixels to the right/below the queried corner, not that corner.
    rows=[[1,1,4,9,0,0,0,0],[1,1,2,9,0,0,0,0]]
    integral=ignore_integral(rows,12,12)
    assert integral[-1,-1]==50  # union, not overlapping-rectangle sum
    assert region_keep([[1,1,8,8],[1,1,9,8]],integral).tolist()==[False,True]
    checks['union_and_exact_half_excluded']=True
    rng=np.random.default_rng(427030)
    boxes=rng.uniform(-2,17,(400,4));boxes[:,2:]=rng.uniform(0.1,20,(400,2))
    observed=region_keep(boxes,integral)
    bitmap=np.zeros((12,12),dtype=int);bitmap[:10,:5]=1
    expected=[]
    for box in boxes:
        x,y,w,h=np.maximum(1,matlab_round(box));x=min(x,12);y=min(y,12)
        # Explicit pixel enumeration of the pinned integral-image expression.
        count=sum(int(bitmap[yy-1,xx-1]) for yy in range(y+1,min(12,y+h)+1)
                  for xx in range(x+1,min(12,x+w)+1))
        expected.append(count/(w*h)<.5)
    assert np.array_equal(observed,expected)
    checks['region_formula_400_enumerated_rectangles']=True
    fixture_raw=[];detections=[]
    for i in range(6):
        rows=[];db=[];dl=[];ds=[]
        for category in range(1,4):
            if i==0 and category==3:continue
            for j in range(1+(i+category)%3):
                box=[10+45*j,10+50*category,12+8*(i%2),15+24*(i%3)]
                rows.append(box+[1,category,0,0])
                if rng.random()<.78:
                    new=np.array(box,dtype=float);new[:2]+=rng.uniform(-3,3,2)
                    db.append(new.tolist());dl.append(category);ds.append(float(rng.uniform(.2,.99)))
                db.append([float(rng.uniform(0,190)),float(rng.uniform(0,190)),10,10])
                dl.append(category);ds.append(float(rng.uniform(.01,.95)))
        fixture_raw.append({'file':str(i),'prefix':str(i),'height':250,'width':250,'rows':rows})
        detections.append({'boxes':np.asarray(db),'labels':np.asarray(dl),'scores':np.asarray(ds)})
    cases=prepare_cases(fixture_raw)
    blocks,reference=coco_artifacts(cases,detections)
    point=metrics_from_blocks(blocks,np.ones(6,dtype=int))
    max_error=max(abs(point[k]-reference[k]) for k in ['AP','APsmall','AP50'])
    assert max_error<1e-12,(point,reference,max_error)
    assert 0<point['AP']<1 and 0<point['APsmall']<1
    checks['COCO_AP500_ones_max_error']=max_error
    weight_errors=[]
    for weights in [[2,0,1,3,0,1],[0,0,0,0,0,6],[1,1,1,1,1,1]]:
        expanded_cases=[];expanded_dt=[]
        for i,multiplicity in enumerate(weights):
            for _ in range(multiplicity):
                c=copy.deepcopy(cases[i]);c['id']=len(expanded_cases)+1
                expanded_cases.append(c);expanded_dt.append(detections[i])
        _,direct=coco_artifacts(expanded_cases,expanded_dt)
        weighted=metrics_from_blocks(blocks,weights)
        error=max(abs(weighted[k]-direct[k]) for k in ['AP','APsmall','AP50'])
        assert error<1e-12,(weights,weighted,direct,error)
        weight_errors.append(error)
    checks['integer_multiplicity_vs_explicit_copies_max_errors']=weight_errors
    empty=[{'boxes':np.empty((0,4)),'labels':np.empty(0,dtype=int),'scores':np.empty(0)} for _ in cases]
    empty_blocks,empty_ref=coco_artifacts(cases,empty)
    assert all(x==0 for x in empty_ref.values())
    assert all(x==0 for x in metrics_from_blocks(empty_blocks,np.ones(6,dtype=int)).values())
    checks['empty_detections_AP_zero']=True
    # A true positive below rank100 must survive the explicitly selected cap500.
    dense_case=prepare_cases([{'file':'dense','prefix':'dense','height':100,'width':100,
                               'rows':[[10,10,10,10,1,1,0,0]]}])
    dense={'boxes':np.vstack([np.tile([70,70,10,10],(150,1)),[10,10,10,10]]),
           'labels':np.ones(151,dtype=int),'scores':np.linspace(.99,.01,151)}
    dense_blocks,dense_ref=coco_artifacts(dense_case,[dense])
    assert dense_ref['AP']>0 and metrics_from_blocks(dense_blocks,[1])['AP']>0
    checks['rank151_true_positive_included_at_cap500']=True
    # Source-equivalent fixed-cap conversion; class0/11 filtering occurs afterTopK.
    case={'width':800,'height':600,'integral':None}
    b=rng.uniform(.1,.8,(1,250,4)).astype(np.float32)
    l=rng.normal(0,1,(1,250,12)).astype(np.float32)
    got=postprocess_fixed(b,l,case)
    score,index=torch.from_numpy(l).sigmoid().flatten(1).topk(500,dim=-1)
    labels=index%12;q=index//12;valid=((labels>=1)&(labels<=10))[0].numpy()
    xyxy=box_convert(torch.from_numpy(b),'cxcywh','xyxy')*torch.tensor([800,600,800,600])
    expected=box_convert(xyxy.gather(1,q[:,:,None].expand(-1,-1,4)),'xyxy','xywh')[0].numpy()[valid]
    assert got['counts']['before_label_and_region_filter']==500
    assert np.array_equal(got['boxes'],expected) and np.array_equal(got['query_ids'],q[0].numpy()[valid])
    assert np.array_equal(got['scores'],score[0].numpy()[valid])
    checks['fixed_cap_500_flat_class_scores_not_native_override']=True
    p={'harm_rate':.03,'APsmall_O_minus_N':.006,'AP_O_minus_N':.002,
       'APsmall_O_minus_T':.003,'APsmall_O_minus_B':.004}
    ci={k:[.001,.01] for k in p};ci['harm_rate']=[.01,.08]
    exposure={'base_correct_small':30,'prefixes_with_base_small_TP':8}
    assert decision(p,ci,exposure)['pilot_all_gates_pass']
    for key in p:
        altered=dict(p);altered[key]=0 if key!='AP_O_minus_N' else -.001
        assert not decision(altered,ci,exposure)['pilot_all_gates_pass']
    assert not decision(p,ci,{'base_correct_small':23,'prefixes_with_base_small_TP':8})['pilot_all_gates_pass']
    checks['zero_negative_and_low_exposure_not_promoted']=True
    result={'status':'SYNTHETIC_METRIC_PREFLIGHT_PASS','checks':checks,'whole_detector_calls':0,
            'decoder_calls':0,'optimizer_updates':0,'real_prediction_files_read':0,
            'metric_script_sha256':hashlib.sha256((ROOT/'o27_metrics.py').read_bytes()).hexdigest(),
            'self_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'versions':{'torch':torch.__version__,'numpy':np.__version__},
            'CUDA_initialized':torch.cuda.is_initialized()}
    (ROOT/'metric_preflight_result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
