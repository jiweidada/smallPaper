"""Independent NumPy native MAL reconstruction from immutable saved heads."""
from pathlib import Path
from collections import defaultdict
import csv, hashlib, json, numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/matched_quality_signal_20261010_v1'
INPUT=ROOT/'outputs/native_augmented_supervision_20261010_v1'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sigmoid(z):
    result=np.empty_like(z,dtype=np.float64);pos=z>=0
    result[pos]=1/(1+np.exp(-z[pos]));ez=np.exp(z[~pos]);result[~pos]=ez/(1+ez)
    return result
def xyxy(b):
    # Native FP32 primal, including representable tiny edges.
    return np.concatenate((b[:,:2]-b[:,2:]/np.float32(2),b[:,:2]+b[:,2:]/np.float32(2)),axis=1)
def iou_pairs(pred,gt):
    a,b=xyxy(pred),xyxy(gt)
    inter_wh=np.maximum(np.minimum(a[:,2:],b[:,2:])-np.maximum(a[:,:2],b[:,:2]),np.float32(0))
    inter=inter_wh[:,0]*inter_wh[:,1]
    awh=a[:,2:]-a[:,:2];bwh=b[:,2:]-b[:,:2]
    union=awh[:,0]*awh[:,1]+bwh[:,0]*bwh[:,1]-inter
    return inter/union
def pair(data,event,j):return data[event['prefix']+f'.batch{j}.query_index'],data[event['prefix']+f'.batch{j}.GT_row_index']
def summarize(rows):
    n=len(rows)
    def count(name):return sum(r[name] for r in rows)
    return {'matched_rows':n,'GT_score_down':count('GT_score_down'),'GT_score_up':count('GT_score_up'),'margin_down':count('margin_down'),'margin_up':count('margin_up'),'wrong_class_argmax':count('wrong_class_argmax'),'IoU_ge50':count('IoU_ge50'),'IoU_ge75':count('IoU_ge75'),'well_localized_weak_score':count('well_localized_weak_score'),'well_localized_weak_score_margin_down':sum(r['well_localized_weak_score'] and r['margin_down'] for r in rows),'mean_IoU':float(np.mean([r['IoU'] for r in rows])) if n else None,'median_p_GT':float(np.median([r['p_GT'] for r in rows])) if n else None,'fraction_GT_score_down':count('GT_score_down')/n if n else None,'fraction_margin_down':count('margin_down')/n if n else None,'fraction_wrong_class':count('wrong_class_argmax')/n if n else None}
def main():
    protocol=read(OUT/'protocol.json');assert sha(Path(__file__))==protocol['code_sha256']['analyze_saved.py']
    target=OUT/'numpy_analysis';target.mkdir(exist_ok=False)
    annotation=ROOT/'数据/03_本轮实验记录_20260928/转换COCO标注原件_20260929/VisDrone2019-DET-train/annotations/train.json'
    assert sha(annotation)=='32c1cae852e084430ea74503eb660f0707ec6b5e65d0998ce3ec8b448265c883'
    anns={a['id']:a for a in read(annotation)['annotations']}
    rows=[];calls=[];expected={};max_error=0.;narrays=0
    for seed in (1,2):
      assert sha(INPUT/f'collection_seed{seed}.tar.gz')==protocol['archive_sha256'][str(seed)]
      capture=INPUT/f'collection_seed{seed}/capture_v1';source=read(capture/'source_identity.json');cfg=source['config']['DEIMCriterion'];assert cfg['gamma']==1.5 and cfg.get('mal_alpha') is None and cfg.get('no_weight_vfl_epoch',-1)==-1 and cfg['weight_dict']['loss_mal']==1
      for endpoint in (12,132):
       for batch in range(16):
        path=capture/f'endpoint{endpoint}/batch_{batch:03d}';manifest=read(path/'manifest.json')
        for name,h in manifest['files'].items():assert sha(path/name)==h
        ev=read(path/'evidence.json');assert ev['status']=='CAPTURED' and not ev['errors']
        with np.load(path/'arrays.npz',allow_pickle=False) as npz:data={k:npz[k] for k in npz.files}
        assert all(np.isfinite(a).all() for a in data.values());narrays+=len(data)
        loss_events=[e for e in ev['events'] if e['kind']=='loss_call' and e['loss']=='mal' and not e['is_DN']];assert len(loss_events)==5
        for stage_index,event in enumerate(loss_events):
          stage=event['stage'];z=data[stage+'.pred_logits'].astype(np.float64);p=sigmoid(z);t=np.zeros_like(z);onehot=np.zeros_like(z);native_scale=event['num_boxes_normalizer'];q_all=[];j_all=[];g_all=[];c_all=[];ious_all=[];other_all=[]
          for j in range(4):
            q,g=pair(data,event,j);boxes=data[f'target.{j}.boxes'];labels=data[f'target.{j}.labels'];ids=data[f'target.{j}.verified_source_annotation_ids'];classes=labels[g]
            assert len(set(q))==len(q)==len(set(g));assert len(q)==min(300,len(boxes))
            iou=iou_pairs(data[stage+'.pred_boxes'][j,q],boxes[g]);assert np.isfinite(iou).all() and (iou>=0).all() and (iou<=1.000001).all()
            t[j,q,classes]=iou.astype(np.float64)**1.5;onehot[j,q,classes]=1
            gt_z=z[j,q,classes];other_z=z[j,q].copy();other_z[np.arange(len(q)),classes]=-np.inf;other=other_z.argmax(1)
            gt_p=p[j,q,classes];other_p=p[j,q,other];gt_descent=t[j,q,classes]-gt_p;margin_descent=gt_descent+other_p**2.5
            size=ev['metadata']['input_shape'][-1];short=boxes[g,2:].astype(np.float64).min(1)*size
            weights=data.get(f'target.{j}.mixup',np.ones(len(boxes)))
            for k,(qq,gg,cc,ii,ss) in enumerate(zip(q,g,classes,iou,short)):
              ann=anns[int(ids[gg])];assert ann['category_id']==int(cc)
              r=dict(seed=seed,endpoint=endpoint,batch=batch,item=j,stage=stage,query=int(qq),GT_row=int(gg),source_annotation_id=int(ids[gg]),source_image_id=ann['image_id'],category=int(cc),original_COCO_small=ann['area']<1024,GT_count_in_composite=len(boxes),dense_gt300=len(boxes)>300,input_size=size,input_short_pixels=float(ss),input_size_bin='lt4' if ss<4 else '4to8' if ss<8 else '8to16' if ss<16 else 'ge16',mixup=f'target.{j}.mixup' in data,mixup_weight=float(weights[gg]),IoU=float(ii),quality_bin='lt25' if ii<.25 else '25to50' if ii<.5 else '50to75' if ii<.75 else 'ge75',p_GT=float(gt_p[k]),p_max_other=float(other_p[k]),GT_logit=float(gt_z[k]),logit_margin=float(gt_z[k]-other_z[k,other[k]]),target_IoU_pow_gamma=float(t[j,qq,cc]),GT_score_descent_direction=float(gt_descent[k]),margin_descent_direction=float(margin_descent[k]),GT_score_down=bool(gt_descent[k]<-1e-7),GT_score_up=bool(gt_descent[k]>1e-7),margin_down=bool(margin_descent[k]<-1e-7),margin_up=bool(margin_descent[k]>1e-7),wrong_class_argmax=bool(other_z[k,other[k]]>gt_z[k]),IoU_ge50=bool(ii>=.5),IoU_ge75=bool(ii>=.75),well_localized_weak_score=bool(ii>=.75 and gt_p[k]<.05))
              rows.append(r)
            q_all.extend(q);j_all.extend([j]*len(q));g_all.extend(g);c_all.extend(classes);ious_all.extend(iou);other_all.extend(other)
          weight=p**1.5*(1-onehot)+onehot
          bce=np.maximum(z,0)-z*t+np.log1p(np.exp(-np.abs(z)))
          raw_loss=float((weight*bce).sum()/native_scale);native_loss=event['native_unweighted_loss_scalars']['loss_mal'];error=abs(raw_loss-native_loss);max_error=max(max_error,error);assert error<=protocol['verification']['raw_MAL_abs_tolerance'],(seed,endpoint,batch,stage,error)
          grad=weight*(p-t)/native_scale
          key=f'seed{seed}.endpoint{endpoint}.batch{batch:03d}.stage{stage_index}'
          q_all=np.asarray(q_all);j_all=np.asarray(j_all);c_all=np.asarray(c_all);other_all=np.asarray(other_all)
          expected[key+'.GT_grad']=grad[j_all,q_all,c_all];expected[key+'.other_grad']=grad[j_all,q_all,other_all];expected[key+'.IoU']=np.asarray(ious_all);expected[key+'.GT_row']=np.asarray(g_all);expected[key+'.query']=q_all;expected[key+'.max_other_class']=other_all
          calls.append(dict(key=key,stage=stage,native_raw_MAL=native_loss,numpy_raw_MAL=raw_loss,abs_error=error,num_boxes_normalizer=native_scale,matched_count=len(q_all)))
    with (target/'per_match_signals.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(target/'native_comparison_expected.npz',**expected)
    primary=[];strata=[]
    for seed in (1,2):
      for endpoint in (12,132):
        current=[r for r in rows if r['seed']==seed and r['endpoint']==endpoint]
        final_unsat=[r for r in current if r['stage']=='model' and not r['dense_gt300']]
        for size in ['lt4','4to8','8to16','ge16']:
          selected=[r for r in final_unsat if r['input_size_bin']==size];primary.append(dict(seed=seed,endpoint=endpoint,stage='model',GT_le300=True,input_size_bin=size,**summarize(selected)))
        for dim in ['stage','input_size_bin','quality_bin','dense_gt300','mixup','original_COCO_small','category']:
          for value in sorted({r[dim] for r in current},key=str):strata.append(dict(seed=seed,endpoint=endpoint,dimension=dim,value=value,**summarize([r for r in current if r[dim]==value])))
        for stage in sorted({r['stage'] for r in current}):
          tiny=[r for r in current if r['stage']==stage and not r['dense_gt300'] and r['input_size_bin']=='lt4']
          for quality in ['lt25','25to50','50to75','ge75']:strata.append(dict(seed=seed,endpoint=endpoint,dimension='unsaturated_tiny_stage_quality',value=stage+' / '+quality,**summarize([r for r in tiny if r['quality_bin']==quality])))
    summary=dict(status='PASS_NUMPY_NATIVE_RAW_MAL_RECOMPUTATION',normal_loss_calls=len(calls),matched_exposures_all_stages=len(rows),all_archived_arrays_finite=narrays,maximum_raw_MAL_abs_error=max_error,primary_unsaturated_final_by_size=primary,strata=strata,well_localized_weak_score_total=sum(r['well_localized_weak_score'] for r in rows),well_localized_weak_score_margin_down_total=sum(r['well_localized_weak_score'] and r['margin_down'] for r in rows),model_forwards_added=0,model_backwards_added=0,optimizer_steps=0,epochs_added=0,code_sha256=sha(Path(__file__)),protocol_sha256=sha(OUT/'protocol.json'),native_autograd_verified=False,AP_evidence=False,root_cause_established=False)
    save(target/'summary.json',summary);save(target/'loss_verification.json',{'status':'PASS_NUMPY_RAW_MAL','calls':calls,'tolerance':protocol['verification']['raw_MAL_abs_tolerance']})
    print(json.dumps({k:v for k,v in summary.items() if k!='strata'},ensure_ascii=False))
if __name__=='__main__':main()
