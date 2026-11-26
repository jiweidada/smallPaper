"""Descriptive frozen-cache raw support and native endpoint assignment audit."""
from pathlib import Path
import collections,json,sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parent;D=R/'downloaded_evidence'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def overlaps(gt,boxes):
    lo=np.maximum(gt[:2],boxes[:,:2]);hi=np.minimum(gt[2:],boxes[:,2:]);wh=np.maximum(hi-lo,0);inter=wh[:,0]*wh[:,1];ga=np.prod(gt[2:]-gt[:2]);da=np.prod(boxes[:,2:]-boxes[:,:2],axis=1)
    return np.divide(inter,ga+da-inter,out=np.zeros_like(inter),where=ga+da-inter>0)
def main():
    assert not (R/'raw_support_result.json').exists()
    assert read(R/'independent_metric_verification.json')['status'].startswith('PASS_')
    cfg=read(R/'native_dual_head_protocol_FROZEN.json');result=read(D/'result.json');meta=read(D/'capture_metadata.json')
    canonical={x['stable_uid']:x for x in read(Path(cfg['local_raw_root'])/'dataset/diagnostic_gt_orig.json')['annotations'] if not x['iscrowd']}
    matched={k:set(next(x for x in result['metrics'][name]['thresholds'] if x['confidence']==.25)['matched_small_uids']) for k,name in [('M','Mbox_Mscore_NMS'),('O','Obox_Oscore_TOPK')]}
    rows=[];geometric_violations=[]
    for case in meta:
        f=np.load(D/'captures'/('case_%03d.npz'%case['index']),allow_pickle=False);z=cfg['cases'][case['index']];geo=z['geometry'];s=geo['ratio'];l,t=geo['pad_ltrb'][:2];bounds=[l,t,l+s*z['width'],t+s*z['height']]
        points=f['anchors_px'].astype(np.float64);strides=f['strides'].reshape(-1)
        branches={}
        for k in ['M','O']:
            box=f[k+'_decoded_xyxy'].astype(np.float64).copy();box[:,[0,2]]=np.clip(box[:,[0,2]],bounds[0],bounds[2]);box[:,[1,3]]=np.clip(box[:,[1,3]],bounds[1],bounds[3]);branches[k]={'box':box,'p':1/(1+np.exp(-np.clip(f[k+'_scores'].astype(np.float64),-80,80))),'fg':f[k+'_fg'].astype(bool),'gt':f[k+'_target_gt_idx'],'target':f[k+'_target_scores']}
        for gi,uid in enumerate(case['GT_uids']):
            gt=f['GT_xyxy'][gi].astype(np.float64);original=canonical[uid];area=original['bbox'][2]*original['bbox'][3]*s*s;cls=int(f['GT_classes'][gi]);small=area<1024
            center=(gt[:2]+gt[2:])/2;half=np.maximum(gt[2:]-gt[:2],16)/2;lower=center-half;upper=center+half
            inside=(points[:,0]>gt[0])&(points[:,0]<gt[2])&(points[:,1]>gt[1])&(points[:,1]<gt[3])
            outside_expanded=(points[:,0]<lower[0]-1e-4)|(points[:,0]>upper[0]+1e-4)|(points[:,1]<lower[1]-1e-4)|(points[:,1]>upper[1]+1e-4)
            record={'uid':uid,'image_id':case['image_id'],'category_id':cls,'input_area_canonical':area,'small':small,'very_small':area<256}
            for k,b in branches.items():
                assigned=b['fg']&(b['gt']==gi);effective=assigned&(b['target'].sum(axis=1)>0);overlap=overlaps(gt,b['box']);support=overlap>=.5;idx=np.flatnonzero(support);best=int(idx[np.argmax(b['p'][cls,idx])]) if len(idx) else None
                bestscore=float(b['p'][cls,best]) if best is not None else None
                own=float(b['target'][best].sum()) if best is not None and effective[best] else 0.
                record[k]={'matched_small_at025':uid in matched[k] if small else None,'raw_cover_exists':bool(len(idx)),'best_cover_score':bestscore,'best_cover_anchor':best,'best_cover_stride':float(strides[best]) if best is not None else None,'best_cover_inside_original_GT':bool(inside[best]) if best is not None else None,'best_cover_inside_expanded_GT':bool(not outside_expanded[best]) if best is not None else None,'best_cover_own_effective_target_weight':own,'native_fg_count':int(assigned.sum()),'native_effective_positive_count':int(effective.sum()),'max_native_own_target_weight':float(b['target'][assigned].sum(axis=1).max()) if assigned.any() else 0.,'native_fg_outside_original_GT':int((assigned&~inside).sum()),'native_effective_outside_original_GT':int((effective&~inside).sum()),'native_fg_outside_expanded_GT':int((assigned&outside_expanded).sum()),'native_effective_outside_expanded_GT':int((effective&outside_expanded).sum())}
                if np.any(effective&outside_expanded):geometric_violations.append({'uid':uid,'branch':k,'count':int((effective&outside_expanded).sum())})
            rows.append(record)
        if (case['index']+1)%20==0:print(json.dumps({'cached_images_analyzed':case['index']+1}),flush=True)
    assert len(rows)==4491
    selected=[x for x in rows if x['small']];summary={}
    assert len(selected)==4129
    for k in ['M','O']:
        missed=[x for x in selected if not x[k]['matched_small_at025']]
        def count(seq,fn):return sum(bool(fn(x[k])) for x in seq)
        summary[k]={'small_GT':len(selected),'missed_small_at025':len(missed),'small_with_raw_cover':count(selected,lambda y:y['raw_cover_exists']),'missed_with_raw_cover':count(missed,lambda y:y['raw_cover_exists']),'missed_with_raw_cover_score_below025':count(missed,lambda y:y['raw_cover_exists'] and y['best_cover_score']<.25),'missed_with_raw_cover_score_ge025':count(missed,lambda y:y['raw_cover_exists'] and y['best_cover_score']>=.25),'small_zero_effective_native_positives':count(selected,lambda y:y['native_effective_positive_count']==0),'missed_zero_effective_native_positives':count(missed,lambda y:y['native_effective_positive_count']==0),'missed_raw_cover_but_zero_effective_native_positives':count(missed,lambda y:y['raw_cover_exists'] and y['native_effective_positive_count']==0),'missed_low_cover_score_and_zero_effective_native_positives':count(missed,lambda y:y['raw_cover_exists'] and y['best_cover_score']<.25 and y['native_effective_positive_count']==0),'missed_best_cover_outside_original_GT':count(missed,lambda y:y['raw_cover_exists'] and not y['best_cover_inside_original_GT']),'sum_native_effective_small_anchors':sum(x[k]['native_effective_positive_count'] for x in selected),'sum_native_effective_small_anchors_outside_original_GT':sum(x[k]['native_effective_outside_original_GT'] for x in selected),'sum_native_effective_small_anchors_outside_expanded_GT':sum(x[k]['native_effective_outside_expanded_GT'] for x in selected)}
    out={'status':'DESCRIPTIVE_RAW_SUPPORT_AND_FINAL_CHECKPOINT_ASSIGNMENT_AUDIT','all_valid_GT':len(rows),'small_GT_by_canonical_area':len(selected),'coordinate_scope':'Canonical FP64 area bins;IoU/geometry use saved FP32 assignment GT promoted toFP64,so boundary-near raw support is descriptive.','summary':summary,'rows':rows,'effective_assignment_geometry_violations':geometric_violations,'new_model_calls':0,'new_optimizer_updates':0,'mainline_or_causal_mechanism_validated':False,'limitations':read(R/'raw_support_analysis_plan.json')['limits']}
    with (R/'raw_support_result.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['rows','limitations','effective_assignment_geometry_violations']},ensure_ascii=False))
if __name__=='__main__':main()
