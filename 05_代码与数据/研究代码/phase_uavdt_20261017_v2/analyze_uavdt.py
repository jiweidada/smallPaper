"""Fixed test-set size/paired-localization diagnostics; no model or threshold selection."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from uavdt_protocol import NativeScorer,contained_mask,curve,match_frame

BINS=[('<4',0,4),('4-8',4,8),('8-16',8,16),('16-32',16,32),('>=32',32,float('inf'))]

def group_rows(rows):
    rows=rows[np.argsort(rows[:,0],kind='stable')]
    return {int(block[0,0]):block[:,1:] for block in np.split(rows,np.flatnonzero(np.diff(rows[:,0]))+1)} if len(rows) else {}

def quantiles(values):
    a=np.asarray(values,dtype=float)
    return {'n':len(a),'mean':float(a.mean()) if len(a) else None,'p10_p25_p50_p75_p90':np.percentile(a,[10,25,50,75,90]).tolist() if len(a) else None}

def normalized_error(pred,gt):
    center=np.hypot(((pred[:,0]+pred[:,2]/2)-(gt[:,0]+gt[:,2]/2))/gt[:,2],
                    ((pred[:,1]+pred[:,3]/2)-(gt[:,1]+gt[:,3]/2))/gt[:,3])
    return np.column_stack([center,np.log(pred[:,2]/gt[:,2]),np.log(pred[:,3]/gt[:,3])])

def main(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());data=Path(plan['data_root'])
    annotation=json.loads((data/'annotations/test.json').read_text());scorer=NativeScorer(annotation)
    out=root/'analysis';out.mkdir(exist_ok=False)
    by_image={i:[] for i in scorer.images}
    for index,a in enumerate(annotation['annotations']):by_image[a['image_id']].append(index)
    gt=np.asarray([a['bbox'] for a in annotation['annotations']],dtype=float);gt_size=np.zeros(len(gt))
    for i,indices in by_image.items():
        image=scorer.images[i]
        gt_size[indices]=np.sqrt(gt[indices,2]*gt[indices,3]*640/image['width']*640/image['height'])
    summaries={};matched={};sources=[]
    for arm in ['baseline','c','c_mean']:
        job=f'uavdt_{arm}_full96_seed{plan["seed"]}';source=root/'results'/job/'test/predictions.npz'
        with np.load(source,allow_pickle=False) as pack:rows=pack['detections']
        grouped=group_rows(rows);del rows
        bin_parts={name:[] for name,_,_ in BINS};matches={t:np.full((len(gt),4),np.nan) for t in [.5,.75]}
        for i in scorer.order:
            image=scorer.images[i];dt=grouped.pop(i,np.empty((0,5)))
            if image['uav_apply_native_ignore']:dt=dt[~contained_mask(dt,image['uav_ignore_boxes'])]
            indices=np.asarray(by_image[i],dtype=int);g=scorer.gt[i]
            scores,assignment,order=match_frame(g,dt,.7);ordered=dt[order]
            dt_size=np.sqrt(np.maximum(0,ordered[:,2]*ordered[:,3])*640/image['width']*640/image['height'])
            gsize=gt_size[indices];is_match=assignment>=0
            for label,lo,hi in BINS:
                eligible=np.zeros(len(dt),dtype=bool)
                eligible[is_match]=(gsize[assignment[is_match]]>=lo)&(gsize[assignment[is_match]]<hi)
                eligible[~is_match]=(dt_size[~is_match]>=lo)&(dt_size[~is_match]<hi)
                bin_parts[label].append((scores[eligible],is_match[eligible].astype(np.int8)))
            selected=dt[dt[:,4]>=.3]
            for threshold in [.5,.75]:
                _,a,o=match_frame(g,selected,threshold);keep=a>=0
                if keep.any():matches[threshold][indices[a[keep]]]=selected[o[keep],:4]
        assert not grouped
        summary={}
        for label,lo,hi in BINS:
            n=int(((gt_size>=lo)&(gt_size<hi)).sum());parts=bin_parts[label]
            score=np.concatenate([p[0] for p in parts]);flags=np.concatenate([p[1] for p in parts])
            metric=curve(score,flags,n)[0] if n else None
            summary[label]={'GT':n,'AP70_640bin_native_assignment':metric['AP70'] if n else None,'AR70_640bin_native_assignment':metric['AR70'] if n else None}
        summaries[arm]=summary;matched[arm]=matches
        h=hashlib.sha256()
        with source.open('rb') as f:
            for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
        sources.append({'arm':arm,'path':str(source),'sha256':h.hexdigest()})
    pairs=[];count_rows=[]
    for candidate,reference in [('c','baseline'),('c','c_mean'),('c_mean','baseline')]:
        for threshold in [.5,.75]:
            left=matched[candidate][threshold];right=matched[reference][threshold]
            detected_l=np.isfinite(left[:,0]);detected_r=np.isfinite(right[:,0]);common=detected_l&detected_r
            ids=np.flatnonzero(common);errors_l=normalized_error(left[common],gt[common]);errors_r=normalized_error(right[common],gt[common])
            np.savez_compressed(out/f'{candidate}_minus_{reference}_iou{int(threshold*100)}_common_errors.npz',GT_annotation_indices=ids,candidate_errors=errors_l,reference_errors=errors_r)
            details=[]
            for label,lo,hi in BINS+[('all',0,float('inf'))]:
                mask=(gt_size>=lo)&(gt_size<hi);selected=mask[ids]
                counts={'GT':int(mask.sum()),'common':int((mask&common).sum()),'candidate_only':int((mask&detected_l&~detected_r).sum()),
                        'reference_only':int((mask&~detected_l&detected_r).sum()),'both_missed':int((mask&~detected_l&~detected_r).sum())}
                assert sum(counts[k] for k in ['common','candidate_only','reference_only','both_missed'])==counts['GT']
                row={'comparison':candidate+' - '+reference,'IoU':threshold,'bin':label,**counts}
                details.append(dict(row,candidate_common_errors={name:quantiles(values) for name,values in zip(['normalized_center','log_width','log_height'],errors_l[selected].T)},
                                    reference_common_errors={name:quantiles(values) for name,values in zip(['normalized_center','log_width','log_height'],errors_r[selected].T)}));count_rows.append(row)
            pairs.append({'comparison':candidate+' - '+reference,'IoU':threshold,'bins':details})
    record={'dataset':'UAVDT_v1','seed':plan['seed'],'split':'official_test','score_threshold_for_localization':.3,
            'size_definition':'sqrt(wh) after whole-image stretch to640','size_AP_definition':'Native full-GT AP70 greedy assignment first; detections assigned to out-of-bin GT and unmatched out-of-bin-size detections ignored. Custom diagnostic, not COCO APs or an official UAVDT metric.',
            'empty_GT_bin':None,'metrics_by_arm':summaries,'paired_localization':pairs,'sources':sources,
            'claim_limit':'Same-seed controlled descriptions; no significance, equivalence or cross-domain claim.'}
    with (out/'size_localization.json').open('x') as f:json.dump(record,f,indent=2)
    with (out/'paired_detection_counts.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(count_rows[0]));writer.writeheader();writer.writerows(count_rows)
    print(json.dumps({'analysis':str(out),'GT':len(gt),'pairs':len(pairs),'empty_bins_are_NA':True}),flush=True)

if __name__=='__main__':main(sys.argv[1])
