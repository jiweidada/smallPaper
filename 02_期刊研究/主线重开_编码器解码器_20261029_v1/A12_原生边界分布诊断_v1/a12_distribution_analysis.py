"""Frozen descriptive signal screen; no fitting,detector inference or accuracy claim."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import numpy as np


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def score_pairs(good,bad,key):
    a=good[key];b=bad[key]
    return 1. if b>a else (0. if b<a else .5)


def analyze(root):
    root=Path(root)
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    assets=json.loads((root/'asset_selection.json').read_text(encoding='utf-8'))
    image_rows=[];edge_rows=[];pair_rows=[]
    for index,case in enumerate(assets['cases']):
        with np.load(root/'predictions'/('case%02d.npz'%index),allow_pickle=False) as z:
            corners=z['corner_logits'][-1].astype(np.float64)
            grid=z['bin_grid'].astype(np.float64)
            ref=z['ref_initial'].astype(np.float64)
            gt=z['gt_boxes800'].astype(np.float64)
            sizes=z['gt_sizes800'].astype(np.float64)
            labels=z['gt_labels']
            pairs=list(zip(z['match_q'].tolist(),z['match_gt'].tolist()))
            boxes=z['raw_boxes'].astype(np.float64)
        logits=corners-corners.max(-1,keepdims=True)
        probabilities=np.exp(logits);probabilities/=probabilities.sum(-1,keepdims=True)
        mean=(probabilities*grid).sum(-1)
        variance=(probabilities*(grid-mean[...,None])**2).sum(-1)
        interval=np.clip(np.searchsorted(grid,mean,side='right')-1,0,len(grid)-2)
        floor=(mean-grid[interval])*(grid[interval+1]-mean)
        excess=variance-floor
        assert np.min(excess)>=-1e-10
        excess=np.maximum(excess,0.)
        entropy=-(probabilities*np.log(np.maximum(probabilities,1e-300))).sum(-1)
        rows=[];outside=0;tiny_matches=0
        for qi,gi in pairs:
            if sizes[gi]>=8:continue
            tiny_matches+=1
            cx,cy,w,h=ref[qi]
            ref_scale=np.asarray([w,h,w,h])/4.
            if not np.all(ref_scale>0):continue
            normalized_gt=gt[gi]/800.
            target=np.asarray([(cx-normalized_gt[0])/ref_scale[0]-2.,(cy-normalized_gt[1])/ref_scale[1]-2.,(normalized_gt[2]-cx)/ref_scale[2]-2.,(normalized_gt[3]-cy)/ref_scale[3]-2.])
            b=boxes[qi]
            box800=np.r_[b[:2]-b[2:]*.5,b[:2]+b[2:]*.5]*800.
            for side in range(4):
                if not grid[0]<=target[side]<=grid[-1]:
                    outside+=1;continue
                physical_scale=ref_scale[side]*800.
                cell_width=(grid[interval[qi,side]+1]-grid[interval[qi,side]])*physical_scale
                if cell_width<=0:continue
                row={'image':index,'prefix':case['prefix'],'gt_index':gi,'query':qi,'class_id':int(labels[gi]),'side':side,'interval':int(interval[qi,side]),'bin_width_px':float(cell_width),'edge_error_px':float(abs(box800[side]-gt[gi,side])),'entropy':float(entropy[qi,side]),'raw_variance_px2':float(variance[qi,side]*physical_scale**2),'excess_variance_px2':float(excess[qi,side]*physical_scale**2),'interpolation_floor_px2':float(floor[qi,side]*physical_scale**2),'predicted_phase':float((mean[qi,side]-grid[interval[qi,side]])/(grid[interval[qi,side]+1]-grid[interval[qi,side]]))}
                rows.append(row);edge_rows.append(row)
        good=[r for r in rows if r['edge_error_px']<=.5]
        bad=[r for r in rows if r['edge_error_px']>=1.]
        scores=[]
        for g in good:
            for b in bad:
                if (g['side'],g['interval'],g['class_id'])!=(b['side'],b['interval'],b['class_id']):continue
                if max(g['bin_width_px'],b['bin_width_px'])/min(g['bin_width_px'],b['bin_width_px'])>2.:continue
                r={'image':index,'good_gt':g['gt_index'],'bad_gt':b['gt_index'],'side':g['side'],'interval':g['interval'],'class_id':g['class_id']}
                for key in ('entropy','raw_variance_px2','excess_variance_px2'):r[key]=score_pairs(g,b,key)
                scores.append(r);pair_rows.append(r)
        image={'image':index,'file':case['file'],'prefix':case['prefix'],'tiny_GT':case['tiny_GT'],'matched_tiny_GT':tiny_matches,'eligible_edges':len(rows),'out_of_support_edges':outside,'good_edges':len(good),'bad_edges':len(bad),'comparable_pairs':len(scores)}
        for key in ('entropy','raw_variance_px2','excess_variance_px2'):
            image[key+'_pair_discrimination']=float(np.mean([r[key] for r in scores])) if scores else None
        if scores:
            image['primary_difference']=image['excess_variance_px2_pair_discrimination']-image['raw_variance_px2_pair_discrimination']
            image['entropy_difference']=image['excess_variance_px2_pair_discrimination']-image['entropy_pair_discrimination']
        else:image['primary_difference']=image['entropy_difference']=None
        image_rows.append(image)
    kept=[r for r in image_rows if r['comparable_pairs']>0]
    total_pairs=len(pair_rows)
    if kept:
        groups=sorted({r['prefix'] for r in kept})
        values=[[r['primary_difference'] for r in kept if r['prefix']==g] for g in groups]
        rng=np.random.Generator(np.random.PCG64(protocol['bootstrap_seed']))
        boot=[np.mean([x for i in rng.integers(len(groups),size=len(groups)) for x in values[i]]) for _ in range(2000)]
        primary={'mean':float(np.mean([r['primary_difference'] for r in kept])),'low':float(np.quantile(boot,.025)),'high':float(np.quantile(boot,.975)),'images':len(kept),'prefixes':len(groups)}
        entropy_difference=float(np.mean([r['entropy_difference'] for r in kept]))
    else:primary={'mean':None,'low':None,'high':None,'images':0,'prefixes':0};entropy_difference=None
    enough=len(kept)>=8 and primary['prefixes']>=8 and assets['selected_tiny_GT']>=120 and total_pairs>=200
    if not enough:decision='INSUFFICIENT_COMPARABLE_PAIRS_NO_MODULE_OR_AUTOMATIC_EXPANSION'
    elif primary['mean']>=.10 and primary['low']>0 and entropy_difference>=0:decision='RETAIN_CORRECTED_SPREAD_FOR_CAUSAL_STUDY_NOT_VALIDATED_MAINLINE'
    else:decision='DEPRIORITIZE_PHASE_CORRECTION_AS_CURRENT_MAINLINE_NO_MODULE_TRAINING'
    summary={'utc':datetime.now(timezone.utc).isoformat(),'status':'COMPLETED_FROZEN_OBSERVATIONAL_SIGNAL_SCREEN','decision':decision,'selected_images':len(assets['cases']),'selected_tiny_GT':assets['selected_tiny_GT'],'eligible_edges':len(edge_rows),'comparable_pairs':total_pairs,'enough_comparable_data':enough,'primary_excess_minus_raw_variance':primary,'secondary_excess_minus_entropy':entropy_difference,'new_fits':0,'main_model_updates':0,'mainline_validated':False,'limits':['Pair discrimination is descriptive signal ranking,not AP,calibration or a causal gain.','Only selected dense official-val images;not a blind test or generalization guarantee.','No learned regression/readout and no intervention tested.','Raw entropy is a secondary comparator;primary was fixed as corrected versus raw variance with matched class,side,bin and approximate physical spacing.']}
    write(root/'analysis_summary.json',summary)
    write(root/'per_image_analysis.json',image_rows)
    write(root/'edge_rows.json',edge_rows)
    write(root/'pair_rows.json',pair_rows)
    return summary


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args()
    print(json.dumps(analyze(a.root)))
