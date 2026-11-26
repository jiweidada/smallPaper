"""Exploratory no-fit signal discrimination audit, clustered by the 100 source images."""
from pathlib import Path
from collections import defaultdict
import json,numpy as np
ROOT=Path('/root/c10_region_restore_20261029_v1');OUT=ROOT/'local_proxy_v1';REG=ROOT/'restoration_probe_v1'
def auc(scores,positive):
    scores=np.asarray(scores);positive=np.asarray(positive,dtype=bool)
    p=scores[positive];n=scores[~positive]
    if not len(p) or not len(n):return None
    return float(np.mean((p[:,None]>n[None,:])+.5*(p[:,None]==n[None,:])))
def stats(rows,score):
    if not rows:return dict(regions=0)
    y=np.asarray([r['utility']>0 for r in rows]);s=np.asarray([r[score] for r in rows]);chosen=s>0
    tpr=float(np.mean(chosen[y])) if y.any() else None;tnr=float(np.mean(~chosen[~y])) if (~y).any() else None
    return dict(regions=len(rows),beneficial_fraction=float(y.mean()),selected_fraction=float(chosen.mean()),
      sign_agreement=float(np.mean(chosen==y)),all_restore_sign_agreement=float(y.mean()),AUROC=auc(s,y),
      beneficial_region_recall=tpr,harmful_region_specificity=tnr,
      balanced_accuracy=(tpr+tnr)/2 if tpr is not None and tnr is not None else None)
gt=json.loads((ROOT/'dataset/diagnostic_gt_orig.json').read_text());meta={i['id']:i for i in gt['images']}
gts=defaultdict(list)
obs={m['image_id']:m for m in json.loads((ROOT/'receipts/diagnostic_observations.json').read_text())['images']}
for a in gt['annotations']:
    if not a['iscrowd'] and a['area']*obs[a['image_id']]['geometry']['ratio']**2<1024:gts[a['image_id']].append(a)
complete=json.loads((REG/'completed.json').read_text());decisions=json.loads((OUT/'decisions.json').read_text());out={}
for arm in ['D0','DA']:
    truths={r['image_id']:r for r in json.loads((REG/(arm+'_loss_diagnostics.json')).read_text())}
    matched=set(next(x for x in complete['metrics'][arm]['subset_15']['matches'] if x['confidence']==.25 and x['iou']==.5)['matched_small_uids'])
    rows=[]
    for d in decisions[arm]:
        iid=d['image_id'];im=meta[iid];regional=defaultdict(list)
        for a in gts[iid]:
            x,y,w,h=a['bbox'];k=2*int(y+h/2>=im['height']/2)+int(x+w/2>=im['width']/2);regional[k].append(a)
        for k in range(4):
            annotations=regional[k];n=len(annotations);hits=sum(a.get('stable_uid',str(a['id'])) in matched for a in annotations)
            label='no_small_GT' if n==0 else 'all_visible_to_candidate' if hits==n else 'none_visible_to_candidate' if hits==0 else 'mixed_visibility'
            rows.append(dict(image_id=iid,region=k,utility=truths[iid]['dynamic']['singles'][k],
              pseudograd=d['regional_pseudograd_scores'][k],confidence=d['confidence_region_deltas'][k],
              small_GT=n,small_GT_candidate_matched=hits,visibility_group=label))
    result=dict(all={score:stats(rows,score) for score in ['pseudograd','confidence']},
                by_candidate_visibility={label:{score:stats([r for r in rows if r['visibility_group']==label],score) for score in ['pseudograd','confidence']} for label in sorted({r['visibility_group'] for r in rows})})
    iid_list=[d['image_id'] for d in decisions[arm]];by_id={iid:[r for r in rows if r['image_id']==iid] for iid in iid_list}
    samples=defaultdict(list);rng=np.random.default_rng(20261029)
    for rep in range(2000):
        draw=rng.integers(0,len(iid_list),len(iid_list));selected=[r for j in draw for r in by_id[iid_list[j]]]
        for score in ['pseudograd','confidence']:
            st=stats(selected,score)
            for key in ['AUROC','balanced_accuracy','sign_agreement']:
                if st[key] is not None:samples[score+'/'+key].append(st[key])
    result['image_cluster_bootstrap95']={k:np.quantile(v,[.025,.975]).tolist() for k,v in samples.items()}
    out[arm]=result
    (OUT/(arm+'_signal_audit_regions.json')).write_text(json.dumps(rows,indent=2))
(OUT/'signal_failure_audit.json').write_text(json.dumps(dict(status='EXPLORATORY_NO_FIT',arms=out,
  grouping='logical four visibility categories, not a causal identification; 400 region rows within100 images',
  scope='Utility uses whole-image native detector loss; candidate visibility only diagnoses a possible explanation, not proof that invisible objects caused errors',
  training_updates=0,detector_forwards=0,thresholds_retuned=0),indent=2))
print(json.dumps(out))
