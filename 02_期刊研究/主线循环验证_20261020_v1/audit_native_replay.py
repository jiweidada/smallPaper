"""Diagnose the failed replay gate. No candidate metrics are evaluated."""
import contextlib, gzip, io, json, sys
from pathlib import Path
import postprocessing_probe as p
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

p.torch.set_num_threads(2)
process=p.psutil.Process(); process.nice(p.psutil.BELOW_NORMAL_PRIORITY_CLASS)
process.cpu_affinity(process.cpu_affinity()[-1::-2][:2])
out=p.ROOT/'native_replay_audit'; out.mkdir(exist_ok=False)
with contextlib.redirect_stdout(io.StringIO()): coco=COCO(str(p.ANN))
for seed in (1,2):
    folder=p.COLLECTION/f'seed{seed}_original_stem/extracted'
    predictions=[]
    for iid in sorted(coco.getImgIds()):
        with np.load(folder/f'diagnostic/stages_epoch12/image_{iid:06d}.npz') as z:
            probability=p.torch.from_numpy(z['final_logits'].copy()).sigmoid()
            coords=p.xywh_original(p.torch.from_numpy(z['final_boxes'].copy()), z['original_hw'])
        scores, flat=p.torch.topk(probability.flatten(),300)
        for score,f in zip(scores.tolist(),flat.tolist()):
            if score>.001: predictions.append(dict(image_id=iid, category_id=f%10,bbox=coords[f//10].tolist(),score=score))
    with contextlib.redirect_stdout(io.StringIO()):
        dt=coco.loadRes(predictions); ev=COCOeval(coco,dt,'bbox'); ev.evaluate();ev.accumulate();ev.summarize()
    with gzip.open(folder/'diagnostic/endpoint_epoch12/matching_all_area.json.gz','rt') as f: old=json.load(f)
    current={(x['image_id'],x['category_id']):x for x in ev.evalImgs if x is not None and x['aRng']==[0,1e10]}
    matching_differences=[]; scoremax=0.; scorechanged=0; scored=0
    for row in old:
        new=current[(row['image_id'],row['category_id'])]
        keys=['dtIds','dtMatches','gtMatches','dtIgnore','gtIgnore']
        different=[k for k in keys if not np.array_equal(np.asarray(new[k]),np.asarray(row[k]))]
        if different: matching_differences.append(dict(image_id=row['image_id'],category_id=row['category_id'],keys=different))
        if np.array_equal(new['dtIds'],row['dtIds']):
            error=np.abs(np.asarray(new['dtScores'])-row['dtScores'])
            scored+=len(error);scorechanged+=int((error!=0).sum());scoremax=max(scoremax,float(error.max(initial=0)))
    with np.load(folder/'diagnostic/endpoint_epoch12/precision_recall.npz') as z:
        precision_old=z['precision']; recall_old=z['recall']
    delta=ev.eval['precision']-precision_old
    ii=np.argwhere(delta!=0)
    report=dict(seed=seed,old_gate_retained=True,candidate_evaluations=0,
        overall_stats=ev.stats.tolist(),matching_differences=matching_differences,
        scores_compared=scored,scores_changed=scorechanged,max_score_abs_difference=scoremax,
        recall_max_abs_difference=float(np.abs(ev.eval['recall']-recall_old).max()),
        precision_nonidentical_cells=len(ii),precision_max_abs_difference=float(np.abs(delta).max()),
        precision_changes=[dict(index=x.tolist(),old=float(precision_old[tuple(x)]),current=float(ev.eval['precision'][tuple(x)])) for x in ii[:40]],
        interpretation='Only an audit; no relaxation of the failed native replay gate and no candidate conclusions.')
    p.dump(out/f'seed{seed}.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['precision_changes','matching_differences']},ensure_ascii=False),flush=True)
