"""Descriptive joint coverage of six archived predictions; no model execution."""
import csv, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
DATA=ROOT/'英文小论文/data/07_第二数据集_20261017'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    annpath=DATA/'独立中期评估_20261018_v1/data/val.json'
    gt=json.loads(annpath.read_text(encoding='utf-8-sig'))
    anns={a['id']:a for a in gt['annotations']}; images={i['id']:i for i in gt['images']}
    models={}; sources={str(annpath.relative_to(ROOT)):sha(annpath)}
    for seed in [0,1]:
        for arm in ['baseline','c','c_mean']:
            key=f'seed{seed}_{arm}'
            p=DATA/f'第36轮候选覆盖与排序筛查_20261019_v1/models/{key}/per_GT_geometry_and_ranks.npz'
            with np.load(p,allow_pickle=False) as z: models[key]={k:z[k] for k in z.files}
            sources[str(p.relative_to(ROOT))]=sha(p)
    first=models['seed0_baseline']; ids=first['annotation_ids']; keys=list(models)
    for m in models.values():
        assert np.array_equal(ids,m['annotation_ids'])
        assert np.array_equal(first['size640'],m['size640'])
    small=first['size640']<32
    sequences=np.array([images[anns[int(a)]['image_id']]['uav_sequence'] for a in ids])
    tracks=[(sequences[i],anns[int(a)]['uav_track_id']) for i,a in enumerate(ids)]
    ious=np.stack([models[k]['best_iou'] for k in keys]); fails=ious<.70
    joint=fails.all(axis=0); best=ious.max(axis=0)
    patterns=np.sum(fails.astype(np.uint8)*(1<<np.arange(6))[:,None],axis=0)
    rows=[]; patterns_rows=[]
    for seq in ['ALL']+sorted(set(sequences)):
        mask=small & ((sequences==seq) if seq!='ALL' else True)
        for seed in [0,1]:
            b=fails[keys.index(f'seed{seed}_baseline')]; c=fails[keys.index(f'seed{seed}_c')]
            cond=mask&b; shared=cond&joint; rescue=cond&~joint
            pertrack=defaultdict(list)
            for i in np.flatnonzero(cond): pertrack[tracks[i]].append(bool(joint[i]))
            rows.append({'sequence':seq,'seed':seed,'small_GT_frames':int(mask.sum()),'baseline_fail':int(cond.sum()),
                         'all_six_fail':int(shared.sum()),'other_model_covers':int(rescue.sum()),
                         'shared_fraction':float(shared.sum()/cond.sum()) if cond.any() else None,
                         'affected_original_tracks':len(pertrack),
                         'track_equal_shared_fraction':float(np.mean([np.mean(v) for v in pertrack.values()])) if pertrack else None,
                         'shared_best_IoU_q10_q50_q90':np.quantile(best[shared],[.1,.5,.9]).tolist() if shared.any() else [],
                         'shared_best_IoU_lt50':int((shared&(best<.5)).sum()),
                         'C_rescues_baseline':int((mask&b&~c).sum()),'C_loses_baseline':int((mask&~b&c).sum())})
        for pat in np.unique(patterns[mask]):
            ix=np.flatnonzero(mask&(patterns==pat))
            patterns_rows.append({'sequence':seq,'failure_pattern_bits_model_order':format(int(pat),'06b')[::-1],
                                  'GT_frames':len(ix),'original_tracks':len({tracks[i] for i in ix})})
    result={'spec':{'scope':'six fixed36 EMA UAVDT predictions, sqrt area in 640<32',
                   'state':'per-GT all-300 best IoU<.70, ignores ranking and competition; not AP',
                   'model_order':keys,'unit':'descriptive GT frames; correlated within 245 original tracks and six sequences',
                   'meaning':'common failure does not establish irreducibility or select an intervention',
                   'new_forwards':0,'optimizer_updates':0,'script_sha256':sha(Path(__file__))},
            'sources_sha256':sources,'rows':rows,'joint_patterns':patterns_rows}
    assert all(r['all_six_fail']+r['other_model_covers']==r['baseline_fail'] for r in rows)
    (OUT/'14_CPU六模型共失覆盖与余量.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    with (OUT/'14b_CPU六模型模式.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(patterns_rows[0]));w.writeheader();w.writerows(patterns_rows)
    print(json.dumps(rows,ensure_ascii=False))

if __name__=='__main__': main()
