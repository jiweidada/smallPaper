"""Test saved query-scale allocation versus same-site regression differences."""
from pathlib import Path
import json
import hashlib
import datetime
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/random_Stem_query_levels12_20261011_v1'
assert not OUT.exists()
OUT.mkdir()
sources = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    sources[str(path.relative_to(ROOT))] = sha(path)
    return json.loads(path.read_text(encoding='utf-8'))


def load(path):
    sources[str(path.relative_to(ROOT))] = sha(path)
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


all_rows = []
for seed in (1,2):
    root = ROOT/'outputs/native_selection_collection12_20261011_v2'
    trace_root = ROOT/'outputs/native_selection_analysis12_20261011_v5/actual_six'
    values = {arm:load(trace_root/f'{arm}_seed{seed}'/'per_GT.npz') for arm in ('original_stem','native_stem_scratch')}
    for key in ('annotation_id','category_id','image_id','original_area'):
        assert np.array_equal(values['original_stem'][key],values['native_stem_scratch'][key])
    f, r = values['original_stem'], values['native_stem_scratch']
    manifests = {}
    folders = {}
    totals = {arm:np.zeros(3,np.int64) for arm in values}
    per_image = []
    same_site = np.zeros(len(f['annotation_id']),bool)
    baseline_best_level = np.full(len(same_site),-1,np.int8)
    enhanced_best_level = baseline_best_level.copy()
    for arm in values:
        p = root/f'seed{seed}_{arm}'
        verification = read(p/'packet_verification.json')
        assert verification['status']=='PASS_DOWNLOADED_SELECTION_DIAGNOSTIC_PACKET_HASHES_AND_BINDINGS'
        folders[arm]=p/'extracted/diagnostic/selection_epoch12'
        manifests[arm]=read(folders[arm]/'manifest.json')
    for iid in range(548):
        selected={}
        counts={}
        for arm in values:
            name=f'image_{iid:06d}.npz';path=folders[arm]/name
            expected=manifests[arm]['files'][name]
            expected=expected['sha256'] if isinstance(expected,dict) else expected
            assert sha(path)==expected
            with np.load(path,allow_pickle=False) as z:
                indices=z['selected_indices'].copy()
                assert indices.shape==(300,) and len(np.unique(indices))==300
                assert (indices>=0).all() and (indices<8400).all()
                selected[arm]=indices
                level=np.searchsorted([6400,8000],indices,side='right')
                counts[arm]=np.bincount(level,minlength=3)
                totals[arm]+=counts[arm]
        mask=f['image_id']==iid
        q=r['best_query_within_run'][mask,0]
        fq=f['best_query_within_run'][mask,0]
        assert (q>=0).all() and (q<300).all() and (fq>=0).all() and (fq<300).all()
        sites=selected['native_stem_scratch'][q]
        same_site[mask]=np.isin(sites,selected['original_stem'])
        baseline_best_level[mask]=np.searchsorted([6400,8000],sites,side='right')
        enhanced_best_level[mask]=np.searchsorted([6400,8000],selected['original_stem'][fq],side='right')
        per_image.append(dict(image_id=iid,random_counts=counts['native_stem_scratch'].tolist(),enhanced_counts=counts['original_stem'].tolist(),shared_selected_sites=len(np.intersect1d(selected['native_stem_scratch'],selected['original_stem']))))
    area=f['original_area']
    cohorts={'small':area<=1024,'medium':(area>=1024)&(area<=9216),'large':area>=9216}
    support_rows=[]
    for name,mask in cohorts.items():
        for threshold in (.5,.75):
            lost=mask&(r['max_iou'][:,0]>=threshold)&(f['max_iou'][:,0]<threshold)
            gained=mask&(r['max_iou'][:,0]<threshold)&(f['max_iou'][:,0]>=threshold)
            n=int(lost.sum());shared=int((lost&same_site).sum())
            support_rows.append(dict(area=name,IoU_threshold=threshold,lost_initial_support=n,gained_initial_support=int(gained.sum()),
                baseline_best_site_still_selected_by_enhanced=shared,baseline_best_site_not_selected_by_enhanced=n-shared,
                shared_site_fraction=shared/n if n else None,
                lost_initial_support_baseline_best_level={f'P{i+3}':int((lost&(baseline_best_level==i)).sum()) for i in range(3)},
                lost_initial_support_enhanced_best_level={f'P{i+3}':int((lost&(enhanced_best_level==i)).sum()) for i in range(3)}))
    all_rows.append(dict(seed=seed,selected_query_totals={arm:count.tolist() for arm,count in totals.items()},
        mean_selected_queries_per_image={arm:(count/548).tolist() for arm,count in totals.items()},
        enhanced_minus_random_selected_queries=(totals['original_stem']-totals['native_stem_scratch']).tolist(),
        initial_support_losses=support_rows,mean_shared_selected_sites=float(np.mean([x['shared_selected_sites'] for x in per_image]))))
    np.savez_compressed(OUT/f'seed{seed}_physical_site_flags.npz',annotation_id=f['annotation_id'],image_id=f['image_id'],original_area=area,
        baseline_best_initial_site_still_selected=same_site,baseline_best_level=baseline_best_level,enhanced_best_level=enhanced_best_level)
    (OUT/f'seed{seed}_image_counts.json').write_text(json.dumps(per_image,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

report=dict(status='TWO_SEED_ACTUAL_TOPK300_LEVEL_ALLOCATION_AND_INITIAL_LOSS_SAME_SITE_CHECKED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=all_rows,
    feature_levels={'P3':[0,6400],'P4':[6400,8000],'P5':[8000,8400]},
    comparison='Enhanced random Stem versus random native Stem, fixed12 saved EMA diagnostic traces.',
    interpretation='A lost baseline supporting physical site that remains selected in the enhanced model requires an initial bbox regression/content difference; it is not explained solely by removal of that site from TopK. Level allocation changes do not isolate input features, encoder features or bbox-head learned parameters.',
    query_identity='Same physical site is not the same query across models. No cross-model query-index identity is assumed.',
    hypothesis_not_proven='Counts or proximity alone do not establish foreground/background information loss, a capacity root, or a useful query-reservation intervention.',
    new_model_forwards=0,new_training_epochs=0,new_evaluator_calls=0,root_proven=False,novelty_proven=False,sources_sha256=sources)
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in OUT.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf-8')
for row in all_rows:
    print('SEED',row['seed'],'MEAN_LEVEL_QUERIES',row['mean_selected_queries_per_image'])
    for s in row['initial_support_losses']:
        if s['area'] in ('medium','large') and s['IoU_threshold']==.5: print(json.dumps(s,ensure_ascii=False))
