"""Read both saved native BASE12 and primary enhanced12 caches for the same tiny GT."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    path=ROOT/'英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py'
    spec=importlib.util.spec_from_file_location('actual_existing_ultratiny_pairs',path)
    tools=importlib.util.module_from_spec(spec);spec.loader.exec_module(tools)
    protocol,geometry,onset,rank,readers=tools.prepared()
    cache=ROOT/'outputs/stem_primary_joint_analysis12_20261011_v1'
    assert sha(cache/'report.json')=='787c0a77fe0aa4027097b5cc6f5cb571c6c66d4fb536a85e71382bcfabaf56bd'
    for n,h in read(cache/'manifest.json')['files'].items():assert sha(cache/n)==h
    rows=[]
    for seed in (1,2):
        base=tools.cached_baseline(seed,protocol,readers['full']);bv=base['values']
        with np.load(cache/f'seed{seed}_original_stem/stage_per_GT.npz',allow_pickle=False) as z:
            fv={n:z[n].copy() for n in z.files}
        for n in ('annotation_id','image_id','category_id','original_area'):
            assert np.array_equal(bv[n],fv[n])
        mask=tools.groups(base['annotation'],bv)['input_under4']&(bv['original_area']<=1024)
        assert int(mask.sum())==5096 and not np.any(bv['original_area']==1024)
        categories={int(c['id']):c['name'] for c in base['annotation']['categories']}
        groups={'all_under4_small':mask}|{str(k)+'_'+name:mask&(bv['category_id']==k) for k,name in categories.items()}
        assert len(categories)==10
        descriptions={}
        for name,m in groups.items():
            comparisons={}
            for t,ti in ((.5,0),(.75,5)):
                arms={}
                for arm,v in (('cached_native_BASE12',bv),('primary_enhanced_Stem12',fv)):
                    arms[arm]=dict(supported_GT_by_boundary={stage:int((m&(v['max_iou'][:,i]>=t)).sum())
                        for i,stage in enumerate(geometry.STAGES)},native_all_area_TP=int((m&v['native_TP'][:,ti]).sum()))
                comparisons[str(t)]=arms
            descriptions[name]=dict(GT=int(m.sum()),thresholds=comparisons)
        rows.append(dict(seed=seed,groups=descriptions,provenance=dict(
            cached_source_report_sha256=base['source_report_sha256'],
            cached_stage_manifest_sha256=base['cache_stage_manifest_sha256'],
            cached_rank_manifest_sha256=base['cache_rank_manifest_sha256'],
            full_primary_joint_manifest_sha256=sha(cache/'manifest.json'),
            annotation_sha256=base['summary']['identity']['val_annotation_sha256'])))
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    value=dict(status='EXISTING_BOTH_SEEDS_SAME_GT_NATIVE_BASE12_AND_PRIMARY_ENHANCED12_UNDER4_DESCRIPTION',
        analysis_source_sha256=sha(__file__),prepared_protocol_sha256=sha(tools.PROTOCOL),seeds=rows,
        all10_classes_retained_including_empty_groups=True,independent_enhanced_saved_EMA_replay_pending=True,
        enhanced_vs_native_is_NOT_architecture_only_control=True,matched_unit='GT ID only, not corresponding query/box',
        small_original_area_closed_at1024=True,first_boundary='Selected300 REGRESSED boxes',
        unique_cause_proven=False,new_BASE_epochs=0,new_BASE_forwards=0,new_evaluator_calls=0,
        new_optimizer_updates=0,new_training_epochs=0,
        scope='Existing native BASE12 and already verified primary enhanced12 caches only. This under4-small cohort is an exploratory subset; all-area native GT flags are not APs-specific matching. Differences include loss of native Stem pretrained states in the direct enhanced replacement. No new BASE/model/evaluator execution.')
    output=ROOT/'outputs/stem_new_controls_primary_collection12_20261011_v1/existing_ultratiny_pairs.json'
    with output.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=value['status'],seeds=[dict(seed=r['seed'],values=r['groups']['all_under4_small']) for r in rows])))


if __name__=='__main__':main()
