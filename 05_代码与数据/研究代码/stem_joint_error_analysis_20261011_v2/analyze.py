"""Exploratory GT-identity join of existing BASE12 stage and final-selection caches."""
from pathlib import Path
import argparse, importlib.util, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
ANALYSIS=ROOT/'英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py'
spec=importlib.util.spec_from_file_location('joint_saved_stages',ANALYSIS)
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
read,write,sha=source.read,source.write,source.sha
PROTOCOL=HERE/'protocol.json'
LABELS=('native_TP','miss_never_supported_at_three_boundaries','miss_earlier_supported_lost_at_final',
    'miss_final_supported_no_true_class_after_top300_score_filter',
    'miss_final_supported_true_class_removed_by_class100',
    'miss_final_supported_retained_true_class_no_native_TP','miss_uncertain_ranking_reconstruction')

def freeze():
    p=read(ANALYSIS.parent/'protocol.json')
    protocol=dict(schema='joint_existing_BASE12_error_partition_v1',
        analysis_sha256=sha(HERE/'analyze.py'),stage_analysis_protocol_sha256=sha(ANALYSIS.parent/'protocol.json'),
        stage_analysis_reader_sha256=sha(ANALYSIS),baseline_caches=p['baseline_caches'],
        seeds=[1,2],thresholds=[0.5,0.75],labels=list(LABELS),
        observational_exploratory=True,joined_unit='same GT annotation ID, image ID and category ID',
        boundary_names=['selected300_initial_reference','decoder_layer0_pre_box','final_query_box'],
        scope='Descriptions of saved BASE12 tensors, not AP decomposition, error root identification, dense8400 candidate coverage or training supervision. Categories after top300 combine class score and selection/filter effects; no unique class-head cause.')
    write(PROTOCOL,protocol);print('JOINT_BASE12_PROTOCOL_FROZEN',sha(PROTOCOL))

def partition(stage,ranking,t):
    suffix='50' if t==.5 else '75';ti=0 if t==.5 else 5
    support=stage['max_iou']>=t;native=stage['native_TP'][:,ti]
    assert np.array_equal(native,ranking['native_TP_'+suffix])
    assert np.array_equal(support[:,2],ranking['geometry_'+suffix])
    assert not np.any(native & ~support[:,2])
    cell=ranking['cell_'+suffix];out=np.full(len(native),-1,dtype=np.int8)
    out[native]=0
    out[~native & ~support.any(1)]=1
    out[~native & support[:,:2].any(1) & ~support[:,2]]=2
    mapping={
        'no_true_class_support_after_top300_and_score_threshold':3,
        'true_class_support_removed_by_class100':4,
        'true_class_retained_geometry_without_native_TP':5,
        'uncertain_ranking_reconstruction':6}
    for name,tag in mapping.items():
        mask=~native & support[:,2] & (cell==name)
        assert np.all(out[mask]==-1);out[mask]=tag
    assert np.all(out>=0)
    pattern=(support*np.array([4,2,1])).sum(1).astype(np.int8)
    return out,pattern

def summarize(mask,label,pattern,ranking,t):
    n=int(mask.sum());counts={name:int((mask&(label==i)).sum()) for i,name in enumerate(LABELS)}
    assert sum(counts.values())==n
    missed=n-counts['native_TP']
    cells={format(i,'03b'):{name:int((mask&(pattern==i)&(label==j)).sum())
        for j,name in enumerate(LABELS)} for i in range(8)}
    assert sum(sum(x.values()) for x in cells.values())==n
    return dict(GT=n,native_missed=missed,counts=counts,
        fraction_of_missed={k:(v/missed if missed else None) for k,v in counts.items() if k!='native_TP'},
        observed_stage_pattern_by_final_outcome=cells,
        ranking_ambiguous_GT=int((mask&ranking['ranking_ambiguous']).sum()))

def analyze(output):
    protocol=read(PROTOCOL)
    assert protocol['analysis_sha256']==sha(HERE/'analyze.py')
    assert protocol['stage_analysis_reader_sha256']==sha(ANALYSIS)
    assert protocol['stage_analysis_protocol_sha256']==sha(ANALYSIS.parent/'protocol.json')
    p,geometry,onset,rank,readers=source.prepared()
    assert p['baseline_caches']==protocol['baseline_caches']
    destination=Path(output);assert not destination.exists()
    results=[];arrays={};evidence=[]
    for seed in protocol['seeds']:
        base=source.cached_baseline(seed,p,readers['full']);stage=base['values']
        folder=ROOT/p['baseline_caches'][str(seed)]['rank']
        with np.load(folder/'per_GT.npz',allow_pickle=False) as z:ranking={k:z[k].copy() for k in z.files}
        for name in ('annotation_id','image_id','category_id','original_area'):
            assert np.array_equal(stage[name],ranking[name]),name
        groups=source.groups(base['annotation'],stage)
        side=ranking['input_shortside']
        assert np.array_equal(groups['input_under4'],side<4)
        assert np.array_equal(groups['input_4_to_under8'],(side>=4)&(side<8))
        groups.update({
            'COCO_small_input_under4':groups['COCO_small'] & (side<4),
            'COCO_small_input_4_to_under8':groups['COCO_small'] & (side>=4)&(side<8),
            'COCO_small_input_8_to_under16':groups['COCO_small'] & (side>=8)&(side<16),
            'COCO_small_input_ge16':groups['COCO_small'] & (side>=16)})
        categories={int(c['id']):c['name'] for c in base['annotation']['categories']}
        for cid in sorted(set(stage['category_id'].tolist())):
            groups['COCO_small_class_'+str(cid)+'_'+categories[cid]]=groups['COCO_small']&(stage['category_id']==cid)
        paired={k:stage[k] for k in ('annotation_id','image_id','category_id','original_area')}
        seedrows={}
        for t in protocol['thresholds']:
            label,pattern=partition(stage,ranking,t)
            suffix='50' if t==.5 else '75'
            paired['joint_label_'+suffix]=label;paired['support_pattern_'+suffix]=pattern
            seedrows[str(t)]={name:summarize(mask,label,pattern,ranking,t) for name,mask in groups.items()}
            # Agreement with the previously computed native-score-checked ranking partition.
            for name in ('all','COCO_small','input_under4','input_4_to_under8'):
                old=base['rank_report']['groups'][name][suffix]['cells'];current=seedrows[str(t)][name]['counts']
                assert current['native_TP']==old['native_TP']
                assert current['miss_never_supported_at_three_boundaries']+current['miss_earlier_supported_lost_at_final']==old['no_final_geometric_support']
                assert current['miss_uncertain_ranking_reconstruction']==old.get('uncertain_ranking_reconstruction',0)
        arrays[seed]=paired
        results.append(dict(seed=seed,cached_BASE12_AP_percent=base['summary']['overall_percent'],groups=seedrows))
        evidence.append(dict(seed=seed,baseline_checkpoint_sha256=base['summary']['identity']['baseline_checkpoint_sha256'],
            source_report_sha256=base['source_report_sha256'],stage_manifest_sha256=base['cache_stage_manifest_sha256'],
            rank_manifest_sha256=base['cache_rank_manifest_sha256']))
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    destination.mkdir()
    for seed,a in arrays.items():np.savez_compressed(destination/f'joint_per_GT_seed{seed}.npz',**a)
    fallacies={
        '1_Simpson':'Both seeds, input sizes and every small-object class retained; no aggregation used for a causal claim.',
        '2_Ecological':'GT-ID joins define descriptive individual outcomes; group rates do not identify per-object mechanisms.',
        '3_Berkson':'Counts condition on saved valid GT and selected300 queries; full dense candidate universe is not observed.',
        '4_Collider':'Failure and geometric-support strata are outcomes; their associations are not causal effects.',
        '5_Base_rates':'GT and native-missed denominators both reported; GT counts are not AP weights.',
        '6_Regression_to_mean':'No intervention or before-after efficacy estimated.',
        '7_Survivorship':'Every joined GT, uncertain ranking and both seeds retained; no drop of hard cases.',
        '8_Look_elsewhere':'Exploratory subgroup census, no significance tests or positive-only selection.',
        '9_Forking_paths':'Analysis version frozen before execution; prior aggregate counts were already known, not confirmatory preregistration.',
        '10_Correlation_causation':'First observed support loss is a propagation boundary, not an identified learning or Stem cause.',
        '11_Reverse_causality':'Validation support and ranking outcomes do not reconstruct historical Hungarian training assignments.'}
    report=dict(status='ANALYZED_EXISTING_BASE12_JOINT_GEOMETRY_AND_SELECTION_PARTITION',
        material_passport=dict(skill='academic-research-suite',workflow='experiment-agent',mode='validate',
            verification_status='ANALYZED',data_level='Saved per-GT caches, exact manifests and identity joins verified; no training or evaluator rerun.'),
        protocol_sha256=sha(PROTOCOL),evidence=evidence,seeds=results,labels=list(LABELS),
        statistical_fallacy_coverage=dict(checked=11,total=11,items=fallacies),
        new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,new_training_epochs=0,
        limitation=protocol['scope'],new_Stem_effect_or_root_cause_established=False)
    write(destination/'report.json',report)
    write(destination/'manifest.json',dict(files={p.name:sha(p) for p in destination.iterdir() if p.is_file()}))
    print(json.dumps(dict(status=report['status'],small=[dict(seed=r['seed'],
        IoU50=r['groups']['0.5']['COCO_small']['counts'],IoU75=r['groups']['0.75']['COCO_small']['counts']) for r in results])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--output');a=p.parse_args()
    if a.freeze:freeze()
    else:
        assert a.output;analyze(a.output)
