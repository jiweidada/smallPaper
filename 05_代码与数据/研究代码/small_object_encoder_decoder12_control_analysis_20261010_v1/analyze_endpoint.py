"""Read saved fixed12 tensors and native matching; no Torch/model/evaluator calls."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import geometry_core as geometry
from results import read,write,sha,load_endpoint,pair
from stage_trace import verify_trace
from evaluator_binding import same_result_evaluator,validate_result_source

HERE=Path(__file__).resolve().parent
VAL_SHA='0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'
BASE_CHECKPOINTS={1:'ba3b6351fa8a4f7c769698c616ed6d20acf6a60ba119cf8edc8611851c4d27d4',
                  2:'6198ee5348630ce97c322c16aa4499a50a2f7af0d08dea1ec3a4cb632d9d5046'}


def inspect_run(root,arm,annotation):
    root=Path(root); summary,flags,manifest=load_endpoint(root/'endpoint_epoch12')
    identity=summary['identity'];seed=identity['seed']
    assert identity['arm']==arm and seed in (1,2)
    assert identity['val_annotation_sha256']==VAL_SHA
    assert summary['mode']=='native_runtime' and summary['evaluated_weights']=='EMA'
    validate_result_source(summary['evaluator_source'],identity)
    assert manifest['capture_code_sha256']==sha(HERE/'results.py')
    verify_trace(root/'stages_epoch12',identity)
    completion=read(root/('completion.json' if arm=='baseline' else 'training_completion.json'))
    for folder,key in [('endpoint_epoch12','endpoint_manifest_sha256'),('stages_epoch12','stage_manifest_sha256')]:
        assert sha(root/folder/'manifest.json')==completion[key]
    if arm=='baseline':
        assert completion['historical_log_reproduction_pass'] and completion['all_parameters_frozen_like_native_EMA']
        assert completion['checkpoint_sha256']==BASE_CHECKPOINTS[seed]
        historical=read(root/'historical_comparison.json')
        assert historical['historical_stats']==historical['current_stats']==summary['native_stats']
        config=read(root/'baseline.yml')
    else:
        assert completion['status']=='NATIVE_FIT_FINISHED_EXACT12'
        assert completion['completed_epochs']==12 and completion['entered_epochs']==list(range(12))
        assert completion['identity']==identity and sha(root/'last.pth')==completion['last_checkpoint_sha256']
        rows=[json.loads(s) for s in (root/'log.txt').read_text().splitlines() if s.strip()]
        assert len(rows)==12 and [r['epoch'] for r in rows]==list(range(12))
        assert rows[-1]['test_coco_eval_bbox']==summary['native_stats']
        config=read(root/'args.json')['yaml_cfg']
    assert config['val_dataloader']['dataset']['transforms']=={'type':'Compose','ops':[
        {'type':'Resize','size':[640,640]},{'type':'ConvertPILImage','dtype':'float32','scale':True}]}
    assert not config.get('ram_cache',False)
    with np.load(root/'endpoint_epoch12/precision_recall.npz',allow_pickle=False) as z:
        precision,recall=z['precision'],z['recall']
        assert precision.shape==(10,101,10,4,3) and recall.shape==(10,10,4,3)
        assert np.isfinite(precision).all() and np.isfinite(recall).all()
        avg=lambda a:float(a[a>-1].mean())
        stats=[avg(precision[:,:,:,0,2]),avg(precision[0,:,:,0,2]),avg(precision[5,:,:,0,2]),
               *[avg(precision[:,:,:,i,2]) for i in (1,2,3)],
               *[avg(recall[:,:,0,i]) for i in (0,1,2)],*[avg(recall[:,:,i,2]) for i in (1,2,3)]]
    difference=max(abs(a-b) for a,b in zip(stats,summary['native_stats']));assert difference<1e-12
    invalid_sizes={name:{'queries':0,'images':0} for name in geometry.STAGES}
    def reader(iid):
        with np.load(root/f'stages_epoch12/image_{iid:06d}.npz',allow_pickle=False) as z:
            observed={n:z[n].copy() for n in z.files}
        assert observed['final_boxes'].shape==(300,4)
        for name in geometry.STAGES:
            count=int((observed[name][:,2:]<=0).any(1).sum())
            invalid_sizes[name]['queries']+=count
            invalid_sizes[name]['images']+=int(count>0)
        return observed
    values=geometry.collect(annotation,reader,flags)
    masks={'all':np.ones(len(values['annotation_id']),dtype=bool),
           'COCO_small':values['original_area']<=1024,
           **{f'class_{i}':values['category_id']==i for i in range(10)}}
    description={key:geometry.describe(values,mask) for key,mask in masks.items()}
    return summary,flags,values,description,{'source':str(root.resolve()),'seed':seed,
        'completion_sha256':sha(root/('completion.json' if arm=='baseline' else 'training_completion.json')),
        'endpoint_manifest_sha256':sha(root/'endpoint_epoch12/manifest.json'),
        'stage_manifest_sha256':sha(root/'stages_epoch12/manifest.json'),
        'native_stats_recomputed_from_saved_PR_max_abs':difference,
        'invalid_raw_size_queries_zero_geometric_support':invalid_sizes}


def analyze(baseline,annotation_path,output,candidate=None,candidate_arm='degraded'):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    assert sha(annotation_path)==VAL_SHA
    annotation=read(annotation_path);assert len(annotation['images'])==548
    left,lh,la,ld,lp=inspect_run(baseline,'baseline',annotation)
    report={'status':'EXISTING_BASELINE_STAGE_DESCRIPTION','seed':left['identity']['seed'],
        'stage_order':list(geometry.STAGES),'IoU_thresholds':geometry.THRESHOLDS.tolist(),
        'baseline':ld,'baseline_overall_percent':left['overall_percent'],'provenance':{'baseline':lp},
        'annotation_sha256':VAL_SHA,'new_model_forwards':0,'new_evaluator_calls':0,
        'native_TP_flags_recomputed':False,'claim_boundary':read(HERE/'protocol.json')['scope'],
        'causal_mechanism_proven':False,'automatic_promotion':False}
    arrays={'baseline':la}
    if candidate is not None:
        assert candidate_arm in ('fine','degraded')
        right,rh,ra,rd,rp=inspect_run(candidate,candidate_arm,annotation)
        assert left['identity']['seed']==right['identity']['seed']
        for key in ('common_init_sha256','baseline_checkpoint_sha256','baseline_first100_sha256',
                    'train_annotation_sha256','val_annotation_sha256','image_manifest_sha256'):
            assert left['identity'][key]==right['identity'][key],key
        same_result_evaluator(left,right)
        comparison=pair(left,right,lh,rh);saved=read(Path(candidate)/'fixed12_comparison.json')
        for key,value in comparison.items():assert saved[key]==value,key
        paired={}
        for name,mask in {'all':np.ones(len(la['annotation_id']),dtype=bool),
                          'COCO_small':la['original_area']<=1024,
                          **{f'class_{i}':la['category_id']==i for i in range(10)}}.items():
            paired[name]=geometry.pair_metrics(la,ra,mask)
            paired[name].pop('transition_annotation_ids')
        assert paired['all']['new_native_TP']==comparison['GT_new_TP_by_IoU']
        assert paired['all']['lost_native_TP']==comparison['GT_lost_TP_by_IoU']
        report.update(status='FIXED12_PAIRED_DESCRIPTIVE_STAGE_ANALYSIS',candidate=rd,
            candidate_overall_percent=right['overall_percent'],paired_changes=paired,
            fixed12_delta_AP_metrics_pp=comparison['delta_AP_metrics_pp'],
            fixed12_native_gate_passed=saved['native_gate_passed'])
        report['provenance']['candidate']=rp;arrays['candidate']=ra
    conflicts=sum(sum(desc['all']['native_TP_below_recomputed_final_support'])
                  for desc in [ld]+([rd] if candidate is not None else []))
    report['native_TP_geometry_disagreement_cells']=conflicts
    if conflicts:report['status']+='__NUMERICAL_REVIEW_REQUIRED'
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    output.mkdir(parents=True,exist_ok=False)
    for name,values in arrays.items():np.savez_compressed(output/f'{name}_per_GT.npz',**values)
    write(output/'report.json',report)
    write(output/'manifest.json',{'analysis_protocol_sha256':sha(HERE/'protocol.json'),
        'implementation':{p.name:sha(p) for p in HERE.glob('*.py')},
        'files':{p.name:sha(p) for p in output.iterdir() if p.is_file()}})
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',required=True)
    p.add_argument('--annotation',required=True);p.add_argument('--output',required=True);p.add_argument('--candidate')
    a=p.parse_args();r=analyze(a.baseline,a.annotation,a.output,a.candidate)
    print(json.dumps({'status':r['status'],'seed':r['seed'],'GT':r['baseline']['all']['GT_count'],
        'small_GT':r['baseline']['COCO_small']['GT_count'],'geometry_TP_conflicts':r['native_TP_geometry_disagreement_cells'],
        'new_model_forwards':0,'new_evaluator_calls':0}))
if __name__=='__main__':main()
