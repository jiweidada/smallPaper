"""Join future v5 stage geometry to captured native TP flags. No model/evaluator calls."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
STAGES = ('encoder_initial_boxes', 'pre_after_first_layer_boxes', 'final_boxes')
THRESHOLDS = np.linspace(.5, .95, 10)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def pixel_xywh(boxes, original_hw):
    """Match bound float32 no-padding postprocessing coordinates, without clipping."""
    boxes = np.asarray(boxes)
    hw = np.asarray(original_hw)
    if (boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all()
            or np.any(boxes[:, 2:] < 0) or hw.shape != (2,) or not np.isfinite(hw).all()
            or np.any(hw <= 0)):
        raise ValueError('Malformed boxes or original height/width')
    b = boxes.astype(np.float32)
    xyxy = np.concatenate((b[:, :2] - b[:, 2:]*np.float32(.5),
                           b[:, :2] + b[:, 2:]*np.float32(.5)), axis=1)
    xyxy *= np.array([hw[1], hw[0], hw[1], hw[0]], dtype=np.float32)
    xyxy[:, 2:] -= xyxy[:, :2]  # Native prepare_for_coco_detection xyxy -> xywh.
    if not np.isfinite(xyxy).all():
        raise ValueError('Non-finite restored coordinates')
    return xyxy.astype(np.float64)


def iou_matrix(gt_xywh, candidate_xywh):
    """Geometry only, one GT may have support from the same query as another GT."""
    gt = np.asarray(gt_xywh, dtype=np.float64).reshape(-1, 4)
    dt = np.asarray(candidate_xywh, dtype=np.float64).reshape(-1, 4)
    if not np.isfinite(gt).all() or not np.isfinite(dt).all() or np.any(gt[:,2:] < 0) or np.any(dt[:,2:] < 0):
        raise ValueError('Invalid xywh geometry')
    wh = np.maximum(0, np.minimum(gt[:,None,:2]+gt[:,None,2:],dt[None,:,:2]+dt[None,:,2:])
                    - np.maximum(gt[:,None,:2],dt[None,:,:2]))
    intersection = wh[:,:,0]*wh[:,:,1]
    union = (gt[:,2]*gt[:,3])[:,None]+(dt[:,2]*dt[:,3])[None,:]-intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union>0)


def image_metrics(image, annotations, observed):
    expected_hw = [image['height'], image['width']]
    if np.asarray(observed['original_hw']).tolist() != expected_hw:
        raise ValueError('Stored original_hw differs from annotation height/width')
    arrays = [np.asarray(observed[name]) for name in STAGES]
    count = len(arrays[0])
    if count == 0 or any(a.shape != (count,4) for a in arrays):
        raise ValueError('Stage query counts differ or are empty')
    logits = [np.asarray(observed[name]) for name in ('pre_after_first_layer_logits','final_logits')]
    if any(a.shape != (count,10) or not np.isfinite(a).all() for a in logits):
        raise ValueError('Malformed stage class logits')
    gt = np.asarray([a['bbox'] for a in annotations], dtype=np.float64).reshape(-1,4)
    classes = np.asarray([a['category_id'] for a in annotations], dtype=np.int64)
    if np.any((classes<0)|(classes>9)):
        raise ValueError('Unexpected category identifier')
    matrices = [iou_matrix(gt,pixel_xywh(a,expected_hw)) for a in arrays]
    best = np.stack([m.argmax(axis=1) for m in matrices], axis=1)
    rows = np.arange(len(gt))
    maxima = np.stack([m[rows,best[:,s]] for s,m in enumerate(matrices)], axis=1)
    final_logits = logits[1][best[:,2]]
    target_logit = final_logits[rows,classes]
    other = final_logits.copy()
    other[rows,classes] = -np.inf
    return {'max_iou':maxima, 'best_query_within_run':best,
            'initial_best_at_pre':matrices[1][rows,best[:,0]],
            'initial_best_at_final':matrices[2][rows,best[:,0]],
            'final_best_at_initial':matrices[0][rows,best[:,2]],
            'pre_best_class_argmax':logits[0][best[:,1]].argmax(1),
            'final_best_class_argmax':final_logits.argmax(1),
            'final_best_target_logit_margin':target_logit-other.max(1)}


def collect(annotation, image_reader, flags):
    """Join by annotation ID, never by query row or incidental GT array order."""
    images = {int(im['id']):im for im in annotation['images']}
    annotations = {int(a['id']):a for a in annotation['annotations']}
    if len(images)!=len(annotation['images']) or len(annotations)!=len(annotation['annotations']):
        raise ValueError('Duplicate image or annotation identities')
    ids = np.array(sorted(annotations),dtype=np.int64)
    flag_ids = np.asarray(flags['annotation_id'])
    flag_classes = np.asarray(flags['category_id'])
    hits = np.asarray(flags['hits'])
    if (flag_ids.ndim!=1 or len(set(flag_ids.tolist()))!=len(flag_ids)
            or set(flag_ids.tolist())!=set(ids.tolist()) or flag_classes.shape!=flag_ids.shape
            or hits.shape!=(len(ids),10) or hits.dtype!=np.dtype(bool)):
        raise ValueError('Incomplete, duplicated, or malformed native GT flags')
    flag_row = {int(g):i for i,g in enumerate(flag_ids)}
    order = np.array([flag_row[int(g)] for g in ids],dtype=np.int64)
    classes = np.array([annotations[int(g)]['category_id'] for g in ids],dtype=np.int64)
    if not np.array_equal(flag_classes[order],classes):
        raise ValueError('Native TP flag class differs from the annotation')
    if any(int(a['image_id']) not in images or a.get('iscrowd',0) or a.get('ignore',0) for a in annotations.values()):
        raise ValueError('Unknown image or ignored/crowd GT outside the bound dataset')
    grouped = {iid:[] for iid in images}
    for g in ids:
        a=annotations[int(g)]
        grouped[int(a['image_id'])].append(a)
    row_of = {int(g):i for i,g in enumerate(ids)}
    result = {'annotation_id':ids,'category_id':classes,
              'image_id':np.array([annotations[int(g)]['image_id'] for g in ids],dtype=np.int64),
              'original_area':np.array([annotations[int(g)]['area'] for g in ids],dtype=np.float64),
              'native_TP':hits[order].copy()}
    if not np.isfinite(result['original_area']).all() or np.any(result['original_area']<0):
        raise ValueError('Invalid original annotation area')
    for iid,im in sorted(images.items()):
        metrics = image_metrics(im,grouped[iid],image_reader(iid))
        rows = np.array([row_of[int(a['id'])] for a in grouped[iid]],dtype=np.int64)
        for key,values in metrics.items():
            if key not in result:
                result[key]=np.empty((len(ids),*values.shape[1:]),dtype=values.dtype)
            result[key][rows]=values
    return result


def describe(values, mask=None):
    mask = np.ones(len(values['annotation_id']),dtype=bool) if mask is None else mask
    m = values['max_iou'][mask]
    tp = values['native_TP'][mask]
    count = len(m)
    support = m[:,:,None] >= THRESHOLDS
    # This is an audit flag, not a replacement definition of native TP.
    inconsistent = tp & (m[:,2,None] < THRESHOLDS-1e-12)
    return {'GT_count':count,
        'mean_max_IoU_by_stage':m.mean(0).tolist() if count else [None]*3,
        'geometric_support_counts_by_stage_and_threshold':support.sum(0).tolist(),
        'native_TP_by_threshold':tp.sum(0).tolist(),
        'initial_support_lost_at_final':(support[:,0]&~support[:,2]).sum(0).tolist(),
        'initial_best_query_loses_but_other_query_retains_support':(
            support[:,0] & (values['initial_best_at_final'][mask,None]<THRESHOLDS)
            & support[:,2]).sum(0).tolist(),
        'final_support_without_native_TP':(support[:,2]&~tp).sum(0).tolist(),
        'native_TP_below_recomputed_final_support':inconsistent.sum(0).tolist(),
        'geometry_values_within_1e_minus_6_of_threshold':(
            np.abs(m[:,:,None]-THRESHOLDS)<=1e-6).sum(0).tolist(),
        'final_best_geometry_query_argmax_correct':int((
            values['final_best_class_argmax'][mask]==values['category_id'][mask]).sum())}


def pair_metrics(control, candidate, mask=None):
    for key in ('annotation_id','image_id','category_id','original_area'):
        if not np.array_equal(control[key],candidate[key]):
            raise ValueError('Stage comparison populations differ: '+key)
    mask = np.ones(len(control['annotation_id']),dtype=bool) if mask is None else mask
    a,b = control['max_iou'][mask],candidate['max_iou'][mask]
    ta,tb = control['native_TP'][mask],candidate['native_TP'][mask]
    sa,sb = a[:,:,None]>=THRESHOLDS,b[:,:,None]>=THRESHOLDS
    new_tp,lost_tp = ~ta&tb,ta&~tb
    early_gain,early_loss = ~sa[:,0]&sb[:,0],sa[:,0]&~sb[:,0]
    transition_ids = {}
    ids = control['annotation_id'][mask]
    for t,threshold in enumerate(THRESHOLDS):
        transition_ids[format(threshold,'.2f')]={
            'new_TP':ids[new_tp[:,t]].tolist(), 'lost_TP':ids[lost_tp[:,t]].tolist(),
            'initial_support_gained_but_still_not_TP':ids[(early_gain&~tb)[:,t]].tolist(),
            'new_TP_with_initial_support_gain':ids[(new_tp&early_gain)[:,t]].tolist()}
    return {'GT_count':len(a),
        'mean_max_IoU_delta_by_stage':(b-a).mean(0).tolist() if len(a) else [None]*3,
        'new_geometric_support_by_stage_and_threshold':((~sa)&sb).sum(0).tolist(),
        'lost_geometric_support_by_stage_and_threshold':(sa&~sb).sum(0).tolist(),
        'new_native_TP':new_tp.sum(0).tolist(),'lost_native_TP':lost_tp.sum(0).tolist(),
        'new_TP_with_initial_support_gain':(new_tp&early_gain).sum(0).tolist(),
        'new_TP_without_initial_support_gain':(new_tp&~early_gain).sum(0).tolist(),
        'lost_TP_with_initial_support_loss':(lost_tp&early_loss).sum(0).tolist(),
        'initial_support_gained_but_still_not_TP':(early_gain&~tb).sum(0).tolist(),
        'transition_annotation_ids':transition_ids}


def load_bound_experiment(root):
    root=Path(root).resolve()
    binding=read(HERE/'experiment_binding.json')
    if sha(root/binding['delivery_receipt_name'])!=binding['delivery_receipt_sha256']:
        raise ValueError('Analysis requires the bound v5 experiment delivery')
    for name,expected in binding['implementation'].items():
        if sha(root/name)!=expected:
            raise ValueError('Bound experiment implementation changed: '+name)
    for name in ('runtime','results','stage_trace','evaluator_binding'):
        if name in sys.modules and Path(sys.modules[name].__file__).resolve().parent!=root:
            raise RuntimeError('Use a fresh interpreter; another experiment module is loaded')
    sys.path.insert(0,str(root))
    return importlib.import_module('results')


def analyze(experiment_root, endpoints, annotation_path, output):
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    experiment_root=Path(experiment_root).resolve()
    endpoint_api=load_bound_experiment(experiment_root)
    # Existing strict gate checks all six native endpoints, EMA epoch 132,
    # exact evaluator package, checkpoints, stage manifests and implementation.
    formal=endpoint_api.compare(endpoints)
    annotation=read(annotation_path)
    annotation_sha=sha(annotation_path)
    arrays, run_summaries, provenance={}, {}, {}
    for path in endpoints:
        path=Path(path)
        summary,flags,manifest=endpoint_api.load_endpoint(path)
        if summary['identity']['val_annotation_sha256']!=annotation_sha:
            raise ValueError('Stage analysis annotation differs from native evaluation')
        if not np.array_equal(np.array(summary['protocol']['iou_thresholds']),THRESHOLDS):
            raise ValueError('IoU threshold grid differs')
        key=f"{summary['identity']['arm']}_seed{summary['identity']['seed']}"
        plan=read(experiment_root/'configs'/f'{key}.json')['native_baseline_config']
        ops=plan['val_dataloader']['dataset']['transforms']
        if (plan.get('ram_cache',False) or ops!={'type':'Compose','ops':[
                {'type':'Resize','size':[640,640]}, {'type':'ConvertPILImage','dtype':'float32','scale':True}]}):
            raise ValueError('Geometry reader supports only the bound direct-resize validation path')
        completion=read(path/'training_completion.json')
        trace=Path(completion['stage_trace_manifest']).parent
        def image_reader(iid):
            with np.load(trace/f'image_{iid:06d}.npz',allow_pickle=False) as data:
                result={name:data[name].copy() for name in data.files}
            if result['final_boxes'].shape!=(300,4):
                raise ValueError('Expected exactly 300 native queries')
            return result
        value=collect(annotation,image_reader,flags)
        arrays[key]=value
        run_summaries[key]={'all':describe(value),
            'by_class':{str(c):describe(value,value['category_id']==c) for c in range(10)}}
        provenance[key]={'endpoint':str(path.resolve()),'endpoint_manifest_sha256':sha(path/'manifest.json'),
                         'stage_manifest':str(trace/'manifest.json'),
                         'stage_manifest_sha256':sha(trace/'manifest.json'),
                         'training_completion_sha256':sha(path/'training_completion.json')}
    pairs={}
    for seed in (1,2):
        candidate=arrays[f'fine_residual_seed{seed}']
        for arm in ('baseline','fused_residual'):
            control=arrays[f'{arm}_seed{seed}']
            name=f'fine_residual_minus_{arm}_seed{seed}'
            whole=pair_metrics(control,candidate)
            native=formal['comparisons'][name]
            if (whole['new_native_TP']!=native['GT_new_TP_by_IoU']
                    or whole['lost_native_TP']!=native['GT_lost_TP_by_IoU']):
                raise ValueError('Per-target join disagrees with native endpoint comparison')
            pairs[name]={'all':whole,
                'by_class':{str(c):pair_metrics(control,candidate,control['category_id']==c) for c in range(10)}}
    conflicts=sum(sum(s['all']['native_TP_below_recomputed_final_support']) for s in run_summaries.values())
    output.mkdir(parents=True,exist_ok=False)
    for name,value in arrays.items():
        np.savez_compressed(output/f'{name}_per_GT.npz',**value)
    report={'status':'STAGE_GEOMETRY_REQUIRES_NUMERICAL_REVIEW' if conflicts else 'STAGE_GEOMETRY_JOINED_TO_NATIVE_ENDPOINTS',
        'stage_order':list(STAGES),'IoU_thresholds':THRESHOLDS.tolist(),
        'native_endpoint_comparison':formal,'runs':run_summaries,'paired_stage_changes':pairs,
        'provenance':provenance,'annotation_sha256':annotation_sha,
        'geometry_computation':'float32 direct-resize postprocessing coordinate operations then float64 bbox IoU',
        'native_TP_geometry_disagreement_cells':conflicts,
        'claim_boundary':'Descriptive paired changes, not AP attribution, query identity across runs, causal mediation, or independent validation',
        'native_TP_flags_recomputed':False,'model_calls':0,'evaluator_calls':0,
        'independent_reproduction_verified':False,'causal_mechanism_proven':False,
        'automatic_adoption_decision':False}
    write(output/'report.json',report)
    write(output/'manifest.json',{'analysis_code_sha256':sha(__file__),
          'experiment_binding_sha256':sha(HERE/'experiment_binding.json'),
          'files':{p.name:sha(p) for p in output.iterdir() if p.is_file()}})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root',required=True)
    parser.add_argument('--endpoints',nargs=6,required=True)
    parser.add_argument('--annotation',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    report=analyze(args.experiment_root,args.endpoints,args.annotation,args.output)
    print(json.dumps({'status':report['status'],'native_TP_geometry_disagreement_cells':
                     report['native_TP_geometry_disagreement_cells']},ensure_ascii=False))


if __name__=='__main__':
    main()
