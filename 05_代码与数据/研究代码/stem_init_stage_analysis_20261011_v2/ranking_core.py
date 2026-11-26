"""Offline final-geometry/ranking decomposition, with explicit score/tie uncertainty."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import geometry_core as geometry


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)


def analyze(packet,arm,protocol_path,output,postprocessor_source,annotation_path):
    packet,protocol_path,output=map(Path,(packet,protocol_path,output));assert not output.exists()
    p=read(protocol_path)
    assert p['schema']=='sod_final_selection_v1' and sha(__file__)==p['analysis_code_sha256']
    assert sha(geometry.__file__)==p['geometry_code_sha256']
    assert sha(postprocessor_source)==p['postprocessor_source_sha256']
    folder=packet/'baseline_eval' if arm=='baseline' else packet/'run'
    ep=folder/'endpoint_epoch12';em=read(ep/'manifest.json')
    for name,h in em['files'].items():assert sha(ep/name)==h
    summary=read(ep/'summary.json');identity=summary['identity'];assert identity['arm']==arm
    assert summary['epoch_zero_based']==11 and summary['mode']=='native_runtime'
    assert summary['protocol']['max_dets']==[1,10,100] and summary['protocol']['category_ids']==list(range(10))
    apath=Path(annotation_path);assert sha(apath)==identity['val_annotation_sha256']
    annotation=read(apath);images={int(x['id']):x for x in annotation['images']};grouped={i:[] for i in images}
    for a in annotation['annotations']:grouped[int(a['image_id'])].append(a)
    with np.load(ep/'GT_hits.npz',allow_pickle=False) as z:flags={k:z[k].copy() for k in z.files}
    index={int(g):i for i,g in enumerate(flags['annotation_id'])}
    assert len(index)==38759 and flags['hits'].shape==(38759,10)
    with gzip.open(ep/'matching_all_area.json.gz','rt',encoding='utf8') as f:matching=json.load(f)
    native_scores={}
    for row in matching:
        cell=(int(row['image_id']),int(row['category_id']));assert cell not in native_scores
        assert row['maxDet']==100 and row['aRng']==[0,1e10]
        native_scores[cell]=np.array(row['dtScores'],dtype=np.float64)
    sm=read(folder/'stages_epoch12/manifest.json');stage_summary=read(folder/'stages_epoch12/summary.json')
    assert stage_summary['identity']==identity and stage_summary['image_ids']==list(range(548))
    assert sm['trace_code_sha256']==p['stage_trace_sha256']
    margin=p['logit_boundary_margin'];floor=np.log(p['score_threshold']/(1-p['score_threshold']))
    records=[];diagnostics={};score_mismatch_images=[];native_conflicts=0
    for iid,im in sorted(images.items()):
        path=folder/f'stages_epoch12/image_{iid:06d}.npz';assert sha(path)==sm['files'][path.name]
        with np.load(path,allow_pickle=False) as z:
            boxes=z['final_boxes'].copy();logits=z['final_logits'].astype(np.float64);hw=z['original_hw'].copy()
        assert boxes.shape==(300,4) and logits.shape==(300,10) and np.isfinite(logits).all()
        assert hw.tolist()==[im['height'],im['width']]
        flat=logits.reshape(-1);order=np.argsort(-flat,kind='stable');chosen=order[:300]
        global_cut=float(flat[order[299]])
        probability=(1/(1+np.exp(-logits))).astype(np.float32).astype(np.float64)
        global_mask=np.zeros(3000,dtype=bool);global_mask[chosen]=True;global_mask=global_mask.reshape(300,10)
        global_mask &= probability>p['score_threshold']
        final_mask=np.zeros_like(global_mask);ambiguous=abs(flat[order[299]]-flat[order[300]])<=2*margin
        ambiguous |= bool((np.abs(flat-floor)<=margin).any())
        cut_probability=1/(1+np.exp(-global_cut))
        ambiguous |= bool(cut_probability>1-1e-4 or cut_probability<1e-30)
        max_error=0.;mismatch=[];class_cuts=[]
        for cls in range(10):
            cls_order=np.argsort(-logits[:,cls],kind='stable');class_cuts.append(float(logits[cls_order[99],cls]))
            selected=cls_order[global_mask[cls_order,cls]][:100];final_mask[selected,cls]=True
            if int(global_mask[:,cls].sum())>100:
                ambiguous |= abs(logits[cls_order[99],cls]-logits[cls_order[100],cls])<=2*margin
            predicted=probability[selected,cls];actual=native_scores.get((iid,cls),np.empty(0))
            if len(predicted)!=len(actual):mismatch.append(dict(category=cls,predicted_count=len(predicted),native_count=len(actual)))
            elif len(actual):
                error=float(np.max(np.abs(predicted-actual)));max_error=max(max_error,error)
                if error>p['native_score_abs_tolerance']:mismatch.append(dict(category=cls,max_abs_score_error=error))
        if mismatch:score_mismatch_images.append(iid);ambiguous=True
        diagnostics[str(iid)]=dict(global_logit_cutoff=global_cut,class100_logit_cutoffs=class_cuts,
            native_score_list_max_abs_error=max_error,native_score_list_mismatches=mismatch,ranking_ambiguous=bool(ambiguous))
        entries=sorted(grouped[iid],key=lambda a:int(a['id']));gt=np.array([a['bbox'] for a in entries],dtype=np.float64).reshape(-1,4)
        valid=(boxes[:,2:]>0).all(1);ious=np.zeros((len(gt),300))
        ious[:,valid]=geometry.iou_matrix(gt,geometry.pixel_xywh(boxes[valid],hw))
        for j,a in enumerate(entries):
            cls=int(a['category_id']);g=int(a['id']);frow=index[g]
            assert int(flags['category_id'][frow])==cls
            result=dict(annotation_id=g,image_id=iid,category_id=cls,original_area=float(a['area']),
                input_shortside=float(min(a['bbox'][2]*640/im['width'],a['bbox'][3]*640/im['height'])),
                ranking_ambiguous=bool(ambiguous))
            for threshold,ti,tag in ((.5,0,'50'),(.75,5,'75')):
                support=ious[j]>=threshold;global_ok=bool((support&global_mask[:,cls]).any());retained=bool((support&final_mask[:,cls]).any())
                native=bool(flags['hits'][frow,ti]);geometric=bool(support.any())
                conflict=native and (not geometric or (not ambiguous and not retained));native_conflicts+=int(conflict)
                if native:cell='native_TP'
                elif not geometric:cell='no_final_geometric_support'
                elif ambiguous:cell='uncertain_ranking_reconstruction'
                elif not global_ok:cell='no_true_class_support_after_top300_and_score_threshold'
                elif not retained:cell='true_class_support_removed_by_class100'
                else:cell='true_class_retained_geometry_without_native_TP'
                result.update({f'cell_{tag}':cell,f'geometry_{tag}':geometric,f'global_true_class_{tag}':global_ok,
                               f'retained_true_class_{tag}':retained,f'native_TP_{tag}':native,f'proxy_conflict_{tag}':bool(conflict)})
            records.append(result)
    records.sort(key=lambda x:x['annotation_id']);assert len(records)==38759
    data={k:np.array([r[k] for r in records]) for k in records[0]}
    masks=dict(all=np.ones(len(records),dtype=bool),COCO_small=data['original_area']<=1024,
               input_under4=data['input_shortside']<4,input_4_to_under8=(data['input_shortside']>=4)&(data['input_shortside']<8))
    assert masks['COCO_small'].sum()==26586 and masks['input_under4'].sum()==5096
    counts={}
    for name,mask in masks.items():
        counts[name]={}
        for tag in ('50','75'):
            cells,amount=np.unique(data[f'cell_{tag}'][mask],return_counts=True)
            values={str(k):int(v) for k,v in zip(cells,amount)};assert sum(values.values())==int(mask.sum())
            counts[name][tag]=dict(GT=int(mask.sum()),cells=values)
    output.mkdir();np.savez_compressed(output/'per_GT.npz',**data)
    write(output/'image_score_checks.json',diagnostics)
    report=dict(status='OFFLINE_RANK_PROXY_WITH_NATIVE_SCORE_CHECKS' if not score_mismatch_images and not native_conflicts else 'RANK_PROXY_REVIEW_REQUIRED',
        identity=identity,groups=counts,ambiguous_images=sum(x['ranking_ambiguous'] for x in diagnostics.values()),
        score_mismatch_images=score_mismatch_images,native_TP_proxy_conflicts=native_conflicts,
        native_score_list_max_abs_error=max(x['native_score_list_max_abs_error'] for x in diagnostics.values()),
        protocol_sha256=sha(protocol_path),endpoint_manifest_sha256=sha(ep/'manifest.json'),stage_manifest_sha256=sha(folder/'stages_epoch12/manifest.json'),
        new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,scope=p['scope'])
    write(output/'report.json',report);write(output/'manifest.json',dict(files={q.name:sha(q) for q in output.iterdir() if q.is_file()}))
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    return dict(status=report['status'],seed=identity['seed'],arm=arm,ambiguous_images=report['ambiguous_images'],
                native_TP_proxy_conflicts=native_conflicts,groups=counts)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('packet','arm','protocol','output','postprocessor_source','annotation_path'):p.add_argument('--'+n,required=True)
    a=p.parse_args();print(json.dumps(analyze(a.packet,a.arm,a.protocol,a.output,a.postprocessor_source,a.annotation_path)),flush=True)

