"""Fixed-final-head arithmetic only; not a full modified detector or AP test."""
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
DOWNLOAD=ROOT/'outputs/candidate_all_module_capture_20261010_v1/seed2_download_v1'
OUT=ROOT/'outputs/sod_fdr_tail_support_20261010_v2/frozen_final_head_damage_seed2_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def iou(a,b):
    wh=np.maximum(np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]),0.)
    overlap=wh.prod(-1)
    aa=np.maximum(a[:,2:]-a[:,:2],0.).prod(-1)
    bb=np.maximum(b[:,2:]-b[:,:2],0.).prod(-1)
    return overlap/np.maximum(aa[:,None]+bb[None,:]-overlap,1e-30)


def xyxy(box):
    return np.concatenate((box[:,:2]-box[:,2:]*.5,box[:,:2]+box[:,2:]*.5),-1)


def from_distance(ref,distance):
    return np.stack((ref[:,0]-(distance[:,0]+2)*ref[:,2]/4,
                     ref[:,1]-(distance[:,1]+2)*ref[:,3]/4,
                     ref[:,0]+(distance[:,2]+2)*ref[:,2]/4,
                     ref[:,1]+(distance[:,3]+2)*ref[:,3]/4),-1)


def main():
    metadata=json.loads((DOWNLOAD/'module_candidate_FDR_subset_seed2_v1_metadata.json').read_text())
    archive=DOWNLOAD/'module_candidate_FDR_subset_seed2_v1.zip'
    assert archive.stat().st_size==metadata['bytes'] and sha(archive)==metadata['sha256']
    assert metadata['candidate_forwards_added']==metadata['baseline_forwards']==metadata['epochs_added']==0
    protocol=json.loads((ROOT/'outputs/sod_fdr_tail_support_20261010_v2/protocol.json').read_text())
    grid=np.array(protocol['old_grid'],dtype=np.float64)
    assert grid[0]==-4 and grid[-1]==4 and len(grid)==33
    assert not OUT.exists();OUT.mkdir()
    frozen=dict(schema='frozen_final_head_tail_delta_v1',factor=2.0,
        subset_sha256=metadata['sha256'],full_capture_manifest_sha256='38d9a79310ee73e75d4406b76f85a9017d2afb12e27507b0c0d724eecec3720d',
        probability_arithmetic='FP32 native residual addition, then NumPy FP64 stabilized softmax',
        old_integral_max_abs_tolerance=1e-5,old_boxes_max_abs_tolerance=1e-6,
        cohorts=['all_native_GT','actual_input_short_under4'],thresholds=[.5,.75],
        scale=4.0,old_final_distance_is_saved_native_Integral=True,
        prediction='d8 = saved_d4 + 4*(p_last-p_first), using the frozen final cumulative corner logits',
        later_layer_features_scores_and_attention_not_recomputed=True,
        GT_assisted_geometry_diagnostic_not_deployable=True,full_candidate_or_AP_claim=False,
        new_forward_calls=0,new_optimizer_updates=0,new_training_epochs=0)
    write(OUT/'protocol.json',frozen)
    rows=[];errors=[];tail=[];arrays={}
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert len(names)==len(set(names))
        assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
        for bi in (25,105):
            subset=json.loads(z.read(f'batch_{bi:03d}/subset.json'))
            assert subset['schema']=='actual_raw_FDR_head_subset_v1'
            assert subset['full_capture_manifest_sha256']==frozen['full_capture_manifest_sha256']
            assert subset['metadata']['checkpoint_sha256']=='8113eb35c4ca0519e9757a101327cb95e709ebd3609559f55c240e1a3a472269'
            def load(desc):
                assert desc['kind']=='tensor' and desc['original_dtype']=='torch.float32'
                raw=z.read(f'batch_{bi:03d}/'+desc['blob_file'])
                assert hashlib.sha256(raw).hexdigest()==desc['blob_sha256']
                value=np.load(io.BytesIO(raw),allow_pickle=False)
                assert list(value.shape)==desc['shape'] and str(value.dtype)==desc['storage_numpy_dtype']
                assert np.isfinite(value).all()
                return value
            events=subset['events']
            heads=[load(next(x for x in events if x['module']==f'decoder.dec_bbox_head.{j}')['output']) for j in range(3)]
            distances=[load(next(x for x in events if x['module']=='decoder.integral' and x['invocation']==j)['output']) for j in range(3)]
            cumulative=heads[0].copy()
            for j in range(3):
                if j:
                    cumulative=np.add(heads[j],cumulative,dtype=np.float32)
                values=cumulative.reshape(4,300,4,33).astype(np.float64)
                probabilities=np.exp(values-values.max(-1,keepdims=True));probabilities/=probabilities.sum(-1,keepdims=True)
                error=float(np.max(np.abs(probabilities@grid-distances[j])))
                assert error<=frozen['old_integral_max_abs_tolerance'],error
                errors.append(dict(batch=bi,layer=j,native_Integral_vs_FP64_probability_mean_max_abs_error=error))
            delta=4*(probabilities[:,:,:,-1]-probabilities[:,:,:,0])
            counter_distance=distances[-1].astype(np.float64)+delta
            tail.extend((probabilities[:,:,:,0]+probabilities[:,:,:,-1]).flatten().tolist())
            targets=subset['native_targets'];assert targets['kind']=='list' and len(targets['items'])==4
            for offset,iid in enumerate(subset['metadata']['image_ids']):
                target={entry['key']['value']:entry['value'] for entry in targets['items'][offset]['entries']}
                gt=load(target['boxes']).astype(np.float64)
                with np.load(io.BytesIO(z.read(f'stages/image_{iid:06d}.npz')),allow_pickle=False) as s:
                    ref=s['pre_after_first_layer_boxes'].astype(np.float64)
                    old=xyxy(s['final_boxes'].astype(np.float64))
                    reproduced=from_distance(ref,distances[-1][offset].astype(np.float64))
                    error=float(np.max(np.abs(reproduced-old)))
                    assert error<=frozen['old_boxes_max_abs_tolerance'],error
                new=from_distance(ref,counter_distance[offset])
                valid=(new[:,2]>new[:,0])&(new[:,3]>new[:,1])
                best_old=iou(xyxy(gt),old).max(1) if len(gt) else np.empty(0)
                best_new=iou(xyxy(gt),new[valid]).max(1) if len(gt) and valid.any() else np.zeros(len(gt))
                under4=gt[:,2:].min(1)*640<4
                for index in range(len(gt)):
                    rows.append(dict(image_id=iid,target_index=index,input_short_px=float(gt[index,2:].min()*640),
                        under4=bool(under4[index]),old_any_query_max_IoU=float(best_old[index]),
                        frozen_head_range8_any_query_max_IoU=float(best_new[index])))
                arrays[f'image_{iid:06d}_old_boxes_xyxy']=old
                arrays[f'image_{iid:06d}_frozen_head_range8_boxes_xyxy']=new
                arrays[f'image_{iid:06d}_native_targets_cxcywh']=gt
                errors.append(dict(image=iid,native_boxes_reconstruction_max_abs_error=error,
                    range8_nonpositive_wh_queries=int((~valid).sum())))
    summary={}
    for cohort in ('all_native_GT','actual_input_short_under4'):
        selected=rows if cohort=='all_native_GT' else [x for x in rows if x['under4']]
        old=np.array([x['old_any_query_max_IoU'] for x in selected]);new=np.array([x['frozen_head_range8_any_query_max_IoU'] for x in selected])
        summary[cohort]=dict(GT=len(selected),thresholds={str(t):dict(old=int((old>=t).sum()),
            counterfactual=int((new>=t).sum()),rescued=int(((old<t)&(new>=t)).sum()),
            damaged=int(((old>=t)&(new<t)).sum())) for t in frozen['thresholds']})
    np.savez_compressed(OUT/'counterfactual_arrays.npz',**arrays)
    with (OUT/'per_GT_geometry.csv').open('x',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(OUT/'report.json',dict(status='PASS_VERIFIED_SAVED_HEAD_ARITHMETIC_DIAGNOSTIC',seed=2,images=8,
        geometry=summary,numeric_validation=errors,mean_extreme_bin_probability_mass=float(np.mean(tail)),
        maximum_extreme_bin_probability_mass=float(np.max(tail)),**{k:v for k,v in frozen.items() if k not in ('schema','factor')}))
    write(OUT/'manifest.json',dict(files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},source_sha256=sha(__file__)))
    print(json.dumps(dict(status='SAVED_FINAL_HEAD_DIAGNOSTIC_COMPLETE',geometry=summary),ensure_ascii=False))


if __name__=='__main__':
    main()
