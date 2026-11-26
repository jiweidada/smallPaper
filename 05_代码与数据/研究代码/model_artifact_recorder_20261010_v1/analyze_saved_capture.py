"""Verify downloaded artifacts and generate same-GT/stage diagnostic tables."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def box_iou(pred,gt):
    lo=pred[:,:2]-pred[:,2:]/2; hi=pred[:,:2]+pred[:,2:]/2
    intersection=np.maximum(np.minimum(hi[:,None],gt[None,:,2:])-np.maximum(lo[:,None],gt[None,:,:2]),0).prod(-1)
    gt_area=(gt[:,2:]-gt[:,:2]).prod(-1)
    return intersection/np.maximum(pred[:,2:].prod(-1)[:,None]+gt_area[None]-intersection,1e-30)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--capture',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();root=Path(args.capture);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    manifest=read(root/'manifest.json')
    assert all(sha(root/name)==digest for name,digest in manifest['files'].items())
    completion=read(root/'completion.json')
    assert completion['status']=='PASS_EXISTING_CHECKPOINT_RECORDER_PREFLIGHT'
    assert completion['training_epochs_added']==completion['optimizer_steps']==completion['extra_prediction_head_calls']==0
    assert completion['model_buffers_and_parameters_unchanged'] and completion['native_sources_unchanged']
    rows=[];image_results=[]
    for image_result in completion['results']:
        folder=root/f"image_{image_result['image_id']:06d}"
        assert image_result['plain_recorded_output_equal'] and image_result['RNG_unchanged']
        trace=read(folder/'trace.json');gtfile=read(folder/'GT_original.json')
        with np.load(folder/'tensors.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
        assert len(data)==trace['tensor_count'] and sum(v.nbytes for v in data.values())==trace['tensor_bytes_uncompressed']
        meta=trace['metadata'];wh=np.array(meta['original_wh'])
        gtrows=gtfile['annotations'];gt=np.array([r['bbox'] for r in gtrows],float).reshape(-1,4)
        gt[:,2:]+=gt[:,:2];gt/=np.tile(wh,2)
        categories=np.array([r['category_id'] for r in gtrows],int)
        stages={
            'selected_encoder':data['decoder.decoder.layers.0.native_reference_input'][0,:,0],
            'pre_after_layer0':data['decoder.decoder@0.value.4'][0],
            **{f'FDR{i}':data[f'analysis.FDR{i}.boxes_from_native_integral'][0] for i in range(3)}
        }
        assert np.array_equal(stages['FDR2'],data['final.pred_boxes'][0])
        class_logits={'selected_encoder':data['native_topk.selected_logits'][0],'pre_after_layer0':data['decoder.decoder@0.value.5'][0],'FDR2':data['final.pred_logits'][0]}
        ious={name:box_iou(box,gt) for name,box in stages.items()}
        final_query=ious['FDR2'].argmax(0) if len(gt) else np.array([],int)
        ownership=collections.Counter(final_query.tolist())
        token_indices=data['native_topk.indices'][0]
        shapes=data['decoder.decoder.layers.0.native_spatial_shapes']
        sampling=[]
        for i in range(3):
            prefixes=[k[:-len('.actual_weights')] for k in data if k.startswith(f'decoder.decoder.layers.{i}.sampling_') and k.endswith('.actual_weights')]
            assert len(prefixes)==1
            prefix=prefixes[0];weights=data[prefix+'.actual_weights'];loc=data[prefix+'.actual_locations']
            assert weights.shape==loc.shape[:-1]
            assert np.isfinite(weights).all() and np.isfinite(loc).all()
            normalization_error=float(np.abs(weights.sum(-1)-1).max())
            assert normalization_error<1e-5 and (weights>=0).all()
            sampling.append(dict(layer=i,path=prefix,max_softmax_sum_error=normalization_error,queries=weights.shape[1],locations_shape=list(loc.shape)))
        for j,g in enumerate(gtrows):
            q=int(final_query[j]);token=int(token_indices[q]);prefix=0;level=None;yx=None
            for l,(h,w) in enumerate(shapes):
                count=int(h*w)
                if prefix<=token<prefix+count:
                    level=l;yx=[int((token-prefix)//w),int((token-prefix)%w)];break
                prefix+=count
            assert level is not None
            row=dict(image_id=meta['image_id'],annotation_id=g['id'],category_id=g['category_id'],original_shortside=min(g['bbox'][2:]),input_shortside=min(g['bbox'][2]*640/wh[0],g['bbox'][3]*640/wh[1]),query_chosen_by_final_best_geometry=q,query_shared_by_final_geometry_GT=ownership[q],encoder_token=token,encoder_level_index=level,encoder_stride=[8,16,32][level],encoder_grid_y=yx[0],encoder_grid_x=yx[1])
            for name,v in ious.items():
                best=int(v[:,j].argmax())
                row[name+'_best_geometric_iou']=float(v[best,j])
                row[name+'_best_query']=best
                row[name+'_fixed_query_iou']=float(v[q,j])
                row[name+'_fixed_query_pred_class']=int(class_logits[name][q].argmax()) if name in class_logits else None
                row[name+'_same_class_geometric_support_0_5']=bool((v[class_logits[name].argmax(-1)==categories[j],j]>=.5).any()) if name in class_logits else None
            rows.append(row)
        image_results.append(dict(image_id=meta['image_id'],GT_count=len(gtrows),tensor_count=len(data),called_modules=len(trace['called_module_counts']),uncompressed_bytes=trace['tensor_bytes_uncompressed'],sampling=sampling,geometry_coverage={name:{str(t):int((v.max(0)>=t).sum()) if len(gt) else 0 for t in (.5,.75)} for name,v in ious.items()},not_executed_module_names=sorted(set(trace['module_inventory'])-set(trace['called_module_counts']))))
    if rows:
        with (out/'per_GT_stage_trace.csv').open('x',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(status='PASS_DOWNLOADED_HASHES_AND_SAME_GT_STAGE_RECOMPUTATION',verified_manifest_files=len(manifest['files']),images=image_results,GT_rows=len(rows),training_epochs_added=0,scientific_root_cause_identified=False,training_history_recorded=False,causal_effect_estimated=False,limits='Whole selected-query geometric coverage, not AP or one-to-one detection matching. Fixed-query identity chosen by final best geometry, not Hungarian or historical training ownership; competing GT may select the same query and are flagged. No classification outputs are invented for unexecuted intermediate heads. Every named-module output has statistics but exact tensors are saved only at structural boundaries. Tiny fixed engineering sample is not a representative validation study.')
    (out/'verification_and_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'manifest.json').write_text(json.dumps({'source_manifest_sha256':sha(root/'manifest.json'),'code_sha256':sha(__file__),'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}},indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('images','limits')}))

if __name__=='__main__':main()
