"""Reconstruct native cross-attention from verified raw module outputs only."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf8') as file:json.dump(value,file,ensure_ascii=False,indent=2,allow_nan=False)


def xyxy(box):return np.concatenate((box[...,:2]-box[...,2:]/2,box[...,:2]+box[...,2:]/2),-1)


def distance_box(ref,d):
    x1=ref[...,0]-(2+d[...,0])*(ref[...,2]/4);y1=ref[...,1]-(2+d[...,1])*(ref[...,3]/4)
    x2=ref[...,0]+(2+d[...,2])*(ref[...,2]/4);y2=ref[...,1]+(2+d[...,3])*(ref[...,3]/4)
    return np.stack(((x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1),-1)


def iou(gt,pred):
    wh=np.maximum(np.minimum(gt[:,None,2:],pred[None,:,2:])-np.maximum(gt[:,None,:2],pred[None,:,:2]),0)
    intersection=wh.prod(-1)
    ga=np.maximum(gt[:,2:]-gt[:,:2],0).prod(-1);pa=np.maximum(pred[:,2:]-pred[:,:2],0).prod(-1)
    return intersection/np.maximum(ga[:,None]+pa[None,:]-intersection,1e-30)


def bilinear(feature,locations):
    # Feature [B,256,H,W], locations [B,Q,8,N,2]; output [B,Q,8,N,32].
    b,c,h,w=feature.shape;q=locations.shape[1];n=locations.shape[-2]
    planes=feature.reshape(b,8,32,h,w).astype(np.float64)
    pixel=locations*np.array([w,h],dtype=np.float64)-.5
    lower=np.floor(pixel).astype(np.int64);frac=pixel-lower
    sampled=np.zeros((b,q,8,n,32),dtype=np.float64)
    for bi in range(b):
        for hi in range(8):
            for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
                ix=lower[bi,:,hi,:,0]+dx;iy=lower[bi,:,hi,:,1]+dy
                ok=(ix>=0)&(ix<w)&(iy>=0)&(iy<h)
                weight=(frac[bi,:,hi,:,0] if dx else 1-frac[bi,:,hi,:,0])*(frac[bi,:,hi,:,1] if dy else 1-frac[bi,:,hi,:,1])
                vals=planes[bi,hi][:,np.clip(iy,0,h-1),np.clip(ix,0,w-1)].transpose(1,2,0)
                sampled[bi,:,hi]+=vals*(weight*ok)[...,None]
    return sampled,pixel,lower


def summarize_query(a):
    return dict(queries=len(a['P3_attention_mass']),mean_P3_attention_mass=float(np.mean(a['P3_attention_mass'])),
        median_relative_value_RMS=float(np.median(a['P3_relative_value_RMS'])),
        median_all_head_stencils=float(np.median(a['P3_all_head_stencils'])),
        mean_same_stencil_head_fraction=float(np.mean(a['P3_same_stencil_head_fraction'])),
        mean_joint_collapse_head_fraction=float(np.mean(a['P3_joint_collapse_head_fraction'])),
        mean_ambiguous_head_fraction=float(np.mean(a['P3_boundary_ambiguous_head_fraction'])),
        majority_joint_collapse_queries=int(np.sum(a['P3_joint_collapse_head_fraction']>.5)))


def main(seed):
    p=read(HERE/'protocol.json');assert p['schema']=='saved_native_decoder_sampling_geometry_v2'
    assert sha(__file__)==p['script_sha256'] and seed in p['seeds']
    old=ROOT/f'outputs/scale_protection12_20261010_v1/terminal_collection/seed{seed}_target_protect_v1'
    download=ROOT/f'outputs/candidate_all_module_capture_20261010_v1/seed{seed}_download_v1'
    binding=p['sources'][str(seed)]
    assert sha(old/'verification_report_v1.json')==binding['target_report_sha256']
    verified=read(old/'verification_report_v1.json')
    assert verified['status']=='PASS_FIXED12_DOWNLOADED_EVIDENCE_AND_SAVED_WEIGHT_RELOAD'
    assert sha(download/'raw_verification_report_v1.json')==binding['raw_report_sha256']
    for name,h in p['native_source_sha256'].items():assert sha(old/'extracted/native'/name)==h,name
    archive=download/f'module_candidate_panel_seed{seed}_raw_v1.zip'
    assert archive.stat().st_size==binding['archive_bytes'] and sha(archive)==binding['archive_sha256']
    recipe=read(old/'extracted/prepared/plan.json')['model_recipe']['DFINETransformer']
    assert recipe['num_points']==p['points_per_level'] and recipe['num_layers']==3
    out=ROOT/f'outputs/decoder_sampling_geometry_20261011_v2/seed{seed}'
    assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);out.mkdir()
    rows=[];gt_rows=[];controls=[];derived={};consumed={}
    try:
        sm=read(old/'extracted/run/stages_epoch12/manifest.json')
        with zipfile.ZipFile(archive) as z:
            raw_manifest=z.read('capture/manifest.json');assert hashlib.sha256(raw_manifest).hexdigest()==binding['capture_manifest_sha256']
            manifest=json.loads(raw_manifest)
            for bi in p['native_batches']:
                prefix=f'capture/batch_{bi:03d}/';raw=z.read(prefix+'record.json')
                assert hashlib.sha256(raw).hexdigest()==manifest['files'][f'batch_{bi:03d}/record.json']['sha256']
                record=json.loads(raw);ids=record['metadata']['image_ids']
                assert record['metadata']['arm']=='target_protect' and record['metadata']['seed']==seed
                assert record['metadata']['checkpoint_sha256']==binding['checkpoint_sha256']
                def event(name,inv=0):
                    found=[e for e in record['invocation_events'] if e['module']==name and e['invocation']==inv]
                    assert len(found)==1,(name,inv);return found[0]['output']
                def load(desc):
                    assert desc['kind']=='tensor';raw=z.read(prefix+desc['blob_file'])
                    h=hashlib.sha256(raw).hexdigest();assert h==desc['blob_sha256']
                    consumed[prefix+desc['blob_file']]=h
                    value=np.load(io.BytesIO(raw),allow_pickle=False)
                    assert list(value.shape)==desc['shape'] and str(value.dtype)==desc['storage_numpy_dtype'] and np.isfinite(value).all()
                    return value
                features=[load(event(f'decoder.input_proj.{i}')) for i in range(3)]
                assert [list(f.shape[-2:]) for f in features]==p['spatial_shapes']
                distances=[load(event('decoder.integral',i)).astype(np.float64) for i in range(3)]
                native=[];targets=[];labels=[]
                for offset,iid in enumerate(ids):
                    sp=old/f'extracted/run/stages_epoch12/image_{iid:06d}.npz';assert sha(sp)==sm['files'][sp.name]
                    with np.load(sp,allow_pickle=False) as f:native.append({k:f[k].copy() for k in f.files})
                    entries=record['native_targets']['items'][offset]['entries'];descs={x['key']['value']:x['value'] for x in entries}
                    targets.append(load(descs['boxes']).astype(np.float64));labels.append(load(descs['labels']).astype(np.int64))
                    assert targets[-1].ndim==2 and targets[-1].shape[1]==4
                    assert (targets[-1][:,2:]>targets[-1][:,:2]).all() and (targets[-1]>=0).all() and (targets[-1]<=640.0001).all()
                ref0=np.stack([n['encoder_initial_boxes'] for n in native]).astype(np.float64)
                pre=np.stack([n['pre_after_first_layer_boxes'] for n in native]).astype(np.float64)
                refs=[ref0,distance_box(pre,distances[0]),distance_box(pre,distances[1])]
                final=np.stack([n['final_boxes'] for n in native]).astype(np.float64)
                reconstruction_error=float(np.max(np.abs(distance_box(pre,distances[2])-final)))
                assert reconstruction_error<=p['reconstruction_control']['final_box_abs_tolerance']
                layer_metrics=[]
                for layer in p['decoder_layers']:
                    raw_offsets=load(event(f'decoder.decoder.layers.{layer}.cross_attn.sampling_offsets')).astype(np.float64).reshape(4,300,8,12,2)
                    attn_logits=load(event(f'decoder.decoder.layers.{layer}.cross_attn.attention_weights')).astype(np.float64).reshape(4,300,8,12)
                    weights=np.exp(attn_logits-attn_logits.max(-1,keepdims=True));weights/=weights.sum(-1,keepdims=True)
                    scales=np.repeat(1/np.array(p['points_per_level'],dtype=np.float64),p['points_per_level'])
                    locations=refs[layer][:,:,None,None,:2]+raw_offsets*scales[None,None,None,:,None]*refs[layer][:,:,None,None,2:]*p['offset_scale']
                    accumulated=np.zeros((4,300,8,32),dtype=np.float64);metrics={}
                    start=0
                    for level,feature in enumerate(features):
                        stop=start+p['points_per_level'][level]
                        values,pixel,lower=bilinear(feature,locations[:,:,:,start:stop])
                        w=weights[:,:,:,start:stop];mass=w.sum(-1)
                        accumulated+=(values*w[...,None]).sum(-2)
                        local=w/np.maximum(mass[...,None],1e-30)
                        mean=(values*local[...,None]).sum(-2)
                        energy=(np.square(values).sum(-1)*local).sum(-1)
                        variance=(np.square(values-mean[...,None,:]).sum(-1)*local).sum(-1)
                        relative=np.sqrt(variance/np.maximum(energy,1e-30))
                        same=(lower==lower[:,:,:,:1]).all((-1,-2))
                        h,width=feature.shape[-2:]
                        full=(lower[...,0]>=0)&(lower[...,0]<width-1)&(lower[...,1]>=0)&(lower[...,1]<h-1)
                        full=full.all(-1)
                        ambiguous=(np.abs(pixel-np.round(pixel))<=p['occupancy_boundary_ambiguity_cells']).any((-1,-2))
                        collapse=same & full & ~ambiguous & (energy>1e-20) & (relative<p['near_constant_relative_RMS'])
                        tag=f'P{level+3}'
                        unique=np.array([[len(np.unique(lower[b,q].reshape(-1,2),axis=0)) for q in range(300)] for b in range(4)])
                        metrics.update({tag+'_attention_mass':mass.mean(-1),tag+'_relative_value_RMS':np.median(relative,axis=-1),
                            tag+'_same_stencil_head_fraction':same.mean(-1),tag+'_joint_collapse_head_fraction':collapse.mean(-1),
                            tag+'_boundary_ambiguous_head_fraction':ambiguous.mean(-1),tag+'_all_head_stencils':unique,
                            tag+'_fully_in_bounds_head_fraction':full.mean(-1),tag+'_median_span_cells':np.median(np.linalg.norm(pixel.max(-2)-pixel.min(-2),axis=-1),axis=-1),
                            tag+'_joint_collapse_attention_mass':(mass*collapse).sum(-1)/8})
                        start=stop
                    reconstructed=accumulated.reshape(4,300,256)
                    actual=load(event(f'decoder.decoder.layers.{layer}.cross_attn')).astype(np.float64)
                    delta=reconstructed-actual
                    max_abs=float(np.max(np.abs(delta)));relative=float(np.linalg.norm(delta)/max(np.linalg.norm(actual),1e-30))
                    controls.append(dict(batch=bi,layer=layer,final_box_reconstruction_max_abs=reconstruction_error,
                        cross_attention_max_abs=max_abs,cross_attention_relative_L2=relative))
                    assert max_abs<=p['reconstruction_control']['cross_attention_max_abs_tolerance'],controls[-1]
                    assert relative<=p['reconstruction_control']['cross_attention_relative_L2_tolerance'],controls[-1]
                    layer_metrics.append(metrics)
                    for k,v in metrics.items():derived[f'batch_{bi:03d}_layer{layer}_{k}']=v
                    for b,iid in enumerate(ids):
                        for q in range(300):
                            row=dict(seed=seed,image_id=iid,layer=layer,query=q,reference_short_px=float(refs[layer][b,q,2:].min()*640))
                            row.update({k:float(v[b,q]) for k,v in metrics.items()});rows.append(row)
                for b,iid in enumerate(ids):
                    gt=targets[b]/640;mat=iou(gt,xyxy(pre[b]));q=mat.argmax(-1);maximum=mat.max(-1)
                    final_mat=iou(gt,xyxy(final[b]));final_max=final_mat.max(-1)
                    for g in range(len(gt)):
                        row=dict(seed=seed,image_id=iid,GT_native_index=g,category_id=int(labels[b][g]),
                            input_short_px=float((targets[b][g,2:]-targets[b][g,:2]).min()),pre_best_query=int(q[g]),
                            pre_any_max_IoU=float(maximum[g]),pre_zero_IoU_link=bool(maximum[g]==0),
                            final_any_max_IoU=float(final_max[g]),final_any_geometry_failed50=bool(final_max[g]<.5))
                        for layer,m in enumerate(layer_metrics):
                            for k,v in m.items():row[f'layer{layer}_'+k]=float(v[b,q[g]])
                        row['majority_P3_joint_collapse_any_layer']=bool(any(m['P3_joint_collapse_head_fraction'][b,q[g]]>.5 for m in layer_metrics))
                        gt_rows.append(row)
        np.savez_compressed(out/'query_metrics.npz',**derived)
        for name,data in (('per_query.csv',rows),('per_GT_link.csv',gt_rows)):
            with (out/name).open('x',encoding='utf8',newline='') as file:w=csv.DictWriter(file,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        groups={}
        for name,lo,hi in [('under4',0,4),('4_to_under8',4,8),('8_to_under16',8,16),('ge16',16,float('inf'))]:
            selected=[r for r in gt_rows if lo<=r['input_short_px']<hi]
            eligible=[r for r in selected if not r['pre_zero_IoU_link']]
            failed=[r for r in eligible if r['final_any_geometry_failed50']]
            groups[name]=dict(GT=len(selected),eligible_positive_pre_IoU_links=len(eligible),excluded_zero_IoU_links=len(selected)-len(eligible),
                eligible_final_geometry_failed50=len(failed),majority_joint_collapse_failed_GT=sum(r['majority_P3_joint_collapse_any_layer'] for r in failed),
                majority_joint_collapse_failed_fraction=float(np.mean([r['majority_P3_joint_collapse_any_layer'] for r in failed])) if failed else None,
                unique_pre_query_links=len({(r['image_id'],r['pre_best_query']) for r in eligible}))
        assert len(gt_rows)==960 and groups['under4']['GT']==134
        # All300 query summaries avoid silently treating shared GT-query links as independent queries.
        summaries=[]
        for layer in range(3):
            selected=[r for r in rows if r['layer']==layer]
            a={k:np.array([r[k] for r in selected]) for k in selected[0]}
            summaries.append(dict(layer=layer,**summarize_query(a)))
        fraction=groups['under4']['majority_joint_collapse_failed_fraction']
        report=dict(status='VERIFIED_SAVED_NATIVE_CROSS_ATTENTION_RECONSTRUCTION_AND_SAMPLING_DESCRIPTION',seed=seed,
            source_arm='target_protect',images=8,native_GT=960,native_under4_GT=134,
            queries_per_layer=2400,query_summaries=summaries,groups=groups,numeric_controls=controls,
            concentration_precondition_this_seed=bool(fraction is not None and fraction>=.25),
            protocol_sha256=sha(HERE/'protocol.json'),script_sha256=sha(__file__),sources=binding,
            actual_consumed_raw_blob_hashes=consumed,new_model_forwards=0,new_evaluator_calls=0,optimizer_updates=0,epochs_added=0,
            limitations=['Eight reused source-GT-selected images, not a whole-dataset or independent test.',
            'Previous target-policy EMA12, not the currently tested FDR endpoint2 model.',
            'Locations, bilinear values and softmax are reconstructed and numerically checked; they were not separately hooked.',
            'Same bilinear stencil alone does not imply constant values, erased position information or manifold collapse.',
            'GT-linked query selection is descriptive and may be many-to-one; it is not native matching or training responsibility.',
            'Concentration precondition does not prove a cause or AP benefit. No parameter sweep or model intervention occurred.'])
        write(out/'report.json',report)
        write(out/'manifest.json',dict(files={x.name:sha(x) for x in out.iterdir() if x.is_file()},protocol_sha256=sha(HERE/'protocol.json')))
        assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
        print(json.dumps(dict(status=report['status'],seed=seed,groups=groups,query_summaries=summaries,precondition=report['concentration_precondition_this_seed'])))
    except BaseException:
        import traceback
        write(out/'failure.json',dict(status='SAVED_SAMPLING_RECONSTRUCTION_FAILED_NO_MODEL_RETRY',traceback=traceback.format_exc(),
            completed_control_checks=controls,model_forwards=0,evaluator_calls=0,optimizer_updates=0,epochs_added=0));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',required=True,type=int,choices=(1,2))
    main(parser.parse_args().seed)
