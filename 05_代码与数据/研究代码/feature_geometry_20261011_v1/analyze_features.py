"""Read verified raw activations; descriptive point neighborhoods, never detector AP."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/feature_geometry_20261011_v1'
COHORTS=['under4','4_to8','8_to16','16_to32','at_least32']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)


def sample(a,points):
    c,h,w=a.shape;p=np.asarray(points,dtype=np.float64).reshape(-1,2)
    q=np.clip(p/640*np.array([w,h])-.5,0,np.array([w-1,h-1]))
    lo=np.floor(q).astype(np.int64);hi=np.minimum(lo+1,[w-1,h-1]);frac=q-lo
    x,y=lo[:,0],lo[:,1];X,Y=hi[:,0],hi[:,1];u,v=frac[:,0,None],frac[:,1,None]
    return a[:,y,x].T*(1-u)*(1-v)+a[:,y,X].T*u*(1-v)+a[:,Y,x].T*(1-u)*v+a[:,Y,X].T*u*v


def neighbors(embedding,iids,groups,keys):
    reports={}
    for cohort in COHORTS:
        selected=[]
        for iid in sorted(set(iids)):
            ids=[i for i in range(len(iids)) if iids[i]==iid and groups[i]==cohort]
            ids.sort(key=lambda i:hashlib.sha256(f'feature_knn:{iid}:{keys[i]}'.encode()).hexdigest())
            selected.extend(ids[:64])
        if not selected:reports[cohort]={'status':'NO_MATCHED_POINTS'};continue
        feats=embedding[selected].reshape(-1,embedding.shape[-1]);ids=np.repeat(iids[selected],2)
        labels=np.tile([1,0],len(selected));folds=[];zero=int((np.linalg.norm(feats,axis=1)==0).sum())
        for iid in sorted(set(ids)):
            train=ids!=iid;test=ids==iid
            if train.sum()<6:continue
            similarities=feats[test]@feats[train].T;order=np.argsort(-similarities,axis=1,kind='stable')
            sorted_sim=np.take_along_axis(similarities,order,axis=1)
            tie=np.abs(sorted_sim[:,4]-sorted_sim[:,5])<=1e-12
            predict=(labels[train][order[:,:5]].sum(1)>=3).astype(np.int64);truth=labels[test]
            values={}
            for label,name in [(1,'foreground_center_recall'),(0,'selected_GT_free_point_recall')]:
                keep=(truth==label)&~tie;values[name]=float(np.mean(predict[keep]==label)) if keep.any() else None
            accuracy=(values['foreground_center_recall']+values['selected_GT_free_point_recall'])/2 if all(x is not None for x in values.values()) else None
            folds.append(dict(image_id=int(iid),train_vectors=int(train.sum()),test_vectors=int(test.sum()),boundary_ties=int(tie.sum()),balanced_accuracy=accuracy,**values))
        valid=[v['balanced_accuracy'] for v in folds if v['balanced_accuracy'] is not None]
        reports[cohort]=dict(status='DESCRIPTIVE_LEAVE_IMAGE_OUT_5NN',matched_GT_pairs=len(selected),zero_norm_vectors=zero,
            image_balanced_accuracy_mean=float(np.mean(valid)) if valid else None,folds=folds,
            chance_balanced_accuracy=.5,semantic_class_or_detection_AP_claim=False)
    return reports


def main(seed):
    protocol=__import__('json').loads((OUT/'protocol.json').read_text());code=Path(__file__).parent
    for name,digest in protocol['code_sha256'].items():assert sha(code/name)==digest
    download=ROOT/f'outputs/candidate_all_module_capture_20261010_v1/seed{seed}_download_v1'
    rawreport=json.loads((download/'raw_verification_report_v1.json').read_text())
    assert rawreport['status']=='PASS_DOWNLOADED_ALL_CALLED_MODULE_RAW_OUTPUTS'
    assert rawreport['archive_sha256']==protocol['raw_archive_sha256'][str(seed)]
    exports=OUT/f'seed{seed}_weights';meta=json.loads((exports/f'projection_seed{seed}_v1_metadata.json').read_text())
    weightfile=exports/f'projection_seed{seed}_v1.npz';assert sha(weightfile)==meta['array_sha256']
    assert meta['export_source_sha256']==protocol['code_sha256']['export_projection.py'] and not meta['CUDA_initialized']
    with np.load(weightfile,allow_pickle=False) as weights:
        conv=weights['conv_weight'][:,:,0,0].astype(np.float64)
        scale=weights['norm_weight']/np.sqrt(weights['norm_running_var'].astype(np.float64)+float(weights['BN_eps']))
        matrix=scale[:,None]*conv;bias=weights['norm_bias']-scale*weights['norm_running_mean']
    singular=np.linalg.svd(matrix,compute_uv=False);tolerance=np.finfo(np.float64).eps*256*singular[0]
    projection=dict(numerical_rank=int((singular>tolerance).sum()),dimension=256,
        min_singular=float(singular[-1]),max_singular=float(singular[0]),rank_tolerance=float(tolerance),
        condition_number=float(singular[0]/singular[-1]) if singular[-1]>0 else None,
        full_rank_affine_change_not_exact_arithmetic_information_loss_proof=True)
    dest=OUT/f'seed{seed}_analysis';dest.mkdir(exist_ok=False)
    rows=[];point_rows=[];checks=[];embeddings={name:[] for name,*rest in protocol['stages']};catalog=[];arrays={}
    saved={}
    with (ROOT/f'outputs/candidate_all_module_capture_20261010_v1/decoder_trajectory_seed{seed}_v1/per_GT_trajectory.csv').open(encoding='utf8',newline='') as f:
        for row in csv.DictReader(f):saved[(int(row['image_id']),int(row['GT_native_index']))]=row
    with zipfile.ZipFile(download/f'module_candidate_panel_seed{seed}_raw_v1.zip') as z:
        for bi in (25,105):
            base=f'capture/batch_{bi:03d}/';record=json.loads(z.read(base+'record.json'))
            def load(desc):
                raw=z.read(base+desc['blob_file']);assert hashlib.sha256(raw).hexdigest()==desc['blob_sha256']
                a=np.load(io.BytesIO(raw),allow_pickle=False);assert list(a.shape)==desc['shape'] and str(a.dtype)==desc['storage_numpy_dtype']
                assert np.isfinite(a).all();return a
            def output(module):
                found=[e['output'] for e in record['invocation_events'] if e['module']==module]
                assert len(found)==1 and found[0]['kind']=='tensor';return found[0]
            panels=[]
            for offset,iid in enumerate(record['metadata']['image_ids']):
                gt={x['key']['value']:x['value'] for x in record['native_targets']['items'][offset]['entries']}
                boxes=load(gt['boxes']).astype(np.float64);labels=load(gt['labels']).astype(np.int64)
                center=(boxes[:,:2]+boxes[:,2:])*.5;short=(boxes[:,2:]-boxes[:,:2]).min(1)
                assert (short>0).all() and boxes.min()>=0 and boxes.max()<=640.0001
                included=[];back16=[];back32=[]
                for g,p in enumerate(center):
                    choices=[]
                    for radius in (16,32):
                        order=sorted(range(16),key=lambda j:hashlib.sha256(f'feature_point:{iid}:{g}:{radius}:{j}'.encode()).hexdigest())
                        chosen=None
                        for j in order:
                            angle=2*np.pi*j/16;q=p+radius*np.array([np.cos(angle),np.sin(angle)])
                            if (q<.5).any() or (q>639.5).any():continue
                            occupied=((q>=boxes[:,:2]-2)&(q<=boxes[:,2:]+2)).all(1).any()
                            if not occupied:chosen=q;break
                        choices.append(chosen)
                    cohort=COHORTS[int(np.searchsorted([4,8,16,32],short[g],side='right'))]
                    status=all(q is not None for q in choices)
                    point_rows.append(dict(image_id=iid,GT_native_index=g,category_id=int(labels[g]),input_short_px=float(short[g]),cohort=cohort,both_radii_valid=status,
                        center_x=float(p[0]),center_y=float(p[1]),background16_x=float(choices[0][0]) if choices[0] is not None else None,background16_y=float(choices[0][1]) if choices[0] is not None else None,
                        background32_x=float(choices[1][0]) if choices[1] is not None else None,background32_y=float(choices[1][1]) if choices[1] is not None else None))
                    if status:
                        included.append(g);back16.append(choices[0]);back32.append(choices[1]);catalog.append((iid,g,cohort))
                included=np.asarray(included,dtype=np.int64)
                panels.append((iid,included,center[included],np.asarray(back16).reshape(-1,2),np.asarray(back32).reshape(-1,2),short[included],labels[included]))
            before=load(output('backbone.stages.1'));after=load(output('encoder.input_proj.0'))
            assert output('encoder.fpn_blocks.1')['blob_sha256']==output('decoder.input_proj.0')['blob_sha256']
            for offset,(iid,selected,pos,b16,b32,short,labels) in enumerate(panels):
                points=np.concatenate([pos,b16,b32]);x=sample(before[offset].astype(np.float64),points);y=sample(after[offset].astype(np.float64),points)
                error=float(np.max(np.abs(x@matrix.T+bias-y))) if len(points) else 0.
                bound=2e-4+1e-3*float(np.max(np.abs(y))) if len(points) else 2e-4
                assert error<=bound,(iid,error,bound)
                checks.append(dict(image_id=iid,P3_projected_sample_affine_error=error,tolerance=bound,decoder_P3_input_exact_identity=True))
            for name,module,layout,h,w in protocol['stages']:
                a=load(output(module))
                if layout=='P3_first6400':assert a.shape==(4,8400,256);a=a[:,:6400].transpose(0,2,1).reshape(4,256,80,80)
                elif layout=='Tokens400':assert a.shape==(4,400,256);a=a.transpose(0,2,1).reshape(4,256,20,20)
                assert a.shape[0]==4 and a.shape[2:]==(h,w)
                for offset,(iid,selected,pos,b16,b32,short,labels) in enumerate(panels):
                    feature=a[offset].astype(np.float64);flat=feature.reshape(feature.shape[0],-1)
                    mean=flat.mean(1);std=flat.std(1);valid=std>1e-8
                    assert valid.any()
                    p=sample(feature,pos);p_norm=np.linalg.norm(p,axis=1)
                    paired=[]
                    for radius,bg in ((16,b16),(32,b32)):
                        b=sample(feature,bg);b_norm=np.linalg.norm(b,axis=1)
                        cosine=1-np.einsum('ij,ij->i',p,b)/np.maximum(p_norm*b_norm,1e-30)
                        rms=np.sqrt(np.mean(((p[:,valid]-b[:,valid])/std[valid])**2,axis=1))
                        cells_p=np.floor(pos/640*np.array([w,h])).astype(np.int64);cells_b=np.floor(bg/640*np.array([w,h])).astype(np.int64)
                        same=(cells_p==cells_b).all(1)
                        for index,g in enumerate(selected):
                            cohort=COHORTS[int(np.searchsorted([4,8,16,32],short[index],side='right'))]
                            final=float(saved[(iid,int(g))]['FDR_layer2_final_maxIoU'])
                            rows.append(dict(stage=name,module=module,H=h,W=w,C=feature.shape[0],image_id=iid,GT_native_index=int(g),category_id=int(labels[index]),cohort=cohort,radius=radius,
                                raw_cosine_distance=float(cosine[index]),standardized_RMS_distance=float(rms[index]),same_nearest_anchor_cell=bool(same[index]),
                                positive_zero_norm=bool(p_norm[index]==0),negative_zero_norm=bool(b_norm[index]==0),variable_channels=int(valid.sum()),final_geometry_IoU50_qualified=final>=.5))
                        if radius==32:
                            vectors=np.stack([(p-mean)/np.maximum(std,1e-8),(b-mean)/np.maximum(std,1e-8)],axis=1)
                            length=np.linalg.norm(vectors,axis=2,keepdims=True);paired=vectors/np.maximum(length,1e-30)
                    embeddings[name].append(paired)
    ids=np.array([v[0] for v in catalog]);keys=np.array([v[1] for v in catalog]);groups=np.array([v[2] for v in catalog])
    summaries={};knn={}
    for name,*rest in protocol['stages']:
        embed=np.concatenate(embeddings[name]);assert len(embed)==len(catalog)
        arrays[name]=embed;knn[name]=neighbors(embed,ids,groups,keys);summaries[name]={}
        for cohort in COHORTS:
            summaries[name][cohort]={}
            for radius in (16,32):
                group=[r for r in rows if r['stage']==name and r['cohort']==cohort and r['radius']==radius]
                if not group:summaries[name][cohort][str(radius)]={'GT':0};continue
                image_values={int(i):float(np.median([r['standardized_RMS_distance'] for r in group if r['image_id']==i])) for i in sorted({r['image_id'] for r in group})}
                summaries[name][cohort][str(radius)]=dict(GT=len(group),images=len(image_values),
                    raw_cosine_median=float(np.median([r['raw_cosine_distance'] for r in group])),
                    standardized_RMS_median=float(np.median([r['standardized_RMS_distance'] for r in group])),
                    image_balanced_mean_of_RMS_medians=float(np.mean(list(image_values.values()))),per_image_RMS_medians=image_values,
                    same_nearest_anchor_cell=int(sum(r['same_nearest_anchor_cell'] for r in group)),
                    zero_norm_pairs=int(sum(r['positive_zero_norm'] or r['negative_zero_norm'] for r in group)),
                    descriptive_final_geometry_subgroups={str(label):dict(GT=sum(r['final_geometry_IoU50_qualified']==label for r in group),
                        RMS_median=float(np.median([r['standardized_RMS_distance'] for r in group if r['final_geometry_IoU50_qualified']==label])) if any(r['final_geometry_IoU50_qualified']==label for r in group) else None) for label in (False,True)})
    arrays.update(image_ids=ids,GT_native_indices=keys,cohorts=groups,projection_singular_values=singular)
    np.savez_compressed(dest/'derived_embeddings.npz',**arrays)
    for filename,values in [('point_catalog.csv',point_rows),('per_GT_feature_metrics.csv',rows)]:
        with (dest/filename).open('x',encoding='utf8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    report=dict(status='VERIFIED_SAVED_FEATURE_DESCRIPTIVE_LOCAL_GEOMETRY',seed=seed,protocol_sha256=sha(OUT/'protocol.json'),
        projection_weights_sha256=meta['array_sha256'],raw_archive_sha256=rawreport['archive_sha256'],
        native_GT_count=len(point_rows),matched_GT_count=len(catalog),matched_by_cohort={c:int((groups==c).sum()) for c in COHORTS},
        excluded_missing_fixed_background_pair=sum(not r['both_radii_valid'] for r in point_rows),
        P3_affine_projection=projection,numeric_checks=checks,local_distances=summaries,leave_image_out_5NN=knn,
        true_background_semantics_not_verified=True,feature_distance_not_information_loss_or_causal_root=True,
        intrinsic_manifold_or_population_detection_AP_not_established=True,new_detector_forwards=0,new_optimizer_updates=0,new_epochs=0)
    write(dest/'report.json',report)
    write(dest/'manifest.json',dict(files={p.name:sha(p) for p in dest.iterdir() if p.is_file()},protocol_sha256=sha(OUT/'protocol.json')))
    print(json.dumps(dict(status=report['status'],seed=seed,matched_GT=len(catalog),matched_by_cohort=report['matched_by_cohort'],projection=projection,
        P3_under4_RMS32={s:summaries[s]['under4']['32'] for s in ['backbone_P3','encoder_proj_P3','FPN_P3','encoder_memory_P3']})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=(1,2),required=True);main(p.parse_args().seed)
