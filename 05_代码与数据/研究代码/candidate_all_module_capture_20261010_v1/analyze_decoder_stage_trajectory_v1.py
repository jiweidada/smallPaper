"""Actual saved outputs within one forward: geometry by layer and score proxy."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from analyze_frozen_tail_delta_v2 import sha, iou, xyxy, from_distance

ROOT=Path(__file__).resolve().parents[3]
STAGES=['encoder_top300','pre_after_layer0','FDR_layer0','FDR_layer1','FDR_layer2_final']


def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def main(seed):
    download=ROOT/f'outputs/candidate_all_module_capture_20261010_v1/seed{seed}_download_v1'
    raw_report=__import__('json').loads((download/'raw_verification_report_v1.json').read_text())
    assert raw_report['status']=='PASS_DOWNLOADED_ALL_CALLED_MODULE_RAW_OUTPUTS'
    assert raw_report['seed']==seed and raw_report['original_capture_exit_code']==0
    archive=download/f'module_candidate_panel_seed{seed}_raw_v1.zip'
    metadata=__import__('json').loads((download/f'module_candidate_panel_seed{seed}_raw_v1_metadata.json').read_text())
    assert metadata['sha256']==raw_report['archive_sha256'] and archive.stat().st_size==metadata['bytes']
    target=ROOT/f'outputs/scale_protection12_20261010_v1/terminal_collection/seed{seed}_target_protect_v1'
    verification=__import__('json').loads((target/'verification_report_v1.json').read_text())
    assert verification['status']=='PASS_FIXED12_DOWNLOADED_EVIDENCE_AND_SAVED_WEIGHT_RELOAD'
    stagepath=target/'extracted/run/stages_epoch12'
    stage_manifest=__import__('json').loads((stagepath/'manifest.json').read_text())
    out=ROOT/f'outputs/candidate_all_module_capture_20261010_v1/decoder_trajectory_seed{seed}_v1'
    assert not out.exists();out.mkdir()
    protocol=dict(schema='saved_decoder_stage_trajectory_v1',seed=seed,source_panel_images=8,
        raw_archive_sha256=metadata['sha256'],raw_verifier_report_sha256=sha(download/'raw_verification_report_v1.json'),
        target_verifier_report_sha256=sha(target/'verification_report_v1.json'),thresholds=[.5,.75],
        stages=STAGES,intermediate_boxes_reconstructed_from_actual_Integral_and_pre_reference=True,
        final_boxes_and_scores_equal_native_saved_outputs_required=True,
        box_reconstruction_tolerance=1e-6,intermediate_threshold_ambiguity_band=1e-4,
        final_score_cutoff_ambiguity='Exact equal logit ties are flagged; Top300 is a logit-order proxy, not a native COCO re-evaluation.',
        native_validation_GT_format='absolute input-pixel XYXY',classification_scores_only_pre_and_final=True,
        intermediate_classification_heads_not_executed_during_eval_not_invented=True,
        within_forward_query_slots_preserved=True,cross_run_query_identity_assumed=False,
        any_query_GT_geometry_not_one_to_one_matching=True,
        no_new_detector_forwards=0,no_evaluator_calls=0,no_optimizer_updates=0,no_training_epochs=0,
        root_cause_or_population_AP_claim=False)
    write(out/'protocol.json',protocol)
    rows=[];image_rows=[];numeric=[];arrays={}
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('capture/manifest.json'))
        assert hashlib.sha256(z.read('capture/manifest.json')).hexdigest()==metadata['manifest_sha256']
        for bi in (25,105):
            base=f'capture/batch_{bi:03d}/';record=json.loads(z.read(base+'record.json'))
            def load(desc):
                raw=z.read(base+desc['blob_file']); assert hashlib.sha256(raw).hexdigest()==desc['blob_sha256']
                a=np.load(io.BytesIO(raw),allow_pickle=False)
                assert list(a.shape)==desc['shape'] and str(a.dtype)==desc['storage_numpy_dtype'] and np.isfinite(a).all()
                return a
            def event(name,inv=0):
                found=[x for x in record['invocation_events'] if x['module']==name and x['invocation']==inv]
                assert len(found)==1;return found[0]
            distances=[load(event('decoder.integral',j)['output']) for j in range(3)]
            raw_score=load(event('decoder.dec_score_head.2')['output'])
            final_score=load(event('decoder.decoder.lqe_layers.2')['output'])
            assert not any(x['module']=='decoder.dec_score_head.1' for x in record['invocation_events'])
            ids=record['metadata']['image_ids']
            for offset,iid in enumerate(ids):
                name=f'image_{iid:06d}.npz';file=stagepath/name
                assert sha(file)==stage_manifest['files'][name]
                entries=record['native_targets']['items'][offset]['entries']
                gt_desc={x['key']['value']:x['value'] for x in entries}
                gt_pixels=load(gt_desc['boxes']).astype(np.float64)
                labels=load(gt_desc['labels']).astype(np.int64)
                assert len(gt_pixels)==len(labels) and (labels>=0).all() and (labels<10).all()
                assert (gt_pixels[:,2:]>gt_pixels[:,:2]).all() and gt_pixels.min()>=0 and gt_pixels.max()<=640.0001
                gt=gt_pixels/640.;side=(gt_pixels[:,2:]-gt_pixels[:,:2]).min(1)
                with np.load(file,allow_pickle=False) as stage:
                    ref=stage['pre_after_first_layer_boxes'].astype(np.float64)
                    boxes=[xyxy(stage['encoder_initial_boxes'].astype(np.float64)),xyxy(ref)]
                    boxes.extend(from_distance(ref,distance[offset].astype(np.float64)) for distance in distances[:2])
                    boxes.append(xyxy(stage['final_boxes'].astype(np.float64)))
                    final_error=float(np.max(np.abs(from_distance(ref,distances[-1][offset].astype(np.float64))-boxes[-1])))
                    assert final_error<=protocol['box_reconstruction_tolerance']
                    assert np.array_equal(final_score[offset],stage['final_logits'])
                matches=[iou(gt,b) for b in boxes]
                maxima=np.stack([m.max(1) for m in matches],axis=1)
                slot=np.stack([m.argmax(1) for m in matches],axis=1)
                rank_masks=[];ties=[]
                for score in (raw_score[offset],final_score[offset]):
                    flat=score.flatten();order=np.argsort(-flat,kind='stable')
                    boundary=float(flat[order[299]])
                    ties.append(bool(np.count_nonzero(flat==boundary)>1))
                    mask=np.zeros(3000,dtype=bool);mask[order[:300]]=True
                    rank_masks.append(mask.reshape(300,10))
                rank_geo=[np.array([matches[-1][g,mask[:,labels[g]]].max(initial=0.) for g in range(len(gt))])
                          for mask in rank_masks]
                for g in range(len(gt)):
                    row=dict(image_id=iid,GT_native_index=g,category_id=int(labels[g]),input_short_px=float(side[g]),under4=bool(side[g]<4))
                    for j,s in enumerate(STAGES):row[s+'_maxIoU']=float(maxima[g,j]);row[s+'_best_query']=int(slot[g,j])
                    row.update(raw_score_Top300_logit_proxy_IoU=float(rank_geo[0][g]),
                        LQE_Top300_logit_proxy_IoU=float(rank_geo[1][g]),
                        raw_score_cutoff_has_ties=ties[0],LQE_score_cutoff_has_ties=ties[1])
                    rows.append(row)
                image_rows.append(dict(image_id=iid,GT=len(gt),under4=int((side<4).sum()),
                    score_proxy_boundary_ties=ties,geometry_IoU_max_change_by_transition=[
                    float(np.abs(maxima[:,j+1]-maxima[:,j]).max()) for j in range(4)]))
                numeric.append(dict(image_id=iid,final_native_box_reconstruction_error=final_error,final_logits_exact=True))
                arrays[f'image_{iid:06d}_GT_stage_maxIoU']=maxima
                arrays[f'image_{iid:06d}_GT_stage_best_query']=slot
                arrays[f'image_{iid:06d}_GT_rank_proxy_maxIoU']=np.stack(rank_geo,axis=1)
                for j in (2,3):arrays[f'image_{iid:06d}_{STAGES[j]}_boxes_xyxy']=boxes[j]
    summary={}
    for cohort in ('all_GT','input_short_under4'):
        selected=rows if cohort=='all_GT' else [x for x in rows if x['under4']]
        maxima=np.array([[x[s+'_maxIoU'] for s in STAGES] for x in selected]).reshape(-1,5)
        thresholds={}
        for t in protocol['thresholds']:
            covered=maxima>=t
            transitions=[dict(from_stage=STAGES[j],to_stage=STAGES[j+1],
                rescued=int((~covered[:,j]&covered[:,j+1]).sum()),
                damaged=int((covered[:,j]&~covered[:,j+1]).sum())) for j in range(4)]
            final_failed=~covered[:,-1];never=~covered.any(1)
            raw=np.array([x['raw_score_Top300_logit_proxy_IoU'] for x in selected])>=t
            lqe=np.array([x['LQE_Top300_logit_proxy_IoU'] for x in selected])>=t
            stable=np.array([not(x['raw_score_cutoff_has_ties'] or x['LQE_score_cutoff_has_ties']) for x in selected],dtype=bool)
            ambiguous=((np.abs(maxima[:,2:4]-t)<=protocol['intermediate_threshold_ambiguity_band']).any(1))
            thresholds[str(t)]=dict(stage_any_query_geometry_counts=dict(zip(STAGES,covered.sum(0).astype(int).tolist())),
                transitions=transitions,final_geometry_failed=int(final_failed.sum()),
                never_geometry_covered_in_observed_stages=int(never.sum()),
                previously_geometry_covered_but_final_failed=int((covered[:,:-1].any(1)&final_failed).sum()),
                intermediate_threshold_ambiguous_GT=int(ambiguous.sum()),
                logit_order_proxy=dict(raw_score_cover=int(raw.sum()),LQE_cover=int(lqe.sum()),
                    LQE_rescued=int((stable&~raw&lqe).sum()),LQE_damaged=int((stable&raw&~lqe).sum()),
                    boundary_tied_images_excluded_GT=int((~stable).sum()),
                    final_geometry_available_but_LQE_proxy_missing=int((covered[:,-1]&~lqe).sum()),
                    native_COCO_matching_or_AP_not_recomputed=True))
        summary[cohort]=dict(GT=len(selected),thresholds=thresholds)
    np.savez_compressed(out/'derived_arrays.npz',**arrays)
    with (out/'per_GT_trajectory.csv').open('x',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(out/'report.json',dict(status='VERIFIED_WITHIN_FORWARD_LAYER_GEOMETRY_AND_LOGIT_ORDER_PROXY',
        seed=seed,images=8,geometry=summary,image_checks=image_rows,numeric_checks=numeric,
        source_panel_selection_was_source_GT_not_model_error=True,intermediate_class_logits_not_invented=True,
        causal_first_failure_or_dataset_population_claim=False,new_model_forwards=0,new_epochs=0))
    write(out/'manifest.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        source_sha256=sha(__file__),helper_source_sha256=sha(Path(__file__).with_name('analyze_frozen_tail_delta_v2.py'))))
    print(json.dumps(dict(status='DECODER_LAYER_GEOMETRY_DIAGNOSTIC_COMPLETE',seed=seed,summary=summary)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',required=True,type=int,choices=(1,2));main(p.parse_args().seed)
