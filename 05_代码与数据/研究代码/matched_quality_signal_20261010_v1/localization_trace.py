"""Prospectively frozen follow-up: fixed-query center/size output trace.

Run --freeze first, then --analyze once. Oracle replacements use GT and are
geometric diagnostics only; they are not deployable predictions or AP bounds.
"""
from pathlib import Path
import argparse, csv, datetime, hashlib, json, numpy as np
from analyze_saved import iou_pairs, xyxy
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/matched_quality_signal_20261010_v1'
INPUT=ROOT/'outputs/native_augmented_supervision_20261010_v1'
SOURCE=ROOT/'outputs/model_structure_html_20261010_v1/collection_complete/captured_complete_graph_v3/native_source/engine/deim/dfine_decoder.py'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
STAGES=['model.enc_aux_outputs.0','model.pre_outputs','model.aux_outputs.0','model.aux_outputs.1','model']
NAMES=['selected_encoder_boxes','pre_head_after_decoder_layer0','FDR_after_layer0','FDR_after_layer1','FDR_after_layer2_final']
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def freeze():
    p=OUT/'localization_followup_protocol.json';assert not p.exists()
    save(p,{'status':'FROZEN_AFTER_MAL_ANALYSIS_BEFORE_LOCALIZATION_COMPONENT_RESULTS','frozen_local_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'motivation':'Preceding verified MAL study shows low pair IoU and intentional class-quality coupling; inspect output-stage center/size errors before proposing a change.','input_protocol_sha256':sha(INPUT/'protocol.json'),'prior_MAL_protocol_sha256':sha(OUT/'protocol.json'),'code_sha256':sha(Path(__file__)),'geometry_helper_sha256':sha(Path(__file__).parent/'analyze_saved.py'),'native_decoder_sha256':sha(SOURCE),'seeds':[1,2],'endpoints':[12,132],'cohort':'Same complete archived64batches. Fixed final own-matcher query/GT pair traced across all5 normal-head boxes; no rematching, no new outputs, no row exclusion.','primary':'Same mature final GT<=300 input-short<4 stratum as preceding MAL; both seeds separately. Other sizes/endpoints/density retained.','query_identity':'Native non-group path; enc_topk boxes become reference boxes in same order; DN prepended then split out consistently. pre_outputs is AFTER first decoder layer and pre_bbox_head, not decoder entry.','stages':dict(zip(STAGES,NAMES)),'statistics':['Pair IoU>=.5/ .75 and mean across stage at fixed query','GT-center replacement keeping predicted size; GT-size replacement keeping predicted center, separately, for same fixed pair','All300 head queries maximum geometric IoU for each saved final-matched GT; class/ownership ignored, not detection recall','Native-center error in input pixels, size ratios, normalized center error and center-vs-size oracle joint categories','Fixed query crossings of .5/.75 from selected-encoder boxes to final FDR'], 'oracle_limits':'GT substitutions are output algebra and not feasible input/model interventions, learned corrections, AP, independent error attribution or proof that a particular network structure caused the defect. Center and size errors interact. Final query selection is prediction dependent.','reporting':'Descriptive counts; no gate/novelty/AP significance and no threshold chosen after results. No detector candidate launched.','budget':{'model_forwards':0,'model_backwards':0,'optimizer_steps':0,'epochs':0}})
    print(json.dumps({'status':'FROZEN','protocol_sha256':sha(p)}))
def best_iou(pred,gt):
    a,b=xyxy(pred),xyxy(gt);wh=np.maximum(np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]),np.float32(0));inter=wh[:,:,0]*wh[:,:,1];ar=(a[:,2]-a[:,0])*(a[:,3]-a[:,1]);br=(b[:,2]-b[:,0])*(b[:,3]-b[:,1]);return (inter/(ar[:,None]+br[None,:]-inter)).max(0)
def summarize(rows):
    n=len(rows)
    return {'rows':n,'mean_pair_IoU':float(np.mean([r['IoU'] for r in rows])) if n else None,'pair_ge50':sum(r['IoU']>=.5 for r in rows),'pair_ge75':sum(r['IoU']>=.75 for r in rows),'center_GT_oracle_ge50':sum(r['center_GT_oracle_IoU']>=.5 for r in rows),'center_GT_oracle_ge75':sum(r['center_GT_oracle_IoU']>=.75 for r in rows),'size_GT_oracle_ge50':sum(r['size_GT_oracle_IoU']>=.5 for r in rows),'size_GT_oracle_ge75':sum(r['size_GT_oracle_IoU']>=.75 for r in rows),'any_query_ge50':sum(r['any_query_max_IoU']>=.5 for r in rows),'any_query_ge75':sum(r['any_query_max_IoU']>=.75 for r in rows),'median_center_error_pixels':float(np.median([r['center_error_pixels'] for r in rows])) if n else None,'median_center_error_relative_short':float(np.median([r['center_error_relative_short'] for r in rows])) if n else None,'median_width_ratio':float(np.median([r['width_ratio'] for r in rows])) if n else None,'median_height_ratio':float(np.median([r['height_ratio'] for r in rows])) if n else None}
def analyze():
    protocol=read(OUT/'localization_followup_protocol.json');assert sha(Path(__file__))==protocol['code_sha256'] and sha(SOURCE)==protocol['native_decoder_sha256'];assert sha(Path(__file__).parent/'analyze_saved.py')==protocol['geometry_helper_sha256']
    target=OUT/'localization_trace';target.mkdir(exist_ok=False);rows=[];transitions=[]
    for seed in (1,2):
      for endpoint in (12,132):
       for batch in range(16):
        path=INPUT/f'collection_seed{seed}/capture_v1/endpoint{endpoint}/batch_{batch:03d}';manifest=read(path/'manifest.json')
        for name,h in manifest['files'].items():assert sha(path/name)==h
        ev=read(path/'evidence.json');final=next(e for e in ev['events'] if e['kind']=='loss_call' and e['loss']=='mal' and e['stage']=='model' and not e['is_DN']);size=ev['metadata']['input_shape'][-1]
        with np.load(path/'arrays.npz',allow_pickle=False) as npz:data={k:npz[k] for k in npz.files}
        for j in range(4):
          q=data[final['prefix']+f'.batch{j}.query_index'];g=data[final['prefix']+f'.batch{j}.GT_row_index'];gt=data[f'target.{j}.boxes'][g];ids=data[f'target.{j}.verified_source_annotation_ids'][g];short=gt[:,2:].astype(np.float64).min(1)*size;n=len(data[f'target.{j}.boxes']);trajectory=[]
          for stage,label in zip(STAGES,NAMES):
            allpred=data[stage+'.pred_boxes'][j];assert allpred.shape==(300,4);pred=allpred[q];iou=iou_pairs(pred,gt);center_oracle=pred.copy();center_oracle[:,:2]=gt[:,:2];size_oracle=pred.copy();size_oracle[:,2:]=gt[:,2:];ciou=iou_pairs(center_oracle,gt);siou=iou_pairs(size_oracle,gt);max_iou=best_iou(allpred,gt);assert (max_iou+1e-7>=iou).all();ce=np.linalg.norm((pred[:,:2]-gt[:,:2]).astype(np.float64)*size,axis=1);ratio=pred[:,2:].astype(np.float64)/gt[:,2:];trajectory.append(iou)
            for k in range(len(q)):rows.append(dict(seed=seed,endpoint=endpoint,batch=batch,item=j,query=int(q[k]),GT_row=int(g[k]),source_annotation_id=int(ids[k]),stage=stage,stage_meaning=label,GT_count_in_composite=n,dense_gt300=n>300,input_short_pixels=float(short[k]),input_size_bin='lt4' if short[k]<4 else '4to8' if short[k]<8 else '8to16' if short[k]<16 else 'ge16',IoU=float(iou[k]),center_GT_oracle_IoU=float(ciou[k]),size_GT_oracle_IoU=float(siou[k]),any_query_max_IoU=float(max_iou[k]),center_error_pixels=float(ce[k]),center_error_relative_short=float(ce[k]/short[k]),width_ratio=float(ratio[k,0]),height_ratio=float(ratio[k,1])))
          for threshold in [.5,.75]:
            for k in range(len(q)):transitions.append(dict(seed=seed,endpoint=endpoint,batch=batch,item=j,query=int(q[k]),GT_row=int(g[k]),dense_gt300=n>300,input_size_bin='lt4' if short[k]<4 else '4to8' if short[k]<8 else '8to16' if short[k]<16 else 'ge16',threshold=threshold,encoder_qualifies=bool(trajectory[0][k]>=threshold),final_qualifies=bool(trajectory[-1][k]>=threshold),all_stages_below=bool(all(x[k]<threshold for x in trajectory))))
    for name,rs in [('per_fixed_query_stage_geometry.csv',rows),('per_fixed_query_encoder_final_crossings.csv',transitions)]:
      with (target/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    primary=[];contrasts=[];joint=[];crossings=[]
    for seed in (1,2):
      for endpoint in (12,132):
        for size in ['lt4','4to8','8to16','ge16']:
          for stage in STAGES:
            subset=[r for r in rows if r['seed']==seed and r['endpoint']==endpoint and r['stage']==stage and not r['dense_gt300'] and r['input_size_bin']==size];result=dict(seed=seed,endpoint=endpoint,stage=stage,input_size_bin=size,GT_le300=True,**summarize(subset));contrasts.append(result)
            if endpoint==132 and size=='lt4':primary.append(result)
        selected=[r for r in rows if r['seed']==seed and r['endpoint']==endpoint and r['stage']=='model' and not r['dense_gt300'] and r['input_size_bin']=='lt4']
        for threshold in [.5,.75]:
          failing=[r for r in selected if r['IoU']<threshold]
          joint.append(dict(seed=seed,endpoint=endpoint,threshold=threshold,final_failing=len(failing),center_only=sum(r['center_GT_oracle_IoU']>=threshold and r['size_GT_oracle_IoU']<threshold for r in failing),size_only=sum(r['size_GT_oracle_IoU']>=threshold and r['center_GT_oracle_IoU']<threshold for r in failing),either=sum(r['center_GT_oracle_IoU']>=threshold and r['size_GT_oracle_IoU']>=threshold for r in failing),neither=sum(r['center_GT_oracle_IoU']<threshold and r['size_GT_oracle_IoU']<threshold for r in failing)))
          tr=[r for r in transitions if r['seed']==seed and r['endpoint']==endpoint and r['threshold']==threshold and not r['dense_gt300'] and r['input_size_bin']=='lt4'];crossings.append(dict(seed=seed,endpoint=endpoint,threshold=threshold,rows=len(tr),encoder_and_final=sum(r['encoder_qualifies'] and r['final_qualifies'] for r in tr),encoder_only=sum(r['encoder_qualifies'] and not r['final_qualifies'] for r in tr),final_only=sum(not r['encoder_qualifies'] and r['final_qualifies'] for r in tr),neither=sum(not r['encoder_qualifies'] and not r['final_qualifies'] for r in tr),all_stages_below=sum(r['all_stages_below'] for r in tr)))
    # Verify exact same final-pair IoU against preceding native-autograd data.
    max_error=0.;checked=0
    lookup={(r['seed'],r['endpoint'],r['batch'],r['item'],r['query'],r['GT_row']):r['IoU'] for r in rows if r['stage']=='model'}
    for seed in (1,2):
      with np.load(OUT/f'collection_seed{seed}/native_verification/matched_native_gradients.npz') as native:
       for endpoint in (12,132):
        for batch in range(16):
          p=INPUT/f'collection_seed{seed}/capture_v1/endpoint{endpoint}/batch_{batch:03d}';ev=read(p/'evidence.json');event=next(e for e in ev['events'] if e['kind']=='loss_call' and e['loss']=='mal' and e['stage']=='model' and not e['is_DN'])
          with np.load(p/'arrays.npz') as a:items=np.concatenate([np.full(len(a[event['prefix']+f'.batch{j}.query_index']),j) for j in range(4)])
          key=f'seed{seed}.endpoint{endpoint}.batch{batch:03d}.stage0'
          for j,q,g,ii in zip(items,native[key+'.query'],native[key+'.GT_row'],native[key+'.IoU']):err=abs(lookup[seed,endpoint,batch,int(j),int(q),int(g)]-float(ii));max_error=max(max_error,err);checked+=1
    assert max_error==0
    summary={'status':'PASS_FIXED_FINAL_QUERY_LOCALIZATION_OUTPUT_TRACE','primary_unsaturated_tiny_mature':primary,'all_unsaturated_size_endpoint_stages':contrasts,'final_oracle_joint_categories':joint,'encoder_final_crossings':crossings,'fixed_pair_stage_rows':len(rows),'final_pair_IoUs_checked_against_native':checked,'maximum_native_final_IoU_error':max_error,'source_query_order_verified_by_code':True,'protocol_sha256':sha(OUT/'localization_followup_protocol.json'),'code_sha256':sha(Path(__file__)),'model_forwards_added':0,'model_backwards_added':0,'optimizer_steps':0,'epochs_added':0,'AP_evidence':False,'root_cause_established':False}
    save(target/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='all_unsaturated_size_endpoint_stages'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--analyze',action='store_true');a=p.parse_args();assert a.freeze!=a.analyze
    freeze() if a.freeze else analyze()
