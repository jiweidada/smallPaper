"""Freeze a bounded paired dual-head diagnostic before any new model call."""
from pathlib import Path
import datetime,hashlib,json,sys
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parent;N=R.parent;E=N.parent
OLD=E/'前期调研_码率受限航拍小目标_20261028_v1'/'远端验证_20261029_v1'
RAW=OLD/'完成结果_20261029_v1'/'raw'
REMOTE='/root/c10_region_restore_20261029_v1'
DEST=REMOTE+'/architecture_probe_v1/a30_yolo26_dual_head_20261030_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
    assert not (R/'native_dual_head_protocol_FROZEN.json').exists()
    pin=read(R/'pinned_source_manifest.json');assert pin['host5_sources_match_pin']
    prior=read(N/'69_A29最近方法与缓存漏检核验回执.json')
    for p,h in prior['files_sha256'].items():assert sha(p)==h,p
    calls=[json.loads(x) for x in (OLD/'ssh_operation_receipts.jsonl').read_text(encoding='utf-8').splitlines() if 'A30_D0_assets_and_gpu_readonly_v1' in x]
    call=next(x for x in reversed(calls) if x.get('action')=='exec');assert call['exit_code']==0;host=json.loads(call['stdout'])
    for name,h in host['files'].items():assert sha(RAW/name)==h,name
    oldfiles={x['file_name']:x for x in read(OLD/'dataset_source_manifest.json')['files']}
    meta=read(RAW/'receipts/diagnostic_observations.json')['images'];assert len(meta)==100
    assert set(host['image_sha256'])=={x['file_name'] for x in meta}
    cases=[]
    for index,x in enumerate(meta):
        expected=oldfiles[x['file_name']]['images']['sha256'];assert host['image_sha256'][x['file_name']]==expected
        cases.append({'index':index,'image_id':x['image_id'],'file_name':x['file_name'],'path':x['source'],'sha256':expected,'geometry':x['geometry'],'width':x['width'],'height':x['height']})
    arms=[{'id':b+'box_'+s+'score_'+p,'boxes':b,'scores':s,'postprocess':p,'kind':'native endpoint' if (b,s,p) in [('M','M','NMS'),('O','O','TOPK')] else 'diagnostic counterfactual,not trained algorithm'} for b in ['M','O'] for s in ['M','O'] for p in ['NMS','TOPK']]
    protocol={'phase':'A30','status':'FROZEN_BEFORE_NEW_MODEL_EXECUTION','recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'question':'Does the observed small-object low-score support survive native one2one readout;how do box head,score head,and postprocessing contribute with shared features fixed?',
      'purpose':'Exploratory mechanism diagnosis of an already trained reference;not an architecture efficacy or novelty confirmation.',
      'remote_root':REMOTE,'remote_output':DEST,'checkpoint':{'path':REMOTE+'/results/D0/weights/last.pt','sha256':host['files']['results/D0/weights/last.pt'],'selection':'existing fixed100 EMA;no weight updates'},
      'source_pin':pin['commit'],'source_files':pin['files'],'host_package':read(R/'host_preflight_readonly.json')['decoded']['package'],
      'remote_asset_sha256':host['files'],'local_raw_root':str(RAW),'helper_functions':['prepare_pixels','detection_records','ground_truth_input','coco_eval','match_summary','bootstrap_ap'],'helper_loading':'Compile only listed AST FunctionDef nodes from hash-verified historical evaluate_fixed.py;never import its experiment body or old queue.',
      'versions':host['versions'],'cases':cases,'repeat_case_index':0,'fidelity_gate_cases':[0,1,2,3],
      'call_caps':{'whole_detector_forwards':101,'standalone_native_TAL_forward_calls':200,'extra_native_head_inference_decodes':200,'new_training_jobs':0,'optimizer_updates':0,'wall_seconds':1800},
      'execution':'One unfused FP32 eval full forward per source image with end2endFalse returns both raw heads;native head decode supplies the other branch. Final repeat of case0 uses end2endTrue and must reproduce cached raw heads and reconstructed native one2one output. No restarts,no auto followup queue.',
      'host_requirements':{'nc':10,'reg_max':1,'strides':[8,16,32],'grid_shapes':[[80,80],[40,40],[20,20]],'anchors':8400,'require_both_heads':True,'CUDA_device':0,'GPU_max_used_MiB_before_start':1024,'no_other_compute_processes':True},
      'fidelity':{'old_source_prediction_score_floor':.05,'count_and_category_match':True,'box_abs_tolerance_input_pixels':.05,'score_abs_tolerance':2e-5,'matching':'Hungarian within each category;no order assumption or threshold relaxation','fresh_one2many_vs_cached_AP_abs_tolerance':1e-4,'repeat_raw_and_native_one2one_output_bitwise_equal':True,'state_dict_digest_unchanged':True,'checkpoint_and_source_hash_unchanged':True},
      'arms':arms,'postprocessing':{'NMS':{'conf':.001,'iou':.7,'multi_label':True,'agnostic':False,'nc':10,'max_det':500,'native_return_idxs':True},'TOPK':'Native get_topk_index withmax_det500,agnosticFalse,exportFalse;gatherxyxy and filter scores>=.001;final native end2end repeat verifies equivalence.','common':'Same original-image clipping and ratio/pad mapping from historical helper,all10 classes,sourceJPEG only.'},
      'raw_capture':'Per image both logits,regression,decodedxyxy,anchors/strides,GT and native TAL target/fg/GTidx arrays. Shared neck features saved only for first4 fidelity cases. No training augmentation,GT modification,feature intervention or oracle deployment scores.',
      'assignment_scope':'Use actual E2ELoss one2many/one2one assigners from frozen model on final-checkpoint outputs and noncrowdGT. This does not reconstruct historical training assignments or establish supervision starvation. Native overlap is nonnegative CIoU,not AP IoU.',
      'metrics':{'primary':'Input-area smallAP50:95,strict area<1024,internalCOCO101,maxDet500,identicalGT/ignore semantics to D0;not officialVisDrone score.','native_endpoints':['Mbox_Mscore_NMS','Obox_Oscore_TOPK'],'factorial':'All8 box/score/postprocessing combinations reported;no best-arm promotion.','same_FP_budgets':[300,1200,3000],'same_FP_primary':1200,'same_FP_method':'IoU0.5,all-area matching records,exclude dtIgnore;whole equal-score tie groups;maximal prefix of one global score threshold withFP<=budget. Report actualFP and unique smallTP.','score_thresholds':[.05,.1,.25,.5],'bootstrap':{'draws':1000,'seed':330103,'unit':'paired image on same100 consumed development images','endpoint':'nativeO-minus-nativeM inputsmallAP','scope':'Checkpoint-conditional image uncertainty only;relatedframes and trainingseeds not covered.'}},
      'scientific_decision':'No new algorithm is approved by a head comparison or factorial score/box swap. If phenomenon is mainly branch or postprocessing dependent,revise its scope before proposing encoder/decoder changes. If it persists,require a distinct operator and new intervention evidence beyond generic TOOD/GFL alignment.',
      'stop_conditions':['Any checkpoint/source/image byte mismatch','Wrongmodel/classes/regmax/grid/branch','Fidelity failure on any image or APpreflight','Nonfinite tensors or changed modelstate','Call cap,wallcap,OOM or runtime error;noautomaticretry'],
      'prior_constraints':'A22/oldDEIM/compression/restoration remain terminal;prior closed architecture forward budget not reused. This separate bounded diagnostic falls within existing remote research authorization.'}
    save(R/'assets_and_gpu_readonly.json',{'call':call,'decoded':host})
    save(R/'native_dual_head_protocol_FROZEN.json',protocol)
    print(json.dumps({'status':'FROZEN_DUAL_HEAD_AND_2x2x2_READOUT_DIAGNOSTIC','source_images_verified':100,'model_forward_cap':101,'TAL_call_cap':200,'new_training_jobs':0,'old_A29_files_reverified':len(prior['files_sha256']),'gpu_snapshot':host['gpu']}))
if __name__=='__main__':main()
