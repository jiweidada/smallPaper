from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent; B=D.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):
    with (D/name).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
assert not (D/'semantic_protocol_FROZEN.json').exists()
for name in ['analytic_local_pre_freeze.json','analytic_remote_pre_freeze.json']:
    x=read(D/name);assert x['status']=='PASS_ANALYTIC_ONLY_NO_AUTHOR_MODEL' and x['author_detector_calls']==0
scaled=read(D/'analytic_scaled_remote_pre_freeze.json');assert scaled['status']=='PASS_SCALE_MATCHED200x200_CPU_ONLY'
preflight=read(D/'remote_input_preflight.json');assert preflight['asset_hashes_verified']==224 and preflight['author_detector_calls']==0
assert not preflight['captures_exist'] and not preflight['forward_ledger_exists']
inputs=read(D/'collector_inputs.json');assert len(inputs['cases'])==112
source_sample=read(B/'A37_局部差分编码与语义判别证据核查_v1'/'semantic_probe_sample_FROZEN.json')
assert [r['file'] for r in inputs['cases']]==[r['file'] for r in source_sample['cases']]
assert not (D/'downloaded_evidence').exists() and not (D/'analysis').exists()
log=Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文\前期调研_码率受限航拍小目标_20261028_v1\远端验证_20261029_v1\ssh_operation_receipts.jsonl')
runtime=[]
for line in log.open(encoding='utf-8'):
    try:r=json.loads(line)
    except ValueError:continue
    if r.get('id')=='A38_readonly_runtime_preflight_01' and r.get('action')=='exec':runtime.append(r)
assert len(runtime)==1 and runtime[0]['exit_code']==0
values=json.loads(runtime[0]['stdout'])
assert values['loader_sha256']=='8dc66cf817d387733a895501dc160dad703eedad9d061ce8f4ffd95296b7c1ab'
assert values['checkpoint_sha256']=='42874302fd97e17c2a23a46a8ec35d2ea4d2ffe479f0d632c1327fafbf910eb7'
save('runtime_preflight.json',dict(receipt=runtime[0],values=values))
tap=dict(source_commit='2dde3bc1946a3e9fad9abd0612b59fc39bd6b861',
    memory='DomeTransformer._get_encoder_input715-738 projects/flat-concats hybrid features; TransformerDecoder.forward376/396 receives memory and calls value_op with no additional projection; first40000tokens reshape to256x200x200.',
    query='Final native dec_score_head[2] input at TransformerDecoder.forward445-448, before Linear and LQE.',
    native_queries='Original PAQI250..500,three decoder layers; no fixed500 initializer override.',
    same_box='Both query-associated fine and contrast features use final predicted xyxy box,clipped to0..1,for all native queries. GT used only later for diagnostic matching.',
    dimensions={'memory':[1,53125,256],'first_fine':[1,256,200,200],'query':['1','N250..500',256]},
    original_size_order='Pinned author coco_dataset.py187 sets orig_size=[w,h]; custom postprocessor repeats[w,h,w,h].',
    original_code_modified=False,source_copies=str(D/'source_pin_and_handoff.json'),
    inference_scope='Existing task-adapted CNN-based Dome-S; not new compactViT training or proof of a ViT-specific defect.')
save('native_source_tap_audit.json',tap)
draws=np.random.default_rng(381031).integers(0,48,size=(1000,48),dtype=np.int64)
with (D/'bootstrap_draws_FROZEN.npz').open('xb') as f:np.savez_compressed(f,draws=draws)
protocol=dict(phase='A38',recorded_utc=datetime.now(timezone.utc).isoformat(),
    status='FROZEN_BEFORE_ANY_NEW_AUTHOR_MODEL_FORWARD',
    prior_turn='Progress:A37addednewclosestforeground-probingpriorand25authorblobs;mechanismstilluntested.',
    question='Does same-predicted-box post-fusion fine-memory evidence make native-unsupported small foreground more linearly accessible than final query features,after matched-background and limited-capacity controls?',
    source_commit=tap['source_commit'],loader_path='/root/c10_region_restore_20261029_v1/architecture_probe_v1/dome_assets_v1/dome_probe_loader.py',
    loader_sha256=values['loader_sha256'],checkpoint_path='/root/c10_region_restore_20261029_v1/architecture_probe_v1/dome_assets_v1/dome-s-visdrone_converted.pth',
    checkpoint_sha256=values['checkpoint_sha256'],host='OriginalundeployedFP32Dome-SHGNetv2B0+hybridencoder;strictstate,eval,frozen;native250..500queries;800x800stretchauthortransform.',
    sample_manifest_sha256=sha(B/'A37_局部差分编码与语义判别证据核查_v1'/'semantic_probe_sample_FROZEN.json'),
    sample={'train':64,'val':48,'selection':'A37fixedoutcome-blindoneimageperfilename-prefixgroup;trainexcludesallofficialvalprefixes;valexcludesold12prefixes.',
        'blind_limit':'New to this diagnostic and held out from probe fitting;not asserted globally unseen across legacy work or checkpoint selection.'},
    representations=['QUERY256','FINE_same_predicted_box7x7mean256','CONTRAST_inner_minus15x15outer2xring256_secondary_known_control',
                     'QUERY_RFF512_fixed_nonlinear_capacity_control','GEOMETRY5_logarea_logaspect_maxvalidnativescore_centerx_centery'],
    interpolation={'align_corners':False,'mode':'bilinear','padding':'border','outer_points_outside_image':'excluded from ring average',
        'first_control_independent_numpy_atol':5e-4,'first_control_independent_numpy_rtol':2e-5},
    annotations='Raw8columnVisDrone;trueforegroundclasses1..11forbackgroundexclusion. Score>0andignore-unionfraction<0.5forgeometryassignment;class0orscore<=0rectanglesformignoreunion.',
    primary_GT='Classes1..10;originalannotationarea<=1024;truncation0;occlusion0;validnonignored;one-to-onegeometryassignmentIoU>=0.5.',
    assignment='Deterministic maximum-cardinality augmenting paths in annotation order; candidateedges prioritizedbydescendingIoUthenqueryid. Not a global max-IoU-weight claim.',
    native_support='OriginalflatTopN12classpairspostprocessN=nativequerycount,no refill;correctclass1..10,score>=0.5,queryignorefraction<0.5,IoU>=0.5. PrimaryvalGTlacksanysuchsupport;notallFP-budgetfalse-negatives.',
    background='Queryboxnonzero,ignoreunionfraction<0.5,IoU<0.1toeveryrealannotationclass1..11regardlessvisibility;notunmatched-target-as-background.',
    pair_matching={'one_background_per_positive':True,'same_image':True,'max_cardinality':True,
        'predicted_input_sqrtarea_bins':[8,16,32,64,128],'log2aspect_bins':[-1,0,1],
        'max_valid_native_score_bin_width':.05,'absolute_score_caliper':.025,
        'priority':'Absolute score difference,log-area difference,log-aspect difference,queryid.',
        'selection':'Match all visible small foreground first,then primaryval subset by fixed native-support absence.'},
    fitting='Ridge on64trainonly;train-onlymean/std floor1e-6;sum-squared objective withlambda1,unpenalizedintercept;balanced1positive1backgroundpairs;y=-1/+1. No hyperparameter search.',
    rff_seed=381032,rff='Train-standardizedquery;GaussianOmega256x512/sqrt256,uniformphase;cos*sqrt(2/512),thenowntrain-onlystandardization;lambda1ridge.',
    class_check='Two positive-only10classridgefits QUERY/FINE;reportallclasses andmacroforclasseswith>=10train and>=5primaryvalpositives;secondary,notselectedto rescue primary.',
    primary_metric='AUC(FINE)-AUC(QUERY)onmatchednative-unsupportedsmallforeground/backgroundvalpairs.',
    bootstrap={'draws':1000,'seed':381031,'unit':'48valimageswithreplacement,literalrepeatedrecords','refit':False,'invalid':'PreserveNaN,no replacement;need>=990validdraws.','CI':'Percentile2.5/97.5ofpairedAUCdifference.'},
    eligibility={'minimum_train_pairs':200,'minimum_train_images':32,'minimum_primary_val_pairs':100,'minimum_primary_val_images':20,'on_failure':'No data probes fitted;do not expand/resample automatically.'},
    advancement_gates={'fine_auc_min':.75,'fine_minus_query_min':.05,'paired_CI_lower_strictly_positive':True,
        'fine_minus_query_RFF_min':.02,'fine_minus_geometry_min':.05,'valid_bootstrap_min':990},
    gate_scope='Only decides whether this fine-memory/query accessibility mechanism merits a concrete readout control;passing is NOT a mainline,novelty,algorithm improvement or causal-information-erasure result.',
    secondary_results='CONTRAST,all-matched-visible-smallAUC,andtrue-classchecksareinterpretive;no choose-best rescue.',
    caps={'whole_detector_entries':114,'complete_expected':114,'composition':'1hook-freebaseline+112instrumentedcases+1firstcase repeat',
        'data_probe_fit_entries':7,'detector_optimizer_updates':0,'new_training_jobs':0,'wall_seconds':900,'cuda_allocated_stop_bytes':20*1024**3},
    amendment_to_A37_draft='Pre-outcome addition of one hook-free native fidelity call:possible113draft becomes frozen114. A37allocated0;no budget was spent or retrospectively enlarged.',
    completed_analytic_preflights={'synthetic_ridge_fits':2,'author_detector_calls':0,'data_probe_fits':0,'scaled_pool_cpu_fixture':True},
    stops=['Any source/inputhash mismatch','Nonfinite or wrongshapecapture','Hook/native or repeated-inputfidelityfailure','20GiBallocatedguard','Entrycap114','900secondworker timeout'],
    continuation='No automatic second seed,extra query/threshold/size grid,model restart,phase rerun,oldA22/DEIM restart or compression work.',
    interpretation_limits=['Fixedlinear/RFFaccessibilitynotallpossibledecoders','Post-fusionmemorynotrawbackbone','OracleGTdiagnosticmatchingnotdeployabletesttimeprivilege','AUCnotdetectorAPorlowFPgain','Checkpointselection/globalprojectblindnessnotproven','ClosestUnmaskingTinyfullmethodnotclosed'],
    new_algorithm_approved=None,mainline_validated=False)
save('semantic_protocol_FROZEN.json',protocol)
names=['semantic_math.py','pool_tensor.py','analytic_fixtures.py','scaled_pool_fixture.py','collect_features.py',
       'analyze_features.py','launch_collector.py','run_and_record.py','prepare_remote_inputs.py',
       'collector_inputs.json','semantic_protocol_FROZEN.json','bootstrap_draws_FROZEN.npz']
for name in names:
    if name.endswith('.py'):ast.parse((D/name).read_text(encoding='utf-8'),feature_version=(3,9))
save('execution_inputs_manifest.json',dict(recorded_utc=protocol['recorded_utc'],files_sha256={n:sha(D/n) for n in names},
    cap=114,data_probe_fit_cap=7,optimizer_update_cap=0))
print(json.dumps(dict(status='A38_FROZEN_BEFORE_MODEL',whole_detector_entry_cap=114,data_probe_fit_cap=7,
    protocol_sha256=sha(D/'semantic_protocol_FROZEN.json'),manifest_sha256=sha(D/'execution_inputs_manifest.json')),ensure_ascii=False))
