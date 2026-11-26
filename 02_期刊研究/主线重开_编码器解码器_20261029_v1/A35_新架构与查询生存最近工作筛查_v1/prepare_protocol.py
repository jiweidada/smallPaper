from pathlib import Path
import ast, datetime, hashlib, json, shutil, subprocess, sys, zipfile
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;B=D.parent
OLD=B/'A27_查询扩容同预算任务诊断_v1'
REMOTE='/root/c10_region_restore_20261029_v1/architecture_probe_v1/a35_query_prefix_cpu_20261031_v1'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)

prior=read(B/'79_A34剩余76图独立复核回执.json')
for name,h in prior['files_sha256'].items():assert sha(name)==h,name
assert len(prior['files_sha256'])==603
old=read(OLD/'task_protocol_FROZEN.json');reference=read(B/'65_A27任务指标与候选判定核验回执.json')
verified={}
for name in ['o27_metrics.py','raw_annotations_verified.json']:
    p=OLD/name;expected=next(x['sha256'] for x in old['files'] if x['path']==name)
    assert sha(p)==expected;target=D/('pinned_o27_metrics.py' if name=='o27_metrics.py' else name)
    assert not target.exists();shutil.copy2(p,target);verified[str(p)]=expected
tree=ast.parse((D/'pinned_o27_metrics.py').read_text(encoding='utf-8'))
defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['matlab_round','ignore_integral','region_keep','prepare_cases']]
ns={'np':np};exec(compile(ast.Module(body=defs,type_ignores=[]),'pinned_annotation_preflight','exec'),ns)
annotated=ns['prepare_cases'](read(D/'raw_annotations_verified.json')['cases'])
assert len(annotated)==12 and all(len(c['gt_xywh'])<=200 for c in annotated)
save(D/'annotation_only_eligibility_before_prefix_outcomes.json',dict(
    status='ALL12_CONTEXTS_HAVE_VALID_GT_COUNT_LE200',not_selected_on_prefix_outputs=True,
    rows=[dict(file=c['file'],valid_GT=len(c['gt_xywh']),original_area_small_GT=int(c['gt_small'].sum())) for c in annotated]))
cases=[]
with zipfile.ZipFile(D/'native500_reference.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for c in old['cases']:
        p=OLD/'verified_task_evidence/predictions'/('case%02d_NATIVE500.npz'%c['index'])
        assert sha(p)==reference['files_sha256'][str(p)]
        verified[str(p)]=sha(p);entry='case%02d_NATIVE500.npz'%c['index'];z.write(p,entry)
        cases.append(dict(c,reference_entry=entry,reference_sha256=sha(p)))
seed=351031;rng=np.random.default_rng(seed)
draws=rng.integers(0,12,size=(1000,12));w=np.column_stack([np.ones(12,dtype=np.int64)]+[np.bincount(x,minlength=12) for x in draws])
np.savez_compressed(D/'bootstrap_draws_FROZEN.npz',weights=w,draws=draws)
fixture=subprocess.run([sys.executable,str(D/'query_retention_analysis.py'),'--fixture'],check=True,capture_output=True,text=True,encoding='utf-8')
save(D/'metric_fixture_receipt.json',json.loads(fixture.stdout))
cfg=dict(
 status='FROZEN_NATIVE_PREFIX_SUPPORT_DIAGNOSIS',phase='A35',recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 question='With enough slots for the annotated objects, does first-layer score-only query truncation disproportionately remove the native final detection support of small objects?',
 prior_goal_turn='A34 progress: rejected the coarse-cell separation directional explanation in remaining76-image replication.',
 remote_output=REMOTE,loader=old['loader'],state_digest=old['state_digest'],checkpoint_sha256=old['checkpoint_sha256'],
 source_commit=old['source_commit'],dependencies=old['dependencies'][:3],cases=cases,
 host='Dome-DETR-S author VisDrone weights, undeployed FP32 eval decoder; fixed Top500 query initialization from cached native image memory.',
 same_as_full_Dome_pipeline=False,checkpoint_early_heads_verified_available=True,
 execution=dict(device='CPU',threads=4,dtype='float32',native_eval_idx_values=[2,0,1],call_order_per_case=[3,1,2],
                copied_decoder_numerical_code=False,train_mode=False,full_image_forward=False,new_GPU_calls=0),
 caps=dict(decoder_entries=37,complete_three_layer_calls=13,partial_prefix_calls=24,decoder_layer_entries=75,
           whole_detector_forwards=0,image_encoder_forwards=0,optimizer_updates=0,wall_seconds=600),
 eligibility=dict(case_selection='Exactly the12 cached official-val contexts already used inA27; no selection on new early scores.',
                  max_valid_GT_per_image=200,retained_slots=250,initial_slots=500,
                  why='Leave at least50 slots beyond annotated object count to avoid a trivial query-count bottleneck.',
                  not_independent_blind_sample=True),
 metrics=dict(primary_boundary='After decoder layer1; layer2 descriptive only.',
              early_rank='Maximum sigmoid probability across valid labels1..10; stable descending sort with ascending original query index for ties.',
              retained_queries=250,small='Original annotated area<=1024, inherited fromA27; not network-input small or the old phase exposure.',
              postprocess='Original pinnedA27 torch sigmoid/top500 class-pairs with valid label1..10, then source-aligned ignore-region filter; no refill.',
              global_FP_budget=111,FP_budget_reason='floor(9.26*12),same preexisting per-image diagnostic rate;fixed before new prefix outcomes.',
              native_matching='Score-ordered class-aware IoU>=0.5 unique matching to all validGT; global complete equal-score blocks.',
              attrition='Among native final TP GT accepted at the global FP prefix, no accepted class-correct IoU>=0.5 final query remains in earlyTop250. Backups prevent attrition.',
              adjusted='Overlap-weighted difference within image,class,final-native-scorebin; predictive adjustment,not causal.',
              confidence_bins=[0,.1,.25,.5,.75,1],bootstrap_draws=1000,bootstrap_seed=seed,
              bootstrap='Image-cluster paired draws; re-estimate global native FP prefix for each draw. Same early masks,unchanged native finalstates.',
              all_required_question_gates=dict(sample_support='Both groups>=50 nativeTP in>=6images.',small_attrition='>=0.05',
                   small_minus_larger='>=0.03',CI='95% lower>0 and>=950validdraws',adjusted='>=0 with>=20GT each group in>=6images of common support.')),
 seed=seed,interpretation_limits=[
  'Support attrition from unchanged native final outputs is not the AP/recall effect of actual pruned decoding.',
  'Original-area groups can depend on image resolution; do not claim scale causality or encoder information loss.',
  'Only12 reused development contexts; image bootstrap does not establish sequence-independent or external generalization.',
  'Generic routing,bypass,re-entry,quality scores and density budgeting already have priors; no novel algorithm approved.',
  'Passing this screen only justifies a fixed-budget actual-pruned-decoder followup; no training,model change or new experiment queue is automatic.',
  'Failure does not exclude every query utility method; do not tune masks,budgets,bins or cohorts to rescue this screen.'
 ],historical_caps_remain_closed=True,automatic_followups=False,new_algorithm_approved=None)
save(D/'native_prefix_protocol_FROZEN.json',cfg)
remote_files=['native_query_prefix_worker.py','native_prefix_protocol_FROZEN.json','pinned_o27_metrics.py','native500_reference.zip']
save(D/'execution_inputs_manifest.json',dict(remote_files_sha256={n:sha(D/n) for n in remote_files},
         analysis_sha256=sha(D/'query_retention_analysis.py'),bootstrap_sha256=sha(D/'bootstrap_draws_FROZEN.npz'),
         archive_verifier_sha256=sha(D/'verify_collected_archive.py'),
         independent_verifier_sha256=sha(D/'verify_retention_scalar.py'),
         pre_model_source_audit_sha256=sha(D/'pre_model_source_audit.json')))
save(D/'prior_A27_reference_subset_verified.json',dict(status='PASS_PINNED_METRIC_ANNOTATIONS_AND12_NATIVE_REFERENCES',files_sha256=verified))
print(json.dumps(dict(status='FROZEN_BEFORE_NEW_PREFIX_OUTCOMES',decoder_entries_cap=37,whole_detector_cap=0,optimizer_cap=0,
                      protocol_sha256=sha(D/'native_prefix_protocol_FROZEN.json'),analysis_sha256=sha(D/'query_retention_analysis.py'))))
