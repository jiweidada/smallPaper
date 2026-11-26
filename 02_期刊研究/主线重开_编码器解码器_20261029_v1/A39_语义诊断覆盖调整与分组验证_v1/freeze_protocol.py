from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,sys,zipfile
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;OLD=D.parent/'A38_细尺度与查询语义成对诊断_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):
    with (D/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
assert not (D/'semantic_protocol_FROZEN.json').exists() and not (D/'analysis').exists()
pre=read(D/'remote_input_preflight.json');assert pre['new_author_model_calls']==0 and pre['asset_hashes_verified']==224
identity=read(D/'handoff_and_sample_receipt.json')['model_identity']
assert pre['loader_sha256']==identity['loader_sha256'] and pre['checkpoint_sha256']==identity['checkpoint_sha256']
assert read(D/'group_bootstrap_fixture_result.json')['status']=='PASS_GROUP_LITERAL_DUPLICATION_HAND_AUC_AND_SCALAR_JSON'
assert read(OLD/'analysis/semantic_analysis_result.json')['probe_fit_entries']==0
cases=read(D/'combined_cases_FROZEN.json');inputs=read(D/'collector_inputs.json')
assert cases['counts']=={'train':119,'val':105} and len(inputs['cases'])==112
assert len(read(D/'bootstrap_groups_FROZEN.json')['groups'])==64
assert np.load(D/'bootstrap_draws_FROZEN.npz',allow_pickle=False)['draws'].shape==(1000,64)
old=read(OLD/'semantic_protocol_FROZEN.json')
for n in ['semantic_math.py','pool_tensor.py','npz_export.py']:assert sha(D/n)==sha(OLD/n)
protocol=dict(phase='A39',recorded_utc=datetime.now(timezone.utc).isoformat(),
    status='FROZEN_BEFORE_ADDITIONAL_MODEL_FORWARD_AND_ANY_DATA_PROBE_FIT',
    prior_goal_turn='Progress:A38collected112cacheswith114frozenentries;independentverificationfound78primarypairs<100;no semantic effect or data-probe fit.',
    question=old['question'],source_commit=identity['commit'],loader_path=old['loader_path'],loader_sha256=identity['loader_sha256'],
    checkpoint_path=old['checkpoint_path'],checkpoint_sha256=identity['checkpoint_sha256'],state_digest=identity['state_digest'],host=old['host'],
    adjustment=dict(kind='Prospectivecount-onlyenrollmentadjustmentbeforeanyproberesults',
        source_plan=str(OLD/'coverage_followup_sample_DRAFT.json'),source_plan_sha256=sha(OLD/'coverage_followup_sample_DRAFT.json'),
        all_frames=224,train_frames=119,val_frames=105,train_prefix_groups=64,val_prefix_groups=64,
        selection='Upto2SHA256-rankedframesperprefix;same64trainprefixes;all64eligiblevalprefixes;singletonsretainone;no label/feature-score selection.',
        old_caches_reused=112,new_train=55,new_val=57,old_A38_cap114_closed=True,old_frames_rerun=0,
        validation_blindness='Heldoutfromprobe fitting,not globallyblindacrosslegacyresearch/checkpointselection;filenameprefixisnotverifiedflightidentity.'),
    sample_manifest_sha256=sha(D/'combined_cases_FROZEN.json'),representations=old['representations'],interpolation=old['interpolation'],
    annotations=old['annotations'],primary_GT=old['primary_GT'],assignment=old['assignment'],native_support=old['native_support'],
    background=old['background'],pair_matching=old['pair_matching'],fitting=old['fitting'],
    rff_seed=old['rff_seed'],rff=old['rff'],class_check=old['class_check'],primary_metric=old['primary_metric'],
    bootstrap=dict(draws=1000,seed=381031,unit='64validationfilename-prefixproxygroups,allframes/primarypairswithliteralgrouprepeats',
        pooled_metric='PooledpairAUC,notequalmeanofgroupAUC',refit=False,invalid='PreserveNaN,no replacements;need>=990valid.',CI='Percentile2.5/97.5pairedAUCdifference.'),
    eligibility=dict(minimum_train_pairs=200,minimum_train_prefixes_with_pairs=32,minimum_primary_val_pairs=100,
        minimum_val_prefixes_with_primary_pairs=20,on_failure='No data probes;no automaticthirdframeexpansionorlowerthreshold.'),
    advancement_gates=old['advancement_gates'],gate_scope=old['gate_scope'],secondary_results=old['secondary_results'],
    caps=dict(whole_detector_entries=113,unique_new_cases=112,repeat_first_new=1,data_probe_fit_entries=7,
        detector_optimizer_updates=0,new_training_jobs=0,wall_seconds=900,cuda_allocated_stop_bytes=20*1024**3),
    fidelity=dict(numerical_helpers_and_feature_block='AST/textidenticaltoA38',new_full_feature_repeat='Allarraysofpredefinedfirstnewcasecomparedafter112newcases',
        old_A38_scope='Oldfirstqueryfeaturefullrepetitionunavailable;oldnative/fine-mapidentityandall112Linearreconstructionsverified,carriedforwardasdeclaredlimitation.'),
    stops=old['stops'][:-2]+['Entrycap113','900secondworkertimeout','No automaticrestartorresampling'],
    continuation='Singleboundednewcollection;old114capclosed;no automaticseed/threshold/view/modelgrid,training,oldDEIM/A22restartorcompression.',
    interpretation_limits=old['interpretation_limits'],new_algorithm_approved=None,mainline_validated=False)
save('semantic_protocol_FROZEN.json',protocol)
names=['semantic_math.py','pool_tensor.py','npz_export.py','collect_features.py','analyze_features.py','launch_collector.py','run_and_record.py',
    'collector_inputs.json','combined_cases_FROZEN.json','bootstrap_groups_FROZEN.json','bootstrap_draws_FROZEN.npz','semantic_protocol_FROZEN.json',
    'native_operator_preservation.json','group_analysis_preservation.json']
for name in names:
    if name.endswith('.py'):ast.parse((D/name).read_text(encoding='utf-8'),feature_version=(3,9))
save('execution_inputs_manifest.json',dict(recorded_utc=protocol['recorded_utc'],files_sha256={n:sha(D/n) for n in names},
    cap=113,data_probe_fit_cap=7,optimizer_update_cap=0,old_A38_cap_closed=True))
bundle=D/'frozen_execution_bundle.zip'
with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
    for name in names+['execution_inputs_manifest.json']:z.write(D/name,name)
print(json.dumps(dict(status='A39_FROZEN_BEFORE_NEW_MODEL_OR_DATA_FIT',protocol_sha256=sha(D/'semantic_protocol_FROZEN.json'),
    manifest_sha256=sha(D/'execution_inputs_manifest.json'),bundle_sha256=sha(bundle),bundle_bytes=bundle.stat().st_size,
    new_model_cap=113,data_probe_fit_cap=7,old_A38_cap_closed=True),ensure_ascii=False))
