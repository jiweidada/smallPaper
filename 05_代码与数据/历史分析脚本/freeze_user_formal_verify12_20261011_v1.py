"""Freeze a readonly formal reader and reject malformed savedEMA engineering proofs."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json
ROOT=Path.cwd();code=ROOT/'英文小论文/code/user_backbone_formal_verify12_20261011_v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ast.parse((code/'verify.py').read_text(encoding='utf8'))
names=['英文小论文/code/user_backbone_primary_collection12_20261011_v2/verify_primary.py','英文小论文/code/user_backbone_primary_collection12_20261011_v2/protocol.json','英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py','英文小论文/code/stem_init_stage_analysis_20261011_v2/protocol.json']
spec=dict(schema='user_fixed12_saved_EMA_formal_reader_v1',reader_sha256=sha(code/'verify.py'),dependencies_sha256={n:sha(ROOT/n) for n in names},original_user28_sources=json.loads((ROOT/'英文小论文/code/user_backbone_primary_collection12_20261011_v2/protocol.json').read_bytes())['user_files_sha256'],allowed_arms=['p2_skip','backbone_silu','backbone_attention'],allowed_seeds=[1,2],no_models_or_evaluators=True,new_training_epochs=0)
with (code/'protocol.json').open('x',encoding='utf8') as f:json.dump(spec,f,indent=2)
s=importlib.util.spec_from_file_location('formal_user_proof_engineering',code/'verify.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
identity=dict(seed=2,arm='p2_skip');done=dict(last_checkpoint_sha256='checkpoint',identity=identity,completed_epochs=12,entered_epochs=list(range(12)),status='NATIVE_FIT_FINISHED_EXACT12');init=dict(state_tensors=801,parameters=10249437);closeout=dict(stats_fraction_tolerance=0)
proof=dict(status='PASS_SAVED_CANDIDATE_RELOAD',identity=identity,checkpoint_sha256='checkpoint',strict_loaded_tensors=801,parameters=10249437,all_parameters_frozen_like_native_EMA=True,native_GT_hits_exact=True,all548_stage_arrays_exact=True,stats_max_abs_fraction_difference=0,stats_tolerance=0,images=548,candidate_model_forwards=137,original_stats=[0.1],reloaded_stats=[0.1],stage_max_abs_difference={'x':0},baseline_model_forwards=0,optimizer_updates=0,epochs_added=0,automatic_retry=False)
exit=dict(exit_code=0,timed_out=False,automatic_retry=False,baseline_model_forwards=0,epochs_added=0)
m.assert_replay(proof,exit,identity,done,init,closeout)
negatives=[]
for field,value in [('status','FAILED'),('checkpoint_sha256','wrong'),('strict_loaded_tensors',794),('parameters',10231997),('all548_stage_arrays_exact',False),('native_GT_hits_exact',False),('stats_max_abs_fraction_difference',1e-8),('baseline_model_forwards',137),('candidate_model_forwards',138),('automatic_retry',True)]:
    bad=copy.deepcopy(proof);bad[field]=value
    try:m.assert_replay(bad,exit,identity,done,init,closeout)
    except AssertionError:negatives.append(field)
    else:raise AssertionError('Malformed proof accepted: '+field)
bad=copy.deepcopy(done);bad['completed_epochs']=11
try:m.assert_replay(proof,exit,identity,bad,init,closeout)
except AssertionError:negatives.append('wrong_final_epoch')
else:raise AssertionError('Wrong final epoch accepted')
review=dict(status='PASS_FORMAL_READER_SOURCE_AND11_NEGATIVE_ENGINEERING_PROOF_CASES',negative_cases=negatives,synthetic_positive_cases=1,actual_formal_artifacts_verified=False,files={p.name:sha(p) for p in code.iterdir() if p.is_file()},source_sha256=sha(__file__),new_model_forwards=0,new_training_epochs=0)
with (ROOT/'outputs/user_formal_verify12_20261011_v1_source_review.json').open('x',encoding='utf8') as f:json.dump(review,f,indent=2)
print(review['status'])
