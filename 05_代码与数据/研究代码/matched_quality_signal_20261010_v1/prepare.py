"""Freeze the archived-output diagnostic before deriving its outcomes."""
from pathlib import Path
import datetime, hashlib, json

CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[2]
OUT=ROOT/'outputs/matched_quality_signal_20261010_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    previous=ROOT/'outputs/native_augmented_supervision_20261010_v1'
    protocol={
      'status':'FROZEN_BEFORE_DERIVED_RESULTS',
      'frozen_local_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'material_passport':{'workflow':'experiment-agent','mode':'run+validate','scope':'archived native normal-head MAL signals; observational, no AP experiment'},
      'question':'Does native MAL suppress the correct-class margin of matched tiny targets, or intentionally calibrate low localization quality? Do well-localized weak-score matches receive positive semantic-margin signals?',
      'input_protocol_sha256':sha(previous/'protocol.json'),
      'archive_sha256':{str(s):sha(previous/f'collection_seed{s}.tar.gz') for s in (1,2)},
      'cohort':'All saved 16 augmented batches of4 per seed and endpoint, unchanged original fixture. All5 normal heads,320 native MAL calls. No sampling, rematching, new forwards or missing-call omission.',
      'seeds':[1,2],'endpoints':[12,132],
      'primary':'Final normal head at mature BASE132, GT count in its composite <=300, actual augmented input short side <4 pixels. Report each seed separately; each row is one target exposure, not an independent observation.',
      'contrasts':'Same final-head unsaturated composites, input short side4-8,8-16,>=16; same fixed targets BASE12. All other heads, dense/non-dense, MixUp, class and original source-area strata retained as descriptive secondary results.',
      'signals':{
        'p':'sigmoid of actual post-LQE native class logit',
        'target':'detached native pair IoU^1.5',
        'matched_GT_logit_descent_direction':'target - p_GT, normalized by native num_boxes for actual leaf derivative',
        'wrong_class_descent_direction':'-p_other^(1+gamma), detached background weight',
        'GT_minus_max_other_margin_descent_direction':'IoU^gamma - p_GT + p_max_other^(1+gamma), treating current largest other-class coordinate as fixed',
        'sign_epsilon_unnormalized':1e-7,
        'quality_bins':[0,.25,.5,.75,1],
        'well_localized_weak_score_control':'IoU>=.75 and p_GT<.05; inspect GT-logit and semantic-margin descent signs',
        'wrong_class_argmax':'maximum other logit strictly greater than GT logit',
        'localization':'Matched IoU>=.5 and>=.75 fractions; no detection TP/FP claim',
        'LQE':'Existing native LQE adds one shared scalar to all class logits and is differentiable through corner distributions. It cannot change their within-query logit differences by that scalar alone. No raw pre-LQE scores/corners archived here.'
      },
      'verification':{'raw_MAL_abs_tolerance':2e-5,'native_leaf_gradient_atol':2e-8,'native_leaf_gradient_rtol':2e-5,'IoU_abs_tolerance':2e-5,'FP32_CPU_vs_CUDA':'Report errors explicitly; no automatic relaxation/retry.'},
      'decision':'No fraction threshold is treated as an AP efficacy gate. GT score suppression at low IoU is baseline intentional quality calibration, not a bug. Claim against correct-margin supervision only with actual native derivative verification and both-seed consistency; even then frozen leaf gradients do not establish shared-parameter training causality. No training launched by this analysis. A new mechanism needs a separate prospective protocol and nearest-precedent comparison.',
      'remote_budget_per_seed':{'CPU_native_MAL_leaf_backward_calls':160,'max_model_forward_calls':0,'model_backward_calls':0,'optimizer_steps':0,'epochs':0,'hard_timeout_seconds':180,'automatic_retry':False},
      'limitations':['Frozen train-mode EMA FP32 fixtures are not historical training outputs/AMP gradients.','Normal MAL targets are stage-own matching; union geometry targets differ. DN logits absent and excluded explicitly.','GT<=300 restriction removes capacity saturation but is a selected descriptive stratum, not a causal intervention.','Aggregated counts share images, source annotations and checkpoints; no p-values or independent-sample inference.','Derivative in logit space is not model-parameter update, AP, learning stagnation or root cause.','Actual absolute score includes a shared native LQE term; semantic logit margin does not.'],
      'code_sha256':{n:sha(CODE/n) for n in ['prepare.py','analyze_saved.py','verify_native.py']}
    }
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    statepath=ROOT/'outputs/sod_roi12_20261009/execution_state.json'
    state=json.loads(statepath.read_text(encoding='utf-8-sig'))
    state['previous_goal_turn_classification']=state['last_goal_turn_classification']
    state['last_goal_turn_classification']='PROGRESS: revalidated both authorized servers idle; froze archived-output MAL comparison before deriving results. Latest UI request delivered with artifact hash matching verified folder-tree revision.'
    state['matched_quality_signal_diagnostic']={'status':'PREPARED_NOT_EXECUTED','protocol':str(OUT/'protocol.json'),'training_epochs_added':0,'optimizer_steps':0}
    state['next']='Execute frozen archived-output MAL loss and leaf-logit signal recomputation; verify native CPU autograd on same saved inputs without model forward or optimizer. Keep intentional quality calibration distinct from a demonstrated learning defect.'
    state['updated_utc']=protocol['frozen_local_utc']
    statepath.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':protocol['status'],'protocol_sha256':sha(OUT/'protocol.json')}))
if __name__=='__main__':main()
