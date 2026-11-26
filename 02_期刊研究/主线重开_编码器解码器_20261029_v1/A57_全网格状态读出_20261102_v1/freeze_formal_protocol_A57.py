"""Local, metadata-only protocol construction after terminal integration proof."""
from pathlib import Path
from datetime import datetime,timezone
import ast
import hashlib
import json

P=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(p.read_text('utf-8'))


def main():
    design=load(P/'pilot_design_FROZEN_v1.json')
    integ=load(P/'eval_integration_result.json')
    audit=load(P/'eval_integration_terminal_audit.json')
    assert integ['status']=='PASS_EVAL_INTEGRATION_ONLY'
    assert integ['whole_detector_forwards']==3 and integ['optimizer_updates']==0
    assert integ['COCO_AP_evaluations']==0 and integ['identical_to_frozen_prior_GT']
    assert audit['status']=='PASS_TERMINAL_INTEGRATION_AND_INPUTS'
    assert not any(r['owned_live'] for r in audit['owned_jobs'])
    assert audit['model_forward_entries']==audit['model_forward_completions']==3
    assert sha(P/'common_initialization.pth')==design['initialization_sha256']
    local_check=load(P/'local_prelaunch_checks_v1.json')
    assert local_check['status']=='PASS_STATIC_AND_CPU_EVAL_GATE_CHECKS'
    for row in local_check['source_AST']:assert sha(P/row['file'])==row['sha256']
    names=['common_A57.py','evaluate_A57.py','train_A57.py','eval_integration_A57.py',
           'eval_integration_supervisor_A57.py','supervise_A57.py','launch_A57.py',
           'native_data.py','native_criterion.py','native_variants_v2.py','state_readout_v2.py',
           'pilot_design_FROZEN_v1.json','validation_GT_FROZEN_v1.json','local_prelaunch_checks_v1.json',
           'capacity_terminal_input_audit.json','common_initialization_local_verification.json',
           'eval_integration_scope_FROZEN_v1.json','eval_integration_result.json',
           'eval_integration_exit.json','eval_integration_terminal_audit.json','formal_runtime_binding.json']
    names+=sorted(p.relative_to(P).as_posix() for p in (P/'host_compat_v2').rglob('*.py'))
    for n in names:
        if n.endswith('.py'):ast.parse((P/n).read_text('utf-8'))
    runtime=load(P/'formal_runtime_binding.json')
    external={'/root/uav_a49_20261101_v1/arm_B/epoch12_EMA_GT.json':{
        'sha256':sha(P/'validation_GT_FROZEN_v1.json'),'use':'GT convention only'}}
    external.update(runtime['external_files'])
    plan={'phase':'A57_FIXED12_BMQ','frozen_utc':datetime.now(timezone.utc).isoformat(),
        'frozen_before_A57_AP':True,'arms':['B','M','Q'],'epochs_per_arm':12,
        'primary_endpoint':'epoch12 RAW; no EMA or best-epoch selection',
        'design_sha256':sha(P/'pilot_design_FROZEN_v1.json'),
        'files_sha256':{n:sha(P/n) for n in names},'external_files':external,
        'initialization_sha256':design['initialization_sha256'],
        'initial_tensor_digest':design['initial_tensor_digest'],
        'integration':{'whole_detector_forwards':3,'optimizer_updates':0,'COCO_AP_evaluations':0,
                       'terminal_audit_sha256':sha(P/'eval_integration_terminal_audit.json')},
        'per_arm_caps':{'whole_model_forwards':19867,'train_forwards':19416,'val_forwards':411,
                        'latency_forwards':40,'optimizer_updates':9708,'epochs':12,'wall_hours':32},
        'three_arm_caps':{'whole_model_forwards':59601,'optimizer_updates':29124,'epochs':36,'wall_hours':72},
        'prior_canonical_after_capacity':{'model_forwards':226281,'formal_training_updates':194686,
            'capacity_only_optimizer_updates':76,'all_detector_optimizer_updates':194762,'formal_epochs':240},
        'global_caps_if_all_three_finish':{'model_forwards':285885,'formal_training_updates':223810,
            'capacity_only_optimizer_updates':76,'all_detector_optimizer_updates':223886,'formal_epochs':276},
        'baseline_gate':design['baseline_learning_gate_at_fixed12'],
        'candidate_gates_pp':design['primary_candidate_gates_pp_all_required'],
        'no_automatic_retry_or_rescue':True,'post_training_queue':None,
        'limits':design['future_journal_validation_risks']}
    assert not (P/'protocol_FROZEN_v1.json').exists()
    with (P/'protocol_FROZEN_v1.json').open('x',encoding='utf-8') as f:
        json.dump(plan,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({'status':'FROZEN_BEFORE_A57_AP','files':len(names),
                      'protocol_sha256':sha(P/'protocol_FROZEN_v1.json'),'model_forward_cap':59601}))


if __name__=='__main__':main()
