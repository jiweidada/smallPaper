"""Independent fixed12 review from verified archives. No detector/optimizer calls."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
A49=ROOT.parent/'A49_下一主线有界筛选_20261101_v1'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    plan=load(ROOT/'protocol_FROZEN_v1.json')
    assert sha(ROOT/'protocol_FROZEN_v1.json')=='c13b61bdbc09d81a051108f7a28516f14ae3a3521a81d8247574129ca413d027'
    assert plan['arms']==['B','M','Q'] and plan['new_training_arms']==['M','Q'] and plan['epochs']==12
    capacity=load(ROOT/'capacity_v2/capacity_result.json')
    assert capacity['status']=='PASS' and capacity['whole_model_forwards']==9
    assert capacity['capacity_only_optimizer_updates']==6
    guard_failure=load(ROOT/'capacity_v1_failed_archive/capacity_result.json')
    assert guard_failure['whole_model_forwards']==1 and guard_failure['capacity_only_optimizer_updates']==0
    folders={'B':A49/'completed_B_v1/arm_B',**{a:ROOT/('completed_'+a+'_v1')/('arm_'+a) for a in ['M','Q']}}
    verifications={'B':load(A49/'B_completed_local_verification.json'),**{a:load(ROOT/(a+'_completed_local_verification.json')) for a in ['M','Q']}}
    assert all(v['status']=='PASS' for v in verifications.values())
    results={a:load(p/'result.json') for a,p in folders.items()}
    expected_init='2c6e0f9c6765fc7ba069c2292e0884bb86924d2232a952499fcd251cc57b47e6'
    for a,r in results.items():
        assert r['status']=='COMPLETED' and r['epochs']==12 and r['optimizer_updates']==9708
        assert r['full_model_forwards']==10159 and r['initialization_sha256']==expected_init
        assert load(folders[a]/'exit.json')['exit_code']==0
        assert r['endpoint_metrics']==load(folders[a]/'epoch12_EMA_metrics.json')==verifications[a]['endpoint_metrics']
        if a!='B':
            assert r['protocol_sha256']==sha(ROOT/'protocol_FROZEN_v1.json')
            assert r['model_parameters']==9809592 and r['extra_parameters']==6480
    init={a:load(folders[a]/'initialization.json') for a in ['M','Q']}
    assert init['M']['new_module_initialization_sha256']==init['Q']['new_module_initialization_sha256']
    assert all(x['all_native_initial_tensors_identical'] and x['scratch_no_capacity_weights'] and x['seed']==20261101 for x in init.values())
    fields=['order_sha256','first_eight_actual_input_sha256','first_eight_pre_forward_rng_sha256','all_batches_targets_including_ignore_tiles_sha256']
    flows={}
    for e in range(1,13):
        logs={a:load(p/('epoch%02d_input_flow.json'%e)) for a,p in folders.items()}
        checks={k:logs['B'][k]==logs['M'][k]==logs['Q'][k] for k in fields}
        assert all(checks.values())
        assert all(x['epoch']==e and x['optimizer_updates']==e*809 for x in logs.values())
        flows[str(e)]=checks
    metrics={a:r['endpoint_metrics'] for a,r in results.items()}
    for a in ['M','Q']:
        for key in ['max_detections','evaluated_GT','ignored_valid_GT','area_convention','scope']:
            assert metrics[a][key]==metrics['B'][key],(a,key)
    g=plan['primary_gates_pp']
    qb=metrics['Q']['APs']-metrics['B']['APs'];qm=metrics['Q']['APs']-metrics['M']['APs'];qa=metrics['Q']['AP']-metrics['B']['AP']
    passed={'Q_minus_B_APs_ge_0_5pp':qb>=g['Q_minus_B_APs_min'],
            'Q_minus_M_APs_ge_0_3pp':qm>=g['Q_minus_M_APs_min'],
            'Q_minus_B_AP_ge_minus_0_2pp':qa>=g['Q_minus_B_AP_min']}
    assert all(k in passed for k in ['Q_minus_B_APs_ge_0_5pp','Q_minus_M_APs_ge_0_3pp','Q_minus_B_AP_ge_minus_0_2pp'])
    pilot=all(passed.values())
    result={'status':'FIXED12_REVIEW_COMPLETE','phase':'A52','goal_status':'active',
      'mainline_confirmed_by_this_file':False,'metrics':metrics,'gates':passed,'pilot_pass':pilot,
      'Q_minus_B_APs_pp':qb,'Q_minus_M_APs_pp':qm,'Q_minus_B_AP_pp':qa,
      'decision':'CANDIDATE_FOR_FURTHER_ASSESSMENT_NOT_NOVELTY_PROOF' if pilot else 'RETIRE_EXACT_RECIPE_NO_RESCUE_SWEEP',
      'all12_recorded_flow_checks':flows,'common_base_init_sha256':expected_init,
      'M_Q_new_module_initialization_sha256':init['M']['new_module_initialization_sha256'],
      'baseline_reused_not_new':'A49_B12 same fixed recipe and observed before candidate design; not an all-outcomes-blind preregistration',
      'new_model_forwards_this_phase':10+sum(results[a]['full_model_forwards'] for a in ['M','Q']),
      'new_formal_optimizer_updates':sum(results[a]['optimizer_updates'] for a in ['M','Q']),
      'new_optimizer_updates_including_capacity':6+sum(results[a]['optimizer_updates'] for a in ['M','Q']),
      'new_formal_epochs':24,'cumulative_model_forwards':185612,'cumulative_optimizer_updates_including_capacity':155903,
      'cumulative_formal_epochs':192,'verification_added_model_forwards':0,'verification_added_optimizer_updates':0,
      'limits':['One seed fixed12 is not statistical significance or cross-dataset generalization',
         'B and val548 were visible in multiple previous selections; this is not an untouched test',
         'M/Q differ in categorical-versus-Gaussian marginal embedding, not calibrated joint-box probability',
         'Support-grid encoding and diagonal per-query quality may contribute; inter-object discrimination is not isolated',
         'Fourier expectations, integrated encoding and position-relation attention have prior foundations',
         'Relation-DETR+ and Exact-NeRF have only abstract-level reading in the current audit'],
      'next_extra_training_queued':False}
    supervisor=load(ROOT/'three_arm_result.json')
    assert supervisor['status']=='COMPLETED' and supervisor['metrics']==metrics
    assert supervisor['gates']==passed and supervisor['pilot_pass']==pilot
    assert supervisor['new_formal_model_forwards']==20318 and supervisor['new_formal_optimizer_updates']==19416
    p=ROOT/'fixed_endpoint_judgment_A52.json';assert not p.exists();p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='all12_recorded_flow_checks'},ensure_ascii=False))


if __name__=='__main__':main()
