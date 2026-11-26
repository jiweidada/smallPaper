"""One-shot local registration of the requested follow-on plan; never deploys or trains."""
import datetime
import hashlib
import json
import os
from pathlib import Path

WORKSPACE=Path(__file__).resolve().parents[3]
ROOT=WORKSPACE/'英文小论文/data/07_第二数据集_20261017'
EVIDENCE=ROOT/'协议核查_v1'

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def main():
    queue=ROOT/'UAVDT_96轮待执行队列_20261017_v1.json'
    backup=ROOT/'第二数据集选择状态_纳入UAVDT计划前_20261017_v1.json'
    policy_path=ROOT/'第二数据集选择状态.json'
    assert not queue.exists() and not backup.exists(),'Registration already attempted; inspect artifacts, do not repeat.'
    prior=policy_path.read_bytes();policy=json.loads(prior)
    assert policy['recommended_dataset']=='UAVDT' and policy['allow_automatic_training_start'] is False
    audit=read(EVIDENCE/'UAVDT完整原始包验收_v1.json')
    assert audit['state']=='complete_original_detection_archive_verified'
    val=['M0603','M1003','M0501','M1002','M0204','M0402']
    train=sorted(set(audit['splits']['train'])-set(val));test=audit['splits']['test']
    counts={r['sequence']:r['images'] for r in audit['per_sequence']}
    assert len(train)==24 and len(val)==6 and len(test)==20
    assert sum(counts[s] for s in train)==18990 and sum(counts[s] for s in val)==5153
    assert not(set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();accounts={}
    arms=['c','c_mean','baseline','lowpass']
    for account,port,base,seed in [('A',30512,'/25085400120/DEIM/experiments',0),('B',30059,'/24085403014/experiments',1)]:
        experiment=base+'/phase_second_dataset_uavdt_20261017_v1';jobs=[]
        for arm in arms:
            job=f'uavdt_{arm}_full96_seed{seed}'
            jobs.append({'job_id':job,'arm':arm,'seed':seed,'epochs':96,'state':'planned_not_launched',
                         'start_from':'dataset_common_initialization','output':experiment+'/outputs/'+job,
                         'log':experiment+'/logs/'+job+'.log','job_status':experiment+'/jobs/'+job+'.json'})
        accounts[account]={'port':port,'seed':seed,'planned_experiment_root':experiment,
                           'maximum_concurrent_trainings':2,'planned_launch_batches':[['c','c_mean'],['baseline','lowpass']],
                           'jobs':jobs}
    plan={'plan_id':'uavdt_second_dataset_full96_20261017_v1','created_utc':now,
          'state':'planned_blocked_on_technical_preflight','selected_dataset':'UAVDT_v1',
          'queue_is_executable':False,'deployed':False,'new_training_started':False,'jobs_total':8,'epochs_per_job':96,
          'accounts':accounts,
          'split':{'tag':'uavdt-validation-v1-20261017','train_sequences':train,'val_sequences':val,'test_sequences':test,
                   'train_images':18990,'val_images':5153,'test_images':16592,'official_test_for_weight_selection':False,
                   'sequence_split_committed_before_training':True},
          'evaluation':{'native_test_primary_metric':'official_UAVDT_vehicle_AP70','class_agnostic':True,
                        'model_vehicle_classes':1,'best_stg2_selection':'heldout_train_sequence_validation_AP70_only',
                        'independent_eval':['val','official_test'],'main_pair':['c','c_mean'],
                        'auxiliary_pairs':[['c','baseline'],['c','lowpass'],['c_mean','lowpass']],
                        'size_input640_edges':[0,4,8,16,32],'empty_size_group':'NA_not_zero',
                        'two_seeds_claim':'paired_description_only_no_significance_or_equivalence'},
          'initialization':{'public_source_sha256':'85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91',
                            'dataset_common_sha256':None,'reset_required_category_parameters_once_for_all_arms':True,
                            'not_from_trained_VisDrone_or_TinyPerson':True},
          'remaining_technical_gates':['fresh_both_old_queues_complete_and_results_archived',
                 'A_full_native_data_replica_and_both_data_view_SHA_verified',
                 'native_scorer_integrated_and_validation_ignore_policy_tested',
                 'category_and_training_ignore_treatment_frozen','common_initialization_and_C_Cmean_tensor_identity_verified',
                 'stage2_checkpoint_binding_and_restore_integration_passed','all_job_runtime_save_paths_distinct_and_verified',
                 'dual_training_memory_throughput_and_timeout_measured',
                 'both_deployment_acceptance_ready_and_single_controller_guaranteed'],
          'entry_command':None,
          'blocked_reason':'UAVDT training/validation adapter and continuation controller have not been implemented and accepted. This is a planned manifest, not a runnable queue.',
          'cost_projection':{'per_job_GPU_hours':40.79621315192744,'total_GPU_hours':326.3697052154195,
                             'two_card_ideal_days':6.799368858654573,'measured_on_UAVDT':False,
                             'excludes_final_independent_eval':True},
          'formal_plan':str(WORKSPACE/'英文小论文/data/00_实验计划与证据索引/UAVDT第二数据集96轮实验计划_20261017_v1.txt'),
          'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                     for p in [EVIDENCE/'UAVDT完整原始包验收_v1.json',EVIDENCE/'UAVDT官方DET参考曲线核验_v1.json',
                               ROOT/'UAVDT与TinyPerson完整证据比较_20261017_v2.json']]}
    with queue.open('x',encoding='utf-8') as f:json.dump(plan,f,ensure_ascii=False,indent=2)
    with backup.open('xb') as f:f.write(prior)
    policy.update(state='UAVDT_selected_in_plan_waiting_technical_preflight',selected_dataset='UAVDT_v1',
        allow_automatic_training_start=False,latest_user_instruction='实验计划 加上去了吗',
        selection_basis='Assistant selects UAVDT for the requested follow-on plan within the prior second-dataset preparation/continuation authorization; not a claimed verbatim user selection.',
        training_authorization_source='那你先核实 UAVDT 和 TinyPerson 的数据和评测协议，确定第二个数据集用哪个，准备好后等这轮跑完接着跑。',
        updated_utc=now,planned_manifest=queue.name,formal_plan=plan['formal_plan'],
        technical_gate_not_new_user_approval=True,UAVDT_training_adapter_ready=False,planned_jobs=8)
    assert policy_path.read_bytes()==prior,'Selection changed concurrently; inspect before any further mutation.'
    next_policy=policy_path.with_suffix('.next_20261017_plan_v1')
    with next_policy.open('x',encoding='utf-8') as f:json.dump(policy,f,ensure_ascii=False,indent=2)
    os.replace(next_policy,policy_path)
    print(json.dumps({'planned_jobs':8,'accounts':{a:{'seed':v['seed'],'jobs':len(v['jobs'])} for a,v in accounts.items()},
        'selected_dataset':policy['selected_dataset'],'allow_automatic_training_start':False,'deployed':False,
        'manifest':str(queue)},ensure_ascii=False))

if __name__=='__main__':main()
