from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parent;A51=ROOT.parent/'A51_跨分辨率守恒交换_20261101_v1'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cap=load(ROOT/'capacity_v2/capacity_result.json')
assert cap['status']=='PASS' and cap['whole_model_forwards']==9 and cap['capacity_only_optimizer_updates']==6
assert cap['protocol_sha256']==sha(ROOT/'capacity_v2/capacity_protocol_FROZEN_v1.json')=='ae74dbad43f4b0d016a69cd67d4d2c17b24735a0fba6aa5bf101f03406b9c636'
obs=load(ROOT/'observer_04_capacity_v2.json');obs=json.loads(obs['stdout'])
assert not obs['capacity_process']['live'] and all(obs['local_frozen_files_unchanged'].values())
assert load(ROOT/'capacity_v1_failed_archive/capacity_result.json')['whole_model_forwards']==1
source=(A51/'train_A51.py').read_text(encoding='utf-8')
source=source.replace('runtime_A51','runtime_A52').replace('9876858','9809592').replace('73746','6480')
source=source.replace("ROOT/'capacity_result.json'","ROOT/'capacity_v2/capacity_result.json'")
source=source.replace("ROOT/'capacity_protocol_FROZEN_v1.json'","ROOT/'capacity_v2/capacity_protocol_FROZEN_v1.json'")
old="assert all(float(p.beta.detach())==0 for p in module.pairs)"
assert source.count(old)==1
source=source.replace(old,"assert all(torch.count_nonzero(x.last.weight)==0 for x in module.maps)")
for old,new in [
 ("'exchange_beta':[float(p.beta.detach()) for p in module.pairs]","'relation_projection_norms':[float(x.last.weight.detach().norm()) for x in module.maps]"),
 ("'exchange_beta_EMA':[float(p.beta.detach()) for p in module.pairs]","'relation_projection_norms_EMA':[float(x.last.weight.detach().norm()) for x in module.maps]")]:
    assert old in source;source=source.replace(old,new)
source=source.replace('native ECDet plus the same A51 exchange module; no GT or teacher at inference',
                      'native ECDet plus the same A52 marginal relation bias; no GT or teacher at inference')
assert 'module.pairs' not in source and 'A51' not in source
supervise=(A51/'supervise_A51.py').read_text(encoding='utf-8').replace('train_A51.py','train_A52.py').replace('A51','A52')
supervise=supervise.replace("ROOT / 'capacity_result.json'","ROOT / 'capacity_v2/capacity_result.json'")
assert 'A51' not in supervise
for name,text in [('train_A52.py',source),('supervise_A52.py',supervise)]:
    ast.parse(text);p=ROOT/name;assert not p.exists();p.write_text(text,encoding='utf-8')
# The observer is separate from frozen training code and keeps the failed first guard visible.
observer=(ROOT/'observe_A52.py').read_text(encoding='utf-8')
observer=observer.replace("ROOT=Path(__file__).resolve().parent","ROOT=Path(__file__).resolve().parent\nCAP=ROOT/'capacity_v2'")
for name in ['capacity_launch_identity.json','capacity_progress.json','capacity_result.json','capacity_stderr.log']:
    observer=observer.replace("ROOT/'"+name+"'","CAP/'"+name+"'")
observer=observer.replace("result['capacity_process']=process(result['capacity_identity'])",
 "result['capacity_process']=process(result['capacity_identity'])\nresult['prior_v1_guard_failure']=load(ROOT/'capacity_result.json')")
ast.parse(observer)
(ROOT/'observe_formal_A52.py').write_text(observer,encoding='utf-8')
prior=load(A51/'protocol_FROZEN_v1.json')
files=['runtime_A52.py','distribution_relation.py','train_A52.py','supervise_A52.py',
       'capacity_v2/runtime_A52.py','capacity_v2/distribution_relation.py','capacity_v2/capacity_A52.py',
       'capacity_v2/capacity_protocol_FROZEN_v1.json','capacity_v2/capacity_result.json']
plan={'phase':'A52','protocol_type':'FORMAL_FIXED12','goal_status':'active','mainline_confirmed':False,
 'question':'Does native categorical marginal shape improve query-pair relations beyond a same-moment Gaussian approximation?',
 'arms':['B','M','Q'],'new_training_arms':['M','Q'],'B_reused':'A49 fixed12 native original',
 'B_result_visible_before_design':True,'not_all_outcomes_blind_preregistration':True,
 'epochs':12,'train_images':6471,'val_images':548,'batch':8,'updates_per_epoch':809,
 'seed':20261101,'new_module_seed':20261102,'new_parameters_M_Q':6480,'model_parameters_M_Q':9809592,
 'interface':'decoder layers2-4 ordinary300 query self-attention additive bias only; previous marginals detached; native DN mask preserved',
 'frequency_bands':[1,2,4,8,16,32,64,128],'angular_frequency_factor':'2*pi','bias_scale':0.1,'relation_hidden_width':48,
 'M':'same physical marginal mean/variance Gaussian characteristic features + point geometry',
 'Q':'native categorical marginal characteristic features + same point geometry',
 'loss_matching_query_count_and_feature_resolutions_unchanged':True,
 'same_parameter_count_not_bitwise_same_operator_cost':True,
 'init':'identical native tensors/common init; identical zero-output relation map tensors; scratch no capacity weights',
 'augmentation':'same deterministic640stretch/flip/noMosaic as reused A49 B',
 'evaluation':'same COCO-style original clipped-GT area small<=1024,class0 union IoA,maxDets300; not official challenge score',
 'fixed_EMA_evaluations':[4,8,12],'primary_endpoint':'fixed12 EMA',
 'primary_gates_pp':{'Q_minus_B_APs_min':0.5,'Q_minus_M_APs_min':0.3,'Q_minus_B_AP_min':-0.2},
 'capacity_protocol_sha256':sha(ROOT/'capacity_v2/capacity_protocol_FROZEN_v1.json'),
 'capacity_result_sha256':sha(ROOT/'capacity_v2/capacity_result.json'),
 'prior_capacity_guard_failure_forwards':1,'capacity_forward_total':10,'capacity_optimizer_total':6,
 'new_formal_forward_cap':20318,'new_formal_optimizer_cap':19416,'new_formal_epoch_cap':24,
 'new_phase_total_forward_cap':20328,'new_phase_total_optimizer_cap_including_capacity':19422,
 'cumulative_forward_cap':185612,'cumulative_optimizer_cap_including_capacity':155903,'cumulative_formal_epoch_cap':192,
 'actual_completed_counts_before_formal':{'whole_model_forwards':165294,'formal_optimizer_updates':136438,'capacity_only_updates':49,'all_optimizer_updates':136487,'formal_epochs':168},
 'timeout_hours_per_arm':6,'automatic_retries':0,'additional_training_queued':False,
 'fail_action':'Retire exact recipe; no frequency/weight/layer/seed/endpoint rescue; broad goal remains active.',
 'validation_reuse_limit':'val548 reused in many prior selections, not an untouched test set; one seed12 not significance or paper readiness',
 'recent_fulltext_overlap_limits':['Relation-DETR+ abstract only','Exact-NeRF abstract only, not a detection method'],
 'files_sha256':{n:sha(ROOT/n) for n in files},'external_sha256':prior['external_sha256']}
p=ROOT/'protocol_FROZEN_v1.json';assert not p.exists();p.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
receipt={'status':'FORMAL_PROTOCOL_FROZEN_NOT_YET_LAUNCHED','goal_status':'active','mainline_confirmed':False,
 'protocol_sha256':sha(p),'capacity_v2_pass':True,'capacity_old_failure_preserved':True,
 'source_origin':{str(A51/n):sha(A51/n) for n in ['train_A51.py','supervise_A51.py']},
 'new_training_source_sha256':{n:sha(ROOT/n) for n in ['train_A52.py','supervise_A52.py']},
 'counts_before_formal':plan['actual_completed_counts_before_formal'],'no_capacity_weights_reused':True}
(ROOT/'06_容量通过与正式协议冻结.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False))

