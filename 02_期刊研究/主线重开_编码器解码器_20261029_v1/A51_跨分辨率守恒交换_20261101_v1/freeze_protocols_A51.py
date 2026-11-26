from pathlib import Path
import hashlib,json,py_compile
ROOT=Path(__file__).resolve().parent
A49=ROOT.parent/'A49_下一主线有界筛选_20261101_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,j):assert not p.exists();p.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
expect=json.loads((A49/'asset_expectations.json').read_text(encoding='utf-8'))
external={'/root/uav_a47_20261101_v1/assets_A43_baseline/common_init_10class.pth':expect['common_init_sha256'],
 '/root/uav_a47_20261101_v1/assets_A43_baseline/runtime_common.py':expect['runtime_common_sha256']}
prefix='/root/uav_a47_20261101_v1/assets_A43_baseline/source/EdgeCrafter-d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecdetseg/'
external.update({prefix+n:h for n,h in expect['native_source_sha256'].items()})
for n in ['runtime_A49.py','ignore_query_supervision.py','training_utils_A49.py','evaluate.py']:
    external['/root/uav_a49_20261101_v1/'+n]=sha(A49/n)
for p in [A49/'completed_B_v1/arm_B/result.json',A49/'completed_B_v1/arm_B/epoch12.pth']:
    external['/root/uav_a49_20261101_v1/arm_B/'+p.name]=sha(p)
for epoch in range(1,13):
    name='epoch%02d_input_flow.json'%epoch
    external['/root/uav_a49_20261101_v1/arm_B/'+name]=sha(A49/'completed_B_v1/arm_B'/name)
names=['runtime_A51.py','conservative_scale_exchange.py','capacity_A51.py','train_A51.py','supervise_A51.py']
for name in names:py_compile.compile(str(ROOT/name),doraise=True)
capacity={'phase':'A51','type':'CAPACITY_ONLY','arms':['B','M','Q'],
 'max_whole_model_forwards':9,'max_optimizer_updates':6,'formal_epochs':0,'no_auto_retry':True,
 'initial_identity':'B/M/Q whole-model eval pred_logits and pred_boxes must be bitwise equal',
 'temporary_training':'M and Q each3 batch8 updates; all new parameter groups must change finitely',
 'memory_ceiling_bytes':23*1024**3,'files_sha256':{n:sha(ROOT/n) for n in ['runtime_A51.py','conservative_scale_exchange.py','capacity_A51.py']},
 'external_sha256':external}
save(ROOT/'capacity_protocol_FROZEN_v1.json',capacity)
plan={'phase':'A51','type':'FIXED12_MATCHED_THREE_ARM_PILOT_REUSED_NATIVE_BASELINE',
 'goal_status':'active','mainline_confirmed':False,'arms':['B','M','Q'],'new_training_arms':['M','Q'],
 'B_reuse':'completed native A49 B12, same seed/init/inputs/optimizer/eval/endpoint; already observed, not all-outcomes-blind preregistration',
 'B_endpoint':json.loads((A49/'completed_B_v1/arm_B/result.json').read_text())['endpoint_metrics'],
 'epochs':12,'updates_per_epoch':809,'fixed_endpoint':'epoch12 EMA','evaluation_epochs':[4,8,12],
 'train_images':6471,'val_images':548,'valid_train_GT':343204,'valid_val_GT':38759,
 'seed':20261101,'new_module_seed':20261102,'input':'640stretch/noMosaic/deterministicflip',
 'batch_size':8,'workers':2,'accumulation':1,'native_optimizer_scheduler_unchanged':True,
 'M':'same orthogonal frames and two independent invariant gates per adjacent scale pair',
 'Q':'same parameters; replace independent directional gates by their mean, shared paired exchange',
 'new_parameters_M_Q':73746,'model_parameters_M_Q':9876858,'native_parameters_B':9803112,
 'alpha':'.25*tanh(beta), beta0; no weight/site/layer/grid sweep',
 'property_scope':'per-forward feature-frame sum and finest block residual in exact arithmetic, not semantic/energy conservation',
 'primary_gates_pp':{'Q_minus_B_APs_min':.5,'Q_minus_M_APs_min':.3,'Q_minus_B_AP_min':-.2},
 'evaluation':'COCO-style original clipped GT area small<=1024, maxDets300, class0 union IoA; not officialVisDrone',
 'per_epoch_required_B_input_checks':['order_sha256','first_eight_actual_input_sha256','first_eight_pre_forward_rng_sha256','all_batches_targets_including_ignore_tiles_sha256'],
 'capacity_protocol_sha256':sha(ROOT/'capacity_protocol_FROZEN_v1.json'),
 'max_capacity_whole_model_forwards':9,'max_capacity_optimizer_updates':6,
 'max_new_formal_whole_model_forwards':20318,'max_new_formal_optimizer_updates':19416,
 'max_new_formal_epochs':24,'max_new_forwards_this_phase':20327,
 'cumulative_whole_model_forward_ceiling':165284,'cumulative_optimizer_update_ceiling_including_capacity':136481,
 'wall_time_limit_hours_per_new_arm':6,'no_auto_retry':True,'no_extra_training_queue':True,
 'stop':'any nonfinite/budget/init/file/data/flow failure stops sequence; scientific fixed12 gate failure retires exact recipe without rescue grid',
 'files_sha256':{n:sha(ROOT/n) for n in names},'external_sha256':external}
save(ROOT/'protocol_FROZEN_v1.json',plan)
print(json.dumps({'capacity_protocol_sha256':sha(ROOT/'capacity_protocol_FROZEN_v1.json'),
 'formal_protocol_sha256':sha(ROOT/'protocol_FROZEN_v1.json'),'forward_ceiling':165284,
 'new_training':['M12','Q12'],'baseline_reused':True}))
