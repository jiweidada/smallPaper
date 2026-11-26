"""Print only contract and ledger fields needed for the review text."""
import sys,importlib,json
sys.stdout.reconfigure(encoding='utf-8');C=importlib.import_module('342_stage5_context')
def show(label,value):print(json.dumps({'label':label,'value':value},ensure_ascii=False),flush=True)
d=C.read(C.S2/'342_阶段2_result.json')
show('loss_formula_contracts',d['loss_formula_contracts']);show('optimizer_routes',d['optimizer_routes']);show('global',d['global'])
d=C.read(C.S3/'342_阶段3_result.json')
show('val_storage',d['storage']);show('val_timing',d['timing_receipts']);show('val_timing_scope',d['timing_scope'])
for p in ('result_task4_pipeline.json','result_redo_operator_logsumexp.json','340_result_unit_tests.json','result_redo3_frozen311_320.json'):
    x=C.read(C.WORK/p);show(p,{k:v for k,v in x.items() if k not in ('source_sha256','frozen_sha','source_files','calls')})
d=C.read(C.S344/'344_result_train344四臂CPU冒烟与续跑.json')
for s in ('1','2'):
    show('head_parameters_seed'+s,{a:d['arms'][s][a]['parameter_count'] for a in 'NSGR'})
    g=d['arms'][s]['G'];show('G_gradients_seed'+s,{k:g[k] for k in ('per_class_nonzero_gradient_steps','class_null_final','class_null_loss_routing')})
for p in sorted(C.S343.glob('343_result_3b构建_seed*.json')):
    x=C.read(p);show('dev_build_receipt',{'name':p.name,'seconds':x['elapsed_seconds'],'counts':x['counts']})
d=C.read(C.S3/'342_val548_逐条SHA清单.json');show('val_manifest_entry',d['entries'][0])
d=C.read(C.SPEC/'327_训练实施输入索引.json');show('index_keys',list(d));show('checkpoint_metadata',d['checkpoints'])
