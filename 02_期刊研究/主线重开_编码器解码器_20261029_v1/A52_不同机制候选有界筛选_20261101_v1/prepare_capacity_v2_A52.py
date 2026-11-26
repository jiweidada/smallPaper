from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
failed=json.loads((ROOT/'capacity_v1_failed_archive/capacity_result.json').read_text())
assert failed['error']=="KeyError('pred_corners')" and failed['whole_model_forwards']==1 and failed['capacity_only_optimizer_updates']==0
obs=json.loads(json.loads((ROOT/'observer_02_capacity_failure_terminal.json').read_text())['stdout'])
assert not obs['capacity_process']['live'] and all(obs['local_frozen_files_unchanged'].values())
plan=json.loads((ROOT/'capacity_protocol_FROZEN_v1.json').read_text())
assert all(sha(ROOT/n)==h for n,h in plan['files_sha256'].items())
dest=ROOT/'capacity_v2';dest.mkdir(exist_ok=False)
for n in ['runtime_A52.py','distribution_relation.py']:
    shutil.copy2(ROOT/n,dest/n)
source=(ROOT/'capacity_A52.py').read_text()
before='        assert_finite_output(output)\n'
after='        assert all(torch.isfinite(output[k]).all() for k in ("pred_logits","pred_boxes"))\n'
assert source.count(before)==1
(dest/'capacity_A52.py').write_text(source.replace(before,after),encoding='utf-8')
for n in ['launch_capacity_A52.py','observe_A52.py']:
    text=(ROOT/n).read_text()
    if n.startswith('launch'):text=text.replace("/root/uav_a52_20261101_v1'","/root/uav_a52_20261101_v1/capacity_v2'")
    (dest/n).write_text(text,encoding='utf-8')
plan.update(protocol_revision=2,previous_guard_failure_whole_forwards=1,
 phase_capacity_total_whole_forward_cap_including_v1_failure=10,whole_model_forward_cap_this_isolated_check=9,
 scientific_recipe_and_hyperparameters_changed=False,
 implementation_fix='Eval checks only actually returned logits and boxes. Training full-output check remains unchanged.',
 old_capacity_lock_or_result_overwritten=False,automatic_retries=0)
plan['files_sha256']={n:sha(dest/n) for n in ['runtime_A52.py','distribution_relation.py','capacity_A52.py']}
p=dest/'capacity_protocol_FROZEN_v1.json';p.write_text(json.dumps(plan,indent=2),encoding='utf-8')
note={'status':'CAPACITY_V1_INTERFACE_GUARD_FAILURE_PRESERVED_V2_FROZEN','goal_status':'active',
 'previous_protocol':str(ROOT/'capacity_protocol_FROZEN_v1.json'),'previous_protocol_sha256':sha(ROOT/'capacity_protocol_FROZEN_v1.json'),
 'failure_archive':str(ROOT/'capacity_v1_failed_archive'),'original_process_live':False,
 'algorithm_sha256_unchanged':sha(ROOT/'distribution_relation.py')==sha(dest/'distribution_relation.py'),
 'fix_scope':'Evaluation-output guard only; no model, loss, thresholds, features, frequencies, layers, init or training recipe changed.',
 'new_protocol':str(p),'new_protocol_sha256':sha(p),
 'model_forwards_already_added':1,'optimizer_updates_already_added':0,'formal_epochs':0,
 'revised_capacity_phase_total_forward_cap':10,'capacity_update_cap':6,
 'new_formal_training_queued':False,'cumulative_forward_lower_bound':165285,
 'why_proceed':'Authorized necessary code validation; explicit local decision after source diagnosis, separate immutable revision. No automated retry, no efficacy-negative rescue.'}
n=ROOT/'05_推理输出检查修复与隔离预检修订.json';n.write_text(json.dumps(note,ensure_ascii=False,indent=2),encoding='utf-8')
current=ROOT/'CURRENT_A52.json';j=json.loads(current.read_text(encoding='utf-8'))
j.update(status=note['status'],latest_record=str(n),receipt_sha256=sha(n),
 capacity_protocol=str(p),capacity_protocol_sha256=sha(p),
 current_priority_candidate='A52_DISTRIBUTIONAL_PAIR_GEOMETRY_CAPACITY_ONLY',
 bridge_session_if_alive=33536,model_forward_lower_bound=165285,optimizer_update_lower_bound=136481)
current.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':note['status'],'algorithm_unchanged':True,'new_protocol_sha256':sha(p),'capacity_forward_cap_including_failed_guard':10}))

