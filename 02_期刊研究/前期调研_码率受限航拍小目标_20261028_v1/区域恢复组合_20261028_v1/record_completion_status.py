"""Synchronize research records with the returned goal-management status; no scientific execution."""
from pathlib import Path
import json,hashlib,datetime
root=Path(__file__).resolve().parent
base=root.parent;prior=base/'主线目标_20261028_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
receipt=read(root/'goal_completion_receipt.json')
assert receipt['goal']['status']=='complete'
audit=read(root/'completion_audit.json')
assert audit['required_work_remaining_for_this_research_only_goal']==[]
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
for p in [prior/'candidate_board.json',root/'candidate_board.json',prior/'research_run_ledger.json',root/'research_run_ledger.json']:
    v=read(p);v['goal_status']='complete';v['updated_at_utc']=now
    v['completion_scope']='User-authorized pre-experimental research line/hypothesis and minimal falsification route; no empirical efficacy/novelty claim.'
    v['completion_audit']=str(root/'completion_audit.json')
    v['goal_completion_receipt']=str(root/'goal_completion_receipt.json')
    if 'execution' in v:assert all(x==0 for x in v['execution'].values())
    if 'next_step' in v:v['next_step']='Research goal achieved; any experimental work requires a new explicit user instruction.'
    write(p,v)
p=root/'02_主线问题卡_C10.json';v=read(p)
v['status']='selected_preexperimental_mainline';v['research_goal_status']='complete';write(p,v)
p=root/'artifact_verification.json';v=read(p)
v.update(goal_status='complete',card_status='selected_preexperimental_mainline',checked_at_utc=now,
         completion_audit=str(root/'completion_audit.json'),goal_receipt=str(root/'goal_completion_receipt.json'))
for a in v['artifacts']:
    f=Path(a['path']);assert f.is_file()
    a['bytes']=f.stat().st_size;a['sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
assert v['scientific_outcome_verified'] is False and v['experiments_run'] is False
write(p,v)
print(json.dumps({'goal_status':'complete','selected':'C10','scope':'pre-experimental research','model_or_experiment_execution':0,'scientific_efficacy_proven':False},ensure_ascii=False))
