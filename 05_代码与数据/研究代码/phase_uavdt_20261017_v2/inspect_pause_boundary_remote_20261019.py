"""Read-only inspection before the user-authorized epoch-boundary pause."""
import json, pathlib, hashlib
root=pathlib.Path(CONFIG['root'])
def load(p):return json.loads(p.read_text()) if p.is_file() else None
plan=load(root/'plan.json')
solver=root/'code/engine/solver/det_solver.py'
lines=solver.read_text().splitlines()
result={'account':CONFIG['account'],'root':str(root),'launch':load(root/'launch_receipt.json'),'queue':load(root/'queue_state.json'),'plan':plan,'solver_sha256':hashlib.sha256(solver.read_bytes()).hexdigest(),'solver_save_sections':[],'jobs':[]}
for i,line in enumerate(lines):
 if "f.write(json.dumps(log_stats)" in line:
  result['solver_save_sections'].append({'line':i+1,'text':'\n'.join(lines[max(0,i-35):i+8])})
for arm in plan['jobs']:
 job=f'uavdt_{arm}_full96_seed{plan["seed"]}';out=root/'outputs'/job
 rows=[json.loads(s) for s in (out/'log.txt').read_text().splitlines() if s.strip()]
 result['jobs'].append({'job':job,'status':load(root/'jobs'/(job+'.json')),'last_log_epoch':rows[-1].get('epoch'),'checkpoint_files':[{'name':p.name,'bytes':p.stat().st_size} for p in out.glob('*.pth')],'config':plan['configs'][arm]})
print(json.dumps(result),flush=True)
