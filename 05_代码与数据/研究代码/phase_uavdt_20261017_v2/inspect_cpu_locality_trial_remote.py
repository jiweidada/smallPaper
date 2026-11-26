"""Read trial receipts and current owned affinities; no process changes."""
import datetime,hashlib,json,os
from pathlib import Path
root=Path(CONFIG['root']);art=root/'artifacts/performance_CPU_locality_v1'
def read(p):return json.loads(p.read_text()) if p.is_file() else None
result={'account':CONFIG['account'],'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'trial':{name:read(art/(name+'.json')) for name in ['launch','baseline','pinned_state','GPU_local_affinity','restore_receipt','restored_baseline','guardian_restore_receipt','result','failure']},'current_jobs':[]}
plan=read(root/'plan.json')
for arm in plan['jobs']:
 job=f'uavdt_{arm}_full96_seed{plan["seed"]}';status=read(root/'jobs'/(job+'.json'));pid=status['pid']
 stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();assert stat[19]==str(status['start_ticks'])
 audit=root/'outputs'/job/'batch_audit.jsonl'
 result['current_jobs'].append({'job':job,'pid':pid,'start_ticks':stat[19],'status':status['state'],'main_affinity':sorted(os.sched_getaffinity(pid)),
  'thread_affinities':{p.name:{'count':len(os.sched_getaffinity(int(p.name))),'minimum':min(os.sched_getaffinity(int(p.name))),'maximum':max(os.sched_getaffinity(int(p.name)))} for p in Path(f'/proc/{pid}/task').iterdir()},
  'first_batch_audits':[json.loads(s) for s in audit.read_text().splitlines()] if audit.is_file() else [],
  'config_sha256':hashlib.sha256(Path(plan['configs'][arm]).read_bytes()).hexdigest()})
if (art/'failure.json').is_file():result['failure_log']='\n'.join((art/'trial.log').read_text(errors='replace').splitlines()[-20:])
print(json.dumps(result),flush=True)
