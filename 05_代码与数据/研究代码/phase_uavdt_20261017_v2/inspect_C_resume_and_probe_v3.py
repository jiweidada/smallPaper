import pathlib,json,re,subprocess
root=pathlib.Path(CONFIG['root']);probe=pathlib.Path(CONFIG['storage'])/'DEIM/experiments/matching_coverage_probe_20261019_v1'
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def tail(p,n=3000):return p.read_text(errors='replace')[-n:] if p.is_file() else None
result={'account':CONFIG['account'],'queue':read(root/'queue_state.json'),'jobs':[],'controller_log_tail':tail(root/'logs/controller_recovery_20261020_v3.log'),'probe_inputs':read(probe/'inputs/complete.json'),'input_progress':read(probe/'inputs/progress.json'),'input_task':read(probe/'task_records/inputs_registered_v3.json')}
plan=read(root/'plan.json')
for arm in plan['jobs']:
 job=f'uavdt_{arm}_full96_seed{plan["seed"]}';record=read(root/'jobs'/(job+'.json'));p=pathlib.Path('/proc')/str(record['pid']);identity=None
 if p.is_file():pass
 if p.is_dir():
  st=(p/'stat').read_text().rsplit(')',1)[1].split();identity={'state':st[0],'start_ticks':st[19],'matches':st[19]==str(record['start_ticks'])}
 text=(root/'logs'/(job+'.log')).read_text(errors='replace').rsplit('RESUME_FROM_OWN_CHECKPOINT ',1)[-1]
 text=re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',text)
 batches=[v for v in text.splitlines() if 'Epoch:' in v]
 loads=[v for v in text.splitlines() if 'Load ' in v or 'Resume checkpoint' in v]
 result['jobs'].append({'job':job,'state':record['state'],'pid':record['pid'],'process':identity,'recovery_id':record.get('recovery_id'),'resumed_from':record.get('resumed_from'),'fresh_batches':batches[-2:],'load_records':loads[-12:],'console_tail':text[-650:]})
result['GPU']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
result['GPU_compute']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
if result['input_task'] and result['input_task']['state']=='failed':result['input_failure_tail']=tail(probe/'task_records/inputs_registered_v3.log',5000)
print(json.dumps(result),flush=True)
