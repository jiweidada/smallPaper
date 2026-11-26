import json
from pathlib import Path
base=Path(CONFIG['storage'])/'datasets_verified_20261017';data=base/'UAVDT_v1';holder=base/'UAVDT_prepare_v2';root=Path(CONFIG['root'])
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def tail(p):return '\n'.join(p.read_text(errors='replace').splitlines()[-18:]) if p.is_file() else None
result={'account':CONFIG['account'],'data_progress':read(data/'verification_progress_v2.json'),
        'data_ready':read(data/'data_ready.json'),'data_failure':read(data/'verification_failed_v2.json'),
        'CPU_preparation_complete':read(holder/'CPU_preparation_complete.json'),
        'CPU_preparation_failed':read(holder/'CPU_preparation_failed.json'),
        'CPU_preparation_worker':read(holder/'CPU_preparation_worker.json'),
        'phase_receipt':read(root/'preparation_receipt.json'),'capacity':read(root/'capacity_receipt.json'),
        'capacity_worker':read(root/'capacity_worker_v2.json'),'capacity_done':read(root/'capacity_worker_result_v2.json')}
result['capacity_cases']=[read(root/'capacity_v2'/str(n)/'case_result.json') for n in [1,2,3]]
result['capacity_supervisors']={}
for version in [2,3]:
 p=holder/f'capacity_supervisor_v{version}.json';entry=read(p)
 if entry:
  proc=Path('/proc')/str(entry['pid']);status={'receipt':entry,'failure':read(holder/f'capacity_supervisor_failed_v{version}.json'),'log_tail':tail(holder/f'capacity_supervisor_v{version}.log')}
  status['alive']=proc.is_dir() and (proc/'stat').read_text().rsplit(')',1)[1].split()[19]==str(entry['start_ticks']) and (proc/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
  result['capacity_supervisors'][str(version)]=status
result['runtime_contract']=read(root/'runtime_contract_receipt.json')
result['runtime_contract_done']=read(root/'runtime_contract_worker_result.json')
if result['runtime_contract_done'] and result['runtime_contract_done']['returncode']!=0:result['runtime_contract_failure_log']=tail(root/'logs/runtime_contract.log')
current=read(holder/'CPU_preparation_current.json')
if current:
 result['CPU_preparation_complete']=read(holder/current['complete'])
 result['CPU_preparation_failed']=read(holder/current['failed'])
 result['CPU_preparation_worker']=read(holder/current['worker'])
if result['CPU_preparation_failed']:result['CPU_failure_log']=tail(holder/(current['log'] if current else 'CPU_preparation.log'))
if result['capacity_done'] and result['capacity_done'].get('returncode')!=0:result['capacity_failure_log']=tail(root/'logs/capacity_probe.log')
if result['capacity_done'] and result['capacity_done'].get('returncode')!=0:
 result['capacity_worker_failure_logs']={str(p.relative_to(root)):tail(p) for p in (root/'capacity_v2').rglob('*.log')}
print(json.dumps(result),flush=True)
