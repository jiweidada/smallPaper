"""Compact view of complete raw preflight evidence, using the same pinned SSH control."""
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
p=subprocess.run([sys.executable,'-W','ignore','-X','utf8','-B',str(HERE/'control_uavdt.py'),'script','--account','both',
                  '--script',str(HERE/'monitor_preflight_remote.py'),'--tag','compact_preflight_status','--timeout','60'],
                 capture_output=True,text=True,encoding='utf-8',timeout=120)
if p.returncode:
    print(p.stdout);print(p.stderr);raise SystemExit(p.returncode)
envelope=json.loads(p.stdout);raw=json.loads(Path(envelope['saved']).read_text(encoding='utf-8'));summary={}
for account,result in raw['results'].items():
    if result.get('returncode')!=0:summary[account]=result;continue
    r=json.loads(result['stdout']);cap=r.get('capacity');phase=r.get('phase_receipt')
    summary[account]={'data_verified_images':(r.get('data_progress') or {}).get('verified_images'),'data_ready':bool(r.get('data_ready')),
        'CPU_complete':bool(r.get('CPU_preparation_complete')),'CPU_error':r.get('CPU_preparation_failed'),
        'initialization_sha256':phase.get('initialization_sha256') if phase else None,
        'capacity_done':r.get('capacity_done'),'capacity_selected':cap.get('selected_concurrency') if cap else None,
        'capacity_maximum_tested_safe':cap.get('maximum_tested_safe_concurrency') if cap else None,
        'capacity_cases':[{'concurrency':c['concurrency'],'passed':c['passed'],'peak_used_MiB':c['peak_GPU_used_MiB'],
                           'minimum_free_MiB':c['minimum_GPU_free_MiB'],'aggregate_images_per_second':c['aggregate_images_per_second'],
                           'seconds_per_step':{arm:w['seconds_per_step_mean'] for arm,w in c['workers'].items()}} for c in r.get('capacity_cases',[]) if c],
        'runtime_contract':r.get('runtime_contract'),'runtime_done':r.get('runtime_contract_done')}
    for field in ['CPU_failure_log','capacity_failure_log','capacity_worker_failure_logs','runtime_contract_failure_log']:
        if r.get(field):summary[account][field]=r[field]
print(json.dumps({'evidence':envelope['saved'],'accounts':summary},ensure_ascii=False,indent=2))
