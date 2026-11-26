"""Read-only actual epoch/validation timings and a small sample of existing log lines."""
import datetime,json,re
from pathlib import Path
root=Path(CONFIG['root']);ansi=re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
report={'account':CONFIG['account'],'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':[]}
plan=json.loads((root/'plan.json').read_text())
for arm in plan['jobs']:
 job=f'uavdt_{arm}_full96_seed{plan["seed"]}';p=root/'logs'/(job+'.log');text=ansi.sub('',p.read_text(errors='replace'));lines=text.splitlines()
 epochs=[line for line in lines if re.search(r'Epoch:\s*\d+\s*/\s*96',line)]
 times=[line for line in lines if 'Total time:' in line or 'total time:' in line or 'Training time' in line or 'Evaluat' in line and 'time' in line.lower()]
 boundaries=[line for line in lines if re.search(r'\[det_solver\.py:|\[uavdt_',line)]
 vals=[line for line in lines if 'AP70' in line or 'Average forward' in line or 'Average FPS' in line]
 bindings=re.findall(r'output_dir:\s*([^\s]+)',text)
 rows=[];log=root/'outputs'/job/'log.txt'
 if log.is_file():rows=[json.loads(s) for s in log.read_text().splitlines() if s.strip()]
 output={'job':job,'started':json.loads((root/'jobs'/(job+'.json')).read_text())['started_utc'],'bindings':bindings,
    'actual_output_ok':bool(bindings) and all(s==str(root/'outputs'/job) for s in bindings),
    'epoch_total_times':times,'solver_boundaries_tail':boundaries[-22:],'validation_lines_tail':vals[-15:],
    'current_progress_line':epochs[-1] if epochs else None,'last_regular_progress_lines':epochs[-8:],
    'structured_row_keys':list(rows[-1]) if rows else [],'structured_last_row':rows[-1] if rows else None}
 report['jobs'].append(output)
print(json.dumps(report),flush=True)
