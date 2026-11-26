"""Read-only end-of-diagnostic proof; inspect source/recipe hashes and numerical-fallback logs."""
import datetime,hashlib,json,re
from pathlib import Path
root=Path(CONFIG['root']);manifest=json.loads((root/'frozen_manifest.json').read_text());changes=[]
for item in manifest['files']:
 p=root/item['relative']
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:changes.append(item['relative'])
result={'account':CONFIG['account'],'checked_remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'frozen_files_checked':len(manifest['files']),'changed_frozen_files':changes,'jobs':[]}
plan=json.loads((root/'plan.json').read_text())
for arm in plan['jobs']:
 job=f'uavdt_{arm}_full96_seed{plan["seed"]}';out=root/'outputs'/job;args=json.loads((out/'args.json').read_text());text=(root/'logs'/(job+'.log')).read_text(errors='replace')
 clean=re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',text)
 matches=[s for s in clean.splitlines() if re.search(r'WARNING|ERROR|Traceback|\bNaN\b|\bInf\b|nonfinite|\brollback\b|AMP fallback',s,re.I)]
 result['jobs'].append({'job':job,'args_fields':{k:args.get(k) for k in ['seed','use_amp','use_amp_fallback_fp32','force_fp32','batch_size','epoches','output_dir']},
  'warning_error_fallback_lines':matches[-15:],'amp_diagnostics':[json.loads(s).get('train_amp_skipped_steps') for s in (out/'log.txt').read_text().splitlines() if s.strip()]})
print(json.dumps(result),flush=True)
