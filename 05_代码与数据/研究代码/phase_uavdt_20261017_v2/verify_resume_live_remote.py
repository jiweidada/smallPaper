"""Read-only proof of checkpoint loading, unique output binding and fresh batches."""
import datetime,hashlib,json,re
from pathlib import Path
root=Path(CONFIG['root']);plan=json.loads((root/'plan.json').read_text())
result={'account':CONFIG['account'],'checked_remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':[]}
for arm in plan['jobs']:
    job=f'uavdt_{arm}_full96_seed{plan["seed"]}';record=json.loads((root/'jobs'/(job+'.json')).read_text())
    raw=(root/'logs'/(job+'.log')).read_text(errors='replace')
    text=re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',raw.rsplit('RESUME_FROM_OWN_CHECKPOINT ',1)[1])
    selected=[s for s in text.splitlines() if re.search(r'Load (last_epoch|\w+.state_dict)|Resume checkpoint|output_dir:|Epoch:|Traceback|Error|nan|inf',s)]
    result['jobs'].append({'job':job,'pid':record['pid'],'recovery_id':record.get('recovery_id'),
        'checkpoint_last_epoch':record['resumed_from']['last_epoch'],'loads':[s for s in selected if 'Load ' in s],
        'output_bindings':list(dict.fromkeys(re.findall(r'output_dir:\s*([^\s]+)',text))),
        'fresh_batches':[s for s in selected if 'Epoch:' in s][-2:],
        'recent_console_tail':text[-1600:], 'args':{k:v for k,v in json.loads((root/'outputs'/job/'args.json').read_text()).items() if k in ['seed','resume','tuning','output_dir','use_amp','epoches']}})
print(json.dumps(result),flush=True)
