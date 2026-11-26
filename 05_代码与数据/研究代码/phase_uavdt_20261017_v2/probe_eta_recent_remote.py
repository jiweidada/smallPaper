"""Read-only recent complete-epoch timing and current progress for a conditional ETA."""
import datetime,json,re,subprocess
from pathlib import Path
root=Path(CONFIG['root']);plan=json.loads((root/'plan.json').read_text())
ansi=re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
report={'account':CONFIG['account'],'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':[]}
for arm in plan['jobs']:
    job=f'uavdt_{arm}_full96_seed{plan["seed"]}';out=root/'outputs'/job
    status=json.loads((root/'jobs'/(job+'.json')).read_text())
    rows=[json.loads(s) for s in (out/'log.txt').read_text().splitlines() if s.strip()]
    rows=[r for r in rows if 'test_uavdt_eval_bbox' in r]
    p=root/'logs'/(job+'.log')
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-512000));text=ansi.sub('',f.read().decode(errors='replace'))
    if 'RESUME_FROM_OWN_CHECKPOINT ' in text:text=text.rsplit('RESUME_FROM_OWN_CHECKPOINT ',1)[1]
    matches=list(re.finditer(r'Epoch:\s*(\d+)\s*/\s*96[^\r\n]*',text))
    progress=matches[-1].group(0) if matches else None
    batch=re.search(r'\[\s*(\d+)\s*/\s*(\d+)\s*\]',progress or '')
    step=re.search(r'\btime:\s*([\d.]+)',progress or '')
    selected=[]
    for row in rows[-12:]:
        selected.append({'epoch_zero_based':row['epoch'],
            'epoch_seconds':row.get('experiment_epoch_seconds'),
            'train_seconds':row.get('experiment_train_seconds'),
            'AP70':row['test_uavdt_eval_bbox'][0],
            'timing_fields':{k:v for k,v in row.items() if k.startswith('experiment_')}})
    result={'job':job,'arm':arm,'seed':plan['seed'],'state':status.get('state'),'pid':status.get('pid'),
        'complete_epochs':len(rows),'last_logged_epoch':rows[-1]['epoch'] if rows else None,
        'epoch_zero_based':int(matches[-1].group(1)) if matches else None,
        'batch_zero_based':int(batch.group(1)) if batch else None,
        'batches_per_epoch':int(batch.group(2)) if batch else None,
        'seconds_per_step':float(step.group(1)) if step else None,
        'recent_complete_timings':selected,'resumed_from':status.get('resumed_from'),
        'already_available_final_evaluations':[]}
    for split in ['val','test']:
        metrics=root/'results'/job/split/'metrics.json'
        if metrics.is_file():
            m=json.loads(metrics.read_text());result['already_available_final_evaluations'].append(
                {'split':split,'images':m.get('images'),'elapsed_seconds':m.get('elapsed_seconds')})
    report['jobs'].append(result)
report['GPU']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
print(json.dumps(report),flush=True)
