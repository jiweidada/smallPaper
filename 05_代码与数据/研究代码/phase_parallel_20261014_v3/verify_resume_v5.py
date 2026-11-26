"""Verify native runtime paths and the first newly saved complete epoch."""
import datetime
import hashlib
import importlib.util
import json
import pathlib
import re
import subprocess
import sys

account=sys.argv[1]
phase=pathlib.Path('/25085400120/DEIM/experiments/phase_mechanism_20261012' if account=='A'
                   else '/24085403014/experiments/phase_mechanism_20261012')
parent=phase/'artifacts/round2_20261014_v1';stage=parent/'execution_v5'
assert json.loads((parent/'current_execution.json').read_text())['execution_directory']=='execution_v5'
plan=json.loads((stage/'resume_plan.json').read_text());queue=json.loads((stage/'queue_state.json').read_text())
spec=importlib.util.spec_from_file_location('frozen_v3',parent/'execution_v3/run_parallel_queue.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
result={'account':account,'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'queue_state':queue,'controller_identity':helper.identity(queue['queue_pid']),
        'main_trainers':helper.main_training_processes(phase),'jobs':[]}
for row in plan['resume_jobs']:
    job=row['job'];output=phase/'outputs'/job
    raw=(phase/'logs'/(job+'.log')).read_bytes()
    text=re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',raw.decode(errors='replace'))
    resumed=text.rsplit('RESUME_V4 ',1)[-1]
    marker=json.loads(resumed.splitlines()[0])
    paths=re.findall(r'output_dir:([^\r\n]+)',resumed)
    actual_path=paths[-1].strip() if paths else None
    restored={k:'Load '+k+'.state_dict' in resumed for k in ['model','optimizer','ema','scaler','lr_warmup_scheduler']}
    steps=[]
    for line in resumed.splitlines():
        match=re.search(r'Epoch:\s*(\d+)/(\d+)\s+\[\s*(\d+)/(\d+)\]',line)
        if match:
            timing=re.search(r'\btime:\s*([0-9.]+)',line)
            steps.append({'epoch_zero_based':int(match[1]),'iteration':int(match[3]),
                          'total_iterations':int(match[4]),'seconds_per_step':float(timing[1]) if timing else None})
    records=[json.loads(line) for line in (output/'log.txt').read_text().splitlines() if line.strip()]
    records=[r for r in records if 'test_coco_eval_bbox' in r]
    epochs=[r['epoch'] for r in records]
    expected=row['last_epoch_zero_based']+1
    progress=bool(steps and steps[0]['epoch_zero_based']==expected and
                  (steps[-1]['epoch_zero_based']>expected or steps[-1]['iteration']>=80))
    contiguous=epochs==list(range(epochs[-1]+1))
    first_new=any(e>=expected for e in epochs)
    item={'job':job,'source_checkpoint':row['backup_checkpoint'],'source_checkpoint_sha256':row['sha256'],
          'expected_output_directory':str(output),'native_output_directory':actual_path,
          'output_path_verified':actual_path==str(output) and marker['checkpoint']==row['backup_checkpoint']
                                 and pathlib.Path(marker['checkpoint']).parent==output,
          'restored_states':restored,'first_step':steps[0] if steps else None,'latest_step':steps[-1] if steps else None,
          'resume_progress_verified':progress,'evaluated_epochs':len(records),'metrics_contiguous':contiguous,
          'first_new_complete_evaluation_saved':first_new,'latest_evaluation':{'epoch':records[-1]['epoch'],
             'test_coco_eval_bbox':records[-1]['test_coco_eval_bbox']},
          'metrics_sha256':hashlib.sha256((output/'log.txt').read_bytes()).hexdigest(),
          'job_state':json.loads((phase/'jobs'/(job+'.json')).read_text()),
          'last_checkpoint_verified':False}
    if first_new:
        import torch
        checkpoint=output/'last.pth';before=checkpoint.stat()
        obj=torch.load(checkpoint,map_location='cpu',weights_only=False)
        digest=hashlib.sha256()
        with checkpoint.open('rb') as handle:
            for block in iter(lambda:handle.read(8*1024**2),b''):
                digest.update(block)
        item['last_checkpoint_sha256']=digest.hexdigest()
        after=checkpoint.stat()
        item['last_checkpoint_epoch_zero_based']=obj['last_epoch']
        item['last_checkpoint_verified']=(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns) and obj['last_epoch']==epochs[-1]
        del obj
    item['startup_verified']=item['output_path_verified'] and progress and all(restored.values()) and contiguous
    item['first_saved_epoch_verified']=item['startup_verified'] and first_new and item['last_checkpoint_verified']
    result['jobs'].append(item)
result['all_startups_verified']=all(r['startup_verified'] for r in result['jobs'])
result['all_first_saved_epochs_verified']=all(r['first_saved_epoch_verified'] for r in result['jobs'])
if result['all_first_saved_epochs_verified']:
    manifest=json.loads((stage/'protected_source_sha256.json').read_text())
    for relative,expected in manifest.items():
        assert hashlib.sha256((parent/'execution_v3'/relative).read_bytes()).hexdigest()==expected,relative
    result['protected_source_files_verified_after_restart']=len(manifest)
for name,args in [('gpu',['nvidia-smi','--query-gpu=name,uuid,memory.used,utilization.gpu','--format=csv,noheader']),
                  ('gpu_compute',['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'])]:
    p=subprocess.run(args,capture_output=True,text=True,timeout=20)
    result[name]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
for p in result['main_trainers']:
    items=(pathlib.Path('/proc')/str(p['pid'])/'environ').read_bytes().split(b'\0')
    p['environment']={k.decode():v.decode(errors='replace') for x in items for k,_,v in [x.partition(b'=')]
                       if k in [b'OMP_NUM_THREADS',b'MKL_NUM_THREADS',b'PYTHONUNBUFFERED',b'PYTHONDONTWRITEBYTECODE']}
print(json.dumps(result,ensure_ascii=False))
