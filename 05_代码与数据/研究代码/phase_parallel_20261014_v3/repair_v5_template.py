"""Preflight and prepare isolated per-job resume paths, without starting training."""
import ast
import base64
import datetime
import gc
import hashlib
import json
import math
import pathlib
import shutil
import subprocess
import sys
import time

PAYLOAD = None  # Generated from the previously verified local deployment evidence.
phase = pathlib.Path(PAYLOAD['phase'])
parent = phase/'artifacts/round2_20261014_v1'
old = parent/'execution_v4'
code = parent/'execution_v3/code'
stage = parent/'execution_v5'
trace = parent/('repair_'+PAYLOAD['action']+'_v5_'+PAYLOAD.get('attempt_tag','initial')+'.jsonl')
trace_handle = trace.open('x')


def progress(name, **fields):
    line=json.dumps({'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     'stage':name,**fields})
    trace_handle.write(line+'\n');trace_handle.flush()
    print(line,file=sys.stderr,flush=True)


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def live_processes():
    result = []
    for path in pathlib.Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            args = [x.decode(errors='replace') for x in (path/'cmdline').read_bytes().split(b'\0') if x]
            names = {pathlib.PurePosixPath(x).name for x in args}
            stat = (path/'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] != 'Z' and any(str(phase) in x for x in args) and names & {
                'train.py', 'train_round2.py', 'run_parallel_queue.py', 'resume_parallel_queue_v4.py',
                'resume_parallel_queue_v5.py', 'dump_preds.py', 'run_round2_queue.py'}:
                result.append({'pid':int(path.name),'argv':args,'start_ticks':int(stat[19]),'state':stat[0]})
        except (OSError, ValueError, IndexError):
            pass
    return result


assert phase.is_dir()
progress('checking_frozen_sources')
assert json.loads((parent/'current_execution.json').read_text())['execution_directory'] == 'execution_v4'
assert not stage.exists(), 'Refusing to overwrite an existing execution_v5'
assert not live_processes(), 'Live experiment process exists; inspect exact identities before repair'
assert shutil.disk_usage(phase).free > 5*1024**3
resume = json.loads((old/'resume_plan.json').read_text())
state = json.loads((old/'queue_state.json').read_text())
assert resume['completed_jobs'] == PAYLOAD['completed_jobs'] == state['training_completed_jobs']
assert sha(phase/'artifacts/common_init_seed0.pth') == PAYLOAD['common_init_sha256']
for relative, expected in PAYLOAD['protected_source_sha256'].items():
    assert sha(parent/'execution_v3'/relative) == expected, relative
for relative, expected in PAYLOAD['code_hashes'].items():
    assert sha(code/relative) == expected, relative
progress('frozen_sources_verified',files=len(PAYLOAD['protected_source_sha256']))
solver_source = (code/'engine/solver/_solver.py').read_text()
assert 'self.output_dir = Path(os.path.dirname(str(self.cfg.resume)))' in solver_source
report = {'account':PAYLOAD['account'],'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'source_files_verified':len(PAYLOAD['protected_source_sha256']),
          'code_hashes':PAYLOAD['code_hashes'],'common_init_sha256':PAYLOAD['common_init_sha256'],
          'existing_processes':live_processes(),'completed_jobs':resume['completed_jobs'],'resume_jobs':[]}
progress('loading_pytorch_runtime')
import torch
progress('pytorch_imported',version=torch.__version__)
report['torch_version'] = torch.__version__
report['gpu_available'] = torch.cuda.is_available()
progress('cuda_checked',available=report['gpu_available'])
if shutil.which('nvidia-smi'):
    p = subprocess.run(['nvidia-smi','--query-gpu=name,uuid,memory.free','--format=csv,noheader'],capture_output=True,text=True,timeout=20)
    report['gpu'] = {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
else:
    report['gpu'] = {'returncode':127,'stdout':'','stderr':'nvidia-smi unavailable'}
for expected in PAYLOAD['resume_jobs']:
    job = expected['job']
    progress('checking_checkpoint',job=job)
    row = next(r for r in resume['resume_jobs'] if r['job']==job)
    original_input = pathlib.Path(row['backup_checkpoint'])
    original_last = phase/'outputs'/job/'last.pth'
    assert original_input.parent == old/'resume_inputs'
    assert original_input.name == job+'_last.pth'
    assert sha(original_input) == sha(original_last) == row['sha256'] == expected['sha256']
    obj = torch.load(original_input,map_location='cpu',weights_only=False)
    required = ['model','optimizer','ema','scaler','lr_warmup_scheduler','best_stat']
    assert all(key in obj for key in required)
    assert obj['last_epoch'] == row['last_epoch_zero_based'] == expected['last_epoch_zero_based']
    assert len(obj['optimizer']['state']) == expected['optimizer_state_entries']
    assert obj['ema']['updates'] == expected['ema_updates']
    rows = [json.loads(line) for line in (phase/'outputs'/job/'log.txt').read_text().splitlines() if line.strip()]
    epochs = [r['epoch'] for r in rows if 'test_coco_eval_bbox' in r]
    assert epochs == row['logged_epochs'] == list(range(obj['last_epoch']+1))
    destination = phase/'outputs'/job/'resume_repair_v5_input.pth'
    assert not destination.exists()
    report['resume_jobs'].append({'job':job,'source':str(original_input),'source_sha256':row['sha256'],
        'destination':str(destination),'last_epoch_zero_based':obj['last_epoch'],
        'next_round_one_based':obj['last_epoch']+2,'optimizer_state_entries':len(obj['optimizer']['state']),
        'ema_updates':obj['ema']['updates'],'all_training_states_present':True,'logged_epochs':epochs})
    del obj
    gc.collect()
    progress('checkpoint_verified',job=job,next_round=expected['next_round_one_based'])
for job in resume['completed_jobs']:
    output = phase/'outputs'/job
    rows = [json.loads(line) for line in (output/'log.txt').read_text().splitlines() if line.strip()]
    rows = [r for r in rows if 'test_coco_eval_bbox' in r]
    assert [r['epoch'] for r in rows] == list(range(96))
    assert all(len(r['test_coco_eval_bbox'])==12 and all(math.isfinite(x) for x in r['test_coco_eval_bbox']) for r in rows)
    assert (output/'best_stg2.pth').is_file()
    job_state = json.loads((phase/'jobs'/(job+'.json')).read_text())
    assert job_state['state'] in ['complete','trained_pending_independent_eval'] and job_state['returncode']==0
report['status'] = 'preflight_passed'
if PAYLOAD['action']=='preflight':
    progress('preflight_passed')
    print(json.dumps(report,ensure_ascii=False))
    raise SystemExit(0)
assert PAYLOAD['action']=='prepare'
assert not live_processes()
stage.mkdir()
archive = stage/'before_repair'
archive.mkdir()
shutil.copytree(old/'resume_inputs',archive/'shared_v4_resume_inputs')
progress('shared_artifacts_archived')
for name in ['queue_state.json','queue_plan.json','resume_plan.json','queue_console.log','queue_events.jsonl','launch_receipt.json']:
    if (old/name).is_file():
        shutil.copy2(old/name,archive/name)
shutil.copy2(parent/'current_execution.json',archive/'previous_execution_pointer.json')
shutil.copy2(old/'queue_plan.json',stage/'queue_plan.json')
controller = base64.b64decode(PAYLOAD['controller_base64'])
assert hashlib.sha256(controller).hexdigest()==PAYLOAD['controller_sha256']
ast.parse(controller.decode('utf-8'))
(stage/'resume_parallel_queue_v5.py').write_bytes(controller)
for row in resume['resume_jobs']:
    entry = next(r for r in report['resume_jobs'] if r['job']==row['job'])
    source = pathlib.Path(entry['source']); destination = pathlib.Path(entry['destination'])
    backup = archive/row['job']; backup.mkdir()
    for path in [phase/'outputs'/row['job']/'log.txt',phase/'outputs'/row['job']/'args.json',
                 phase/'jobs'/(row['job']+'.json'),phase/'logs'/(row['job']+'.log')]:
        if path.is_file():
            shutil.copy2(path,backup/('console.log' if path.parent.name=='logs' else path.name))
    for name in ['last.pth','best_stg1.pth','best_stg2.pth']:
        path = phase/'outputs'/row['job']/name
        if path.is_file():
            shutil.copy2(path,backup/name)
    with destination.open('xb') as handle, source.open('rb') as original:
        shutil.copyfileobj(original,handle,8*1024**2)
    assert sha(destination)==entry['source_sha256']
    row.update(original_v4_backup_checkpoint=str(source),backup_checkpoint=str(destination),
               backup_sha256=entry['source_sha256'])
resume.update(status='ready_to_resume',prepared_at=time.time(),previous_execution='execution_v4',
              repair='Unique immutable resume input inside each original job output directory; shared v4 artifacts archived',
              maximum_concurrent_trainings=2)
(stage/'resume_plan.json').write_text(json.dumps(resume,indent=2))
(stage/'protected_source_sha256.json').write_text(json.dumps(PAYLOAD['protected_source_sha256'],indent=2))
manifest = {str(path.relative_to(archive)):sha(path) for path in sorted(archive.rglob('*')) if path.is_file()}
(stage/'archive_sha256.json').write_text(json.dumps(manifest,indent=2))
report.update(status='ready_to_resume',stage=str(stage),archive_files=len(manifest),training_started=False,
              controller_sha256=PAYLOAD['controller_sha256'])
(stage/'repair_preparation_receipt.json').write_text(json.dumps(report,indent=2))
progress('preparation_complete',archive_files=len(manifest))
print(json.dumps(report,ensure_ascii=False))
