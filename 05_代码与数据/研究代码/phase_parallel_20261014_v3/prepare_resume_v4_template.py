"""Create a new controller directory and immutable local-volume resume copies."""
import base64
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import time

PAYLOAD = None  # Replaced in a generated deployment script; contains no credentials.
account = PAYLOAD['account']
phase = pathlib.Path('/25085400120/DEIM/experiments/phase_mechanism_20261012' if account == 'A'
                     else '/24085403014/experiments/phase_mechanism_20261012')
old = phase/'artifacts/round2_20261014_v1/execution_v3'
stage = old.with_name('execution_v4')
assert not stage.exists(), 'Do not overwrite an existing recovery execution'
assert shutil.disk_usage(phase).free > 2*1024**3
state = json.loads((old/'queue_state.json').read_text())
assert state['training_completed_jobs'] == PAYLOAD['preflight']['completed_jobs']
for path in pathlib.Path('/proc').iterdir():
    if path.name.isdigit():
        try:
            args = (path/'cmdline').read_bytes().decode(errors='replace')
            assert not (str(phase) in args and any(name in args for name in ['train.py', 'train_round2.py', 'run_parallel_queue.py', 'resume_parallel_queue_v4.py'])), 'Live experiment process present'
        except OSError:
            pass
for relative, expected in PAYLOAD['protected_source_sha256'].items():
    assert hashlib.sha256((old/relative).read_bytes()).hexdigest() == expected, relative
for relative, expected in PAYLOAD['preflight']['code_hashes'].items():
    assert hashlib.sha256((old/'code'/relative).read_bytes()).hexdigest() == expected, relative
    if 'code/'+relative not in PAYLOAD['protected_source_sha256']:
        assert hashlib.sha256((phase/'code'/relative).read_bytes()).hexdigest() == expected, relative
assert hashlib.sha256((phase/'artifacts/common_init_seed0.pth').read_bytes()).hexdigest() == PAYLOAD['preflight']['common_init_sha256']
stage.mkdir()
(stage/'resume_inputs').mkdir()
(stage/'before_resume').mkdir()
shutil.copy2(old/'queue_plan.json', stage/'queue_plan.json')
shutil.copy2(old/'queue_state.json', stage/'before_resume/previous_v3_queue_state.json')
shutil.copy2(old.parent/'current_execution.json', stage/'before_resume/previous_execution_pointer.json')
controller = base64.b64decode(PAYLOAD['controller_base64'])
assert hashlib.sha256(controller).hexdigest() == PAYLOAD['controller_sha256']
with (stage/'resume_parallel_queue_v4.py').open('xb') as handle:
    handle.write(controller)
subprocess.run([sys.executable, '-B', '-c', 'import ast,pathlib; ast.parse(pathlib.Path("resume_parallel_queue_v4.py").read_text())'], cwd=stage, check=True)
for row in PAYLOAD['preflight']['resume_jobs']:
    job = row['job']
    destination = stage/'resume_inputs'/(job+'_last.pth')
    shutil.copy2(row['checkpoint'], destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    assert digest == row['sha256']
    row.update(backup_checkpoint=str(destination), backup_sha256=digest)
    backup = stage/'before_resume'/job
    backup.mkdir()
    for source in [phase/'outputs'/job/'args.json', phase/'outputs'/job/'log.txt',
                   phase/'logs'/(job+'.log'), phase/'jobs'/(job+'.json')]:
        shutil.copy2(source, backup/source.name if source.parent.name != 'logs' else backup/'console.log')
    assert hashlib.sha256((backup/'args.json').read_bytes()).hexdigest() == row['args_sha256']
for job in state['training_completed_jobs']:
    output = phase/'outputs'/job
    rows = [json.loads(line) for line in (output/'log.txt').read_text().splitlines() if line.strip()]
    assert [r['epoch'] for r in rows if 'test_coco_eval_bbox' in r] == list(range(96))
    assert (output/'best_stg2.pth').is_file()
    status = json.loads((phase/'jobs'/(job+'.json')).read_text())
    assert status['returncode'] == 0 and status['state'] in ['trained_pending_independent_eval', 'complete']
resume = dict(PAYLOAD['preflight'], completed_jobs=state['training_completed_jobs'], status='ready_to_resume',
              prepared_at=time.time(), source_directory=str(old/'code'), maximum_concurrent_trainings=2,
              fresh_pending_jobs=[v for v in json.loads((old/'queue_plan.json').read_text())['jobs']
                                  if v.replace(':', '_seed') not in state['training_completed_jobs']
                                  and v.replace(':', '_seed') not in state['active_jobs']])
with (stage/'resume_plan.json').open('x') as handle:
    json.dump(resume, handle, indent=2)
print(json.dumps({'account': account, 'stage': str(stage), 'status': resume['status'],
                  'resumed_jobs': [{'job': r['job'], 'next_round': r['next_round_one_based']} for r in resume['resume_jobs']],
                  'completed_preserved': resume['completed_jobs'], 'fresh_pending_jobs': resume['fresh_pending_jobs'],
                  'training_started': False}))
