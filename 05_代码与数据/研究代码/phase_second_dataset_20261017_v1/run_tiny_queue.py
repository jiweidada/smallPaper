"""Bounded four-job account queue, activated only after both original queues finish."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
PLAN = json.loads((ROOT / 'plan.json').read_text())
ENV = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
           PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
STATE = {'account': PLAN['account'], 'dataset': 'TinyPerson_v1', 'state': 'starting',
         'pid': os.getpid(), 'active_jobs': {}, 'training_completed_jobs': [],
         'completed_jobs': [], 'planned_jobs': PLAN['jobs'], 'retries': 0,
         'start_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}


def write(path, value):
    temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(value, indent=2)); os.replace(temporary, path)


def state(name=None, **extra):
    STATE.update(extra, checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if name: STATE['state'] = name
    write(ROOT / 'queue_state.json', STATE)


def jobspec(arm): return f'{arm}_full96_seed{PLAN["seed"]}'


def verify_training(job):
    output = ROOT / 'outputs' / job
    rows = [json.loads(line) for line in (output/'log.txt').read_text().splitlines() if line.strip()]
    rows = [row for row in rows if 'test_tiny_eval_bbox' in row]
    assert [row['epoch'] for row in rows] == list(range(96)), 'Missing or repeated training evaluation epochs'
    import torch
    cp_path = output/'best_stg2.pth'; checkpoint = torch.load(cp_path, map_location='cpu', weights_only=False)
    epoch = checkpoint['last_epoch']; assert 84 <= epoch <= 95
    selected = rows[epoch]['test_tiny_eval_bbox']; assert len(selected) == 42
    assert checkpoint['best_stat']['epoch'] == epoch
    assert abs(checkpoint['best_stat']['tiny_eval_bbox'] - selected[0]) < 1e-12
    from tiny_protocol import METRIC_NAMES
    assert METRIC_NAMES[0] == 'AP50_tiny'
    proof = {'job': job, 'epochs': 96, 'checkpoint': str(cp_path),
             'checkpoint_sha256': hashlib.sha256(cp_path.read_bytes()).hexdigest(),
             'best_round_one_based': epoch+1, 'metric_names': METRIC_NAMES,
             'metrics_raw_0_to_1': dict(zip(METRIC_NAMES, selected)),
             'log_sha256': hashlib.sha256((output/'log.txt').read_bytes()).hexdigest()}
    with (output/'checkpoint_log_binding.json').open('x') as handle: json.dump(proof, handle, indent=2)
    return proof


def run():
    release = json.loads((ROOT/'release_after_round2.json').read_text())
    assert release['both_round2_queues_complete'] is True
    assert release['plan_sha256'] == hashlib.sha256((ROOT/'plan.json').read_bytes()).hexdigest()
    assert json.loads((ROOT/'preparation_receipt.json').read_text())['status'] == 'cpu_verified_ready_for_deferred_gpu_preflight'
    with (ROOT/'queue.lock').open('x') as handle: handle.write(str(os.getpid()))
    query = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    assert not any(line.strip().isdigit() for line in query.splitlines()), 'GPU is not idle at handoff'
    assert hashlib.sha256(Path(PLAN['initialization']).read_bytes()).hexdigest() == PLAN['initialization_sha256']
    # The first new GPU work is explicitly delayed until this global completion gate.
    state('deferred_gpu_preflight')
    with (ROOT/'logs/gpu_preflight.log').open('xb') as log:
        checked = subprocess.run([sys.executable,'-u',str(ROOT/'code/gpu_preflight_tiny.py'),str(ROOT)],
                                 cwd=ROOT/'code',env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    assert checked.returncode == 0, 'Deferred GPU preflight failed; no training starts'
    pending = list(PLAN['jobs']); active = {}; failed = False
    state('training')
    while pending or active:
        for job, info in list(active.items()):
            code = info['process'].poll()
            if code is None: continue
            info['handle'].close(); active.pop(job); STATE['active_jobs'].pop(job)
            record = json.loads((ROOT/'jobs'/(job+'.json')).read_text()); record['returncode'] = code
            if code == 0:
                try:
                    record['best'] = verify_training(job); record['state'] = 'trained_pending_independent_eval'
                    STATE['training_completed_jobs'].append(job)
                except Exception as exc:
                    record.update(state='failed', error=str(exc)); failed = True
            else: record.update(state='failed'); failed = True
            write(ROOT/'jobs'/(job+'.json'),record)
        while pending and len(active) < 2 and not failed:
            arm = pending[0]; job = jobspec(arm); output = ROOT/'outputs'/job
            if output.exists() or (ROOT/'jobs'/(job+'.json')).exists(): raise FileExistsError('Existing experiment: '+job)
            free = subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
            if int(free.strip().splitlines()[0]) < 10240: break
            command = [sys.executable,'-u',str(ROOT/'code/train_tinyperson.py'),'-c',PLAN['configs'][arm],
                       '--seed',str(PLAN['seed']),'--use-amp','-t',PLAN['initialization'],'--output-dir',str(output)]
            handle = (ROOT/'logs'/(job+'.log')).open('xb')
            process = subprocess.Popen(command,cwd=ROOT/'code',env=ENV,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
            active[job] = {'process': process, 'handle': handle}; pending.pop(0)
            STATE['active_jobs'][job] = {'pid':process.pid,'output':str(output)}
            write(ROOT/'jobs'/(job+'.json'),{'job':job,'arm':arm,'seed':PLAN['seed'],'state':'running','pid':process.pid})
        state('draining_after_failure' if failed else 'training', not_started=pending)
        if failed and not active: raise RuntimeError('Training failed; no automatic retries or later jobs')
        time.sleep(10)
    state('independent_evaluation')
    for arm in PLAN['jobs']:
        job = jobspec(arm); checkpoint = ROOT/'outputs'/job/'best_stg2.pth'
        for split in ['val','test']:
            destination = ROOT/'results'/job/split
            with (ROOT/'logs'/(job+'_'+split+'_eval.log')).open('xb') as log:
                result = subprocess.run([sys.executable,'-u',str(ROOT/'code/evaluate_tiny_checkpoint.py'),
                    '--config',PLAN['configs'][arm],'--checkpoint',str(checkpoint),'--data',PLAN['data_root'],
                    '--split',split,'--out',str(destination)],cwd=ROOT/'code',env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            assert result.returncode == 0, 'Independent evaluation failed; no retry: '+job+' '+split
        record = json.loads((ROOT/'jobs'/(job+'.json')).read_text());record.update(state='complete',independent_evaluation_complete=True)
        write(ROOT/'jobs'/(job+'.json'),record); STATE['completed_jobs'].append(job);state(completed_job=job)
    state('complete')


if __name__ == '__main__':
    try: run()
    except BaseException as exc:
        state('failed',error=type(exc).__name__+': '+str(exc));raise

