"""One-shot controller for the explicitly authorized bounded BASE diagnostic."""
from pathlib import Path
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
import traceback

import diagnostic_contract as c

child = None
state = {}

def save(status, **values):
    state.update(status=status, updated_utc=c.now(), **values)
    c.atomic_json(c.ROOT / 'controller_state.json', state)

def tick(pid):
    return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]

def require_free_gpu(expected_uuid):
    deadline = time.monotonic() + 30
    while True:
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,memory.used',
                                       '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        c.require(len(gpu) == 5 and gpu[0] == 'NVIDIA GeForce RTX 3090' and gpu[1] == expected_uuid
                  and gpu[2] == '550.100' and int(gpu[3]) == 24576, 'Expected GPU changed')
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                            text=True).strip()
        if int(gpu[4]) < 256 and not processes:
            return
        c.require(time.monotonic() < deadline, 'GPU is not free')
        time.sleep(2)

def child_setup(parent_pid):
    os.setsid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM) != 0:  # PR_SET_PDEATHSIG
        os._exit(126)
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)

def stop_child():
    global child
    if child is not None and child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()

def on_signal(signum, frame):
    raise SystemExit('Controller received signal %d' % signum)

def phase(name, expected_uuid, **extra):
    global child
    require_free_gpu(expected_uuid)
    dispatch = {'status': 'DISPATCHED', 'run_id': c.RUN_ID, 'phase': name,
                'authorization_sha256': c.AUTH_SHA, 'created_utc': c.now(), **extra}
    c.atomic_json(c.ROOT / 'dispatch.json', dispatch)
    log_path = c.ROOT / 'logs' / (name + '.log')
    with log_path.open('x', encoding='utf-8') as log:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONHASHSEED='0')
        parent_pid = os.getpid()
        child = subprocess.Popen([sys.executable, '-B', '-u', str(c.ROOT / 'diagnostic_worker.py'), '--phase', name],
                                 cwd=str(c.ROOT), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, preexec_fn=lambda: child_setup(parent_pid))
        started = time.monotonic()
        save('PHASE_RUNNING', current_phase=name, child_pid=child.pid, child_start_ticks=tick(child.pid),
             phase_started_utc=c.now())
        while child.poll() is None:
            time.sleep(10)
            save('PHASE_RUNNING', current_phase=name, phase_elapsed_seconds=time.monotonic() - started)
        code = child.wait()
        save('PHASE_EXITED', current_phase=name, child_exit_code=code,
             phase_elapsed_seconds=time.monotonic() - started)
        child = None
    c.require(code == 0, 'Diagnostic phase failed: ' + name)
    result = c.read(c.ROOT / ('worker_result_' + name + '.json'))
    return result

def main():
    c.auth_check()
    c.package_check()
    ready = c.read(c.ROOT / 'setup_ready.json')
    c.require(ready['status'] == 'SETUP_READY_NO_MODEL_EXECUTION' and
              ready['code_manifest_sha256'] == c.sha(c.ROOT / 'code_manifest.json'), 'Setup not ready')
    c.require(not (c.ROOT / 'controller_state.json').exists(), 'Controller already attempted')
    with (c.ROOT / 'controller.one_shot.lock').open('x', encoding='utf-8') as f:
        f.write(str(os.getpid()) + '\n')
        f.flush()
        os.fsync(f.fileno())
    c.fsync_dir(c.ROOT)
    state.update(run_id=c.RUN_ID, controller_pid=os.getpid(), controller_start_ticks=tick(os.getpid()),
                 authorization_sha256=c.AUTH_SHA, code_manifest_sha256=c.sha(c.ROOT / 'code_manifest.json'),
                 setup_ready_sha256=c.sha(c.ROOT / 'setup_ready.json'), automatic_retries=False,
                 max_optimizer_updates=c.MAX_UPDATES, max_full_validations=c.MAX_VALIDATIONS,
                 started_utc=c.now())
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        gpu_uuid = ready['gpu']['uuid']
        save('STARTING', expected_gpu_uuid=gpu_uuid)
        initial_ema = phase('initial_ema', gpu_uuid)
        saved = c.read(c.ROOT / 'input/best.metrics.json')
        observed = float(initial_ema['metrics']['AP'])
        difference = observed - float(saved['metrics']['AP'])
        if abs(difference) > 0.001:
            save('STOPPED_INITIAL_EMA_AP_MISMATCH', observed_AP=observed, saved_AP=saved['metrics']['AP'],
                 AP_difference=difference, completed_validations=1, optimizer_updates=0, completed_utc=c.now())
            return
        initial_raw = phase('initial_raw', gpu_uuid)
        train = phase('train', gpu_uuid, initial_ema_AP=observed)
        c.require(train['status'] in ['AP_DROP_STOP', 'MAX_EPOCHS_STOP'], 'Unexpected training stop')
        c.require(train['optimizer_updates'] <= c.MAX_UPDATES and train['validation_passes'] <= c.MAX_VALIDATIONS - 1,
                  'Training budget exceeded')
        terminal_raw = phase('terminal_raw', gpu_uuid)
        c.require(terminal_raw['epoch_index'] == train['epoch_index'], 'Terminal raw epoch mismatch')
        total_validations = train['validation_passes'] + 1
        c.require(total_validations <= c.MAX_VALIDATIONS, 'Validation budget exceeded')
        result = {'status': 'DIAGNOSTIC_COMPLETE_NOT_A_FAIR_PAIR', 'completed_utc': c.now(),
                  'run_id': c.RUN_ID, 'authorization_sha256': c.AUTH_SHA,
                  'initial_saved_AP': saved['metrics']['AP'], 'initial_ema': initial_ema,
                  'initial_raw': initial_raw, 'training': train, 'terminal_raw': terminal_raw,
                  'optimizer_updates': train['optimizer_updates'], 'full_validation_passes': total_validations,
                  'images_per_validation': 548, 'automatic_retries': False,
                  'interpretation_limit': 'New diagnostic trajectory; not exact continuation, not a BASE/FULL fair comparison, and not proof of a cause.'}
        c.atomic_json(c.ROOT / 'diagnostic_result.json', result, new=True)
        save(result['status'], optimizer_updates=result['optimizer_updates'],
             completed_validations=total_validations, completed_utc=c.now(),
             result_sha256=c.sha(c.ROOT / 'diagnostic_result.json'))
    except BaseException:
        stop_child()
        save('STOPPED_NO_AUTOMATIC_RETRY', failure=traceback.format_exc(), completed_utc=c.now())
        raise

if __name__ == '__main__':
    main()
