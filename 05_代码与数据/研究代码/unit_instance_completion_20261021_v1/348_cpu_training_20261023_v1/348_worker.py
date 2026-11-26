"""One actual run, CPU only, using the unchanged and SHA-bound 347 entry.

The observer reads completed-step locals. Its only state-changing callback is the
original 347 save_state closure, before step0 (or on an explicit pause marker).
No training function, tensor, loss, parameter group or RNG is replaced.
"""
import ast
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')
C = importlib.import_module('348_common')
os.environ.update({k: v for k, v in C.cpu_env().items() if k in
                   ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'MKL_THREADING_LAYER')})
sys.path.insert(0, str(C.S347))
F = importlib.import_module('train347')


def equal(a, b):
    if isinstance(a, F.torch.Tensor):
        return isinstance(b, F.torch.Tensor) and F.torch.equal(a, b)
    if isinstance(a, F.np.ndarray):
        return isinstance(b, F.np.ndarray) and F.np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main(job_path):
    plan = C.plan()
    job = C.read(job_path)
    C.check(job['max_steps'] == 1218 and job['device'] == 'cpu', 'fixed CPU budget')
    identity = f"{job['arm']}_seed{job['seed']}"
    C.check(identity in plan['run_order'], 'unplanned run')
    logical = Path(job['logical_dir'])
    run_dir = Path(job['run_dir'])
    C.check(run_dir.parent == logical and logical.parent == C.HERE / '348_runs', 'run destination')
    progress_path = Path(job['progress_path'])
    outcome_path = Path(job['outcome_path'])
    references = []
    for record in plan['references'][identity]:
        C.verify_record(record['path'], record['sha256'])
        references.append((record['label'], C.read(record['path'])))
    module_ast = ast.parse((C.S347 / 'train347.py').read_text(encoding='utf-8'))
    function = next(n for n in module_ast.body if isinstance(n, ast.FunctionDef) and n.name == 'execute')
    main_loop = [n for n in function.body if isinstance(n, ast.While)]
    C.check(len(main_loop) == 1, 'one unchanged training loop')
    loop_line = main_loop[0].lineno
    first = True
    last_reported = None
    started = time.perf_counter()
    observed = {'completed_steps': None, 'new_optimizer_steps_observed': 0}

    with C.lock(logical / '348_worker.lock'), open(progress_path, 'x', encoding='utf-8', newline='\n') as progress:
        def publish(value):
            value = {'utc': C.now(), 'run_id': identity, **value}
            progress.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
            progress.flush()

        def observe(frame, event, arg):
            nonlocal first, last_reported
            if event != 'line' or frame.f_lineno != loop_line:
                return observe
            local = frame.f_locals
            step = local['step']
            if first:
                first = False
                last_reported = step
                observed['starting_step'] = step
                observed['completed_steps'] = step
                # A valid full-state snapshot even if interrupted before epoch1.
                before_rng = F.T.capture_rng()
                before_model = F.state_sha(local['model'].state_dict())
                local['save_state']()
                C.check(equal(before_rng, F.T.capture_rng()), 'checkpoint callback changed RNG')
                C.check(before_model == F.state_sha(local['model'].state_dict()), 'checkpoint callback changed model')
                publish({'event': 'initial_or_resume_checkpoint', 'completed_steps': step,
                         'rng_unchanged': True, 'model_unchanged': True,
                         'checkpoint': frame.f_locals['last_receipt']})
            if step > last_reported:
                C.check(step == last_reported + 1, 'observer lost a completed step')
                record = local['step_logs'][-1]
                observed.update(completed_steps=step,
                                new_optimizer_steps_observed=local['local_steps'])
                for label, rows in references:
                    if record['step'] < len(rows):
                        C.check(record == rows[record['step']], 'numeric reference mismatch: ' + label)
                publish({'event': 'completed_step', 'completed_steps': step, 'epoch': local['epoch'],
                         'batch_index': local['batch_index'], 'record': record,
                         'wall_seconds_this_process': time.perf_counter() - started})
                last_reported = step
            if (C.HERE / '348_pause_after_checkpoint.request').exists():
                destination = run_dir / f'347_checkpoint_step{step:06d}_receipt.json'
                if not destination.exists():
                    local['save_state']()
                publish({'event': 'explicit_pause_checkpoint', 'completed_steps': step,
                         'checkpoint': frame.f_locals['last_receipt']})
                raise InterruptedError('348 explicit pause marker')
            return observe

        def trace(frame, event, arg):
            return observe if event == 'call' and frame.f_code is F.execute.__code__ else None

        status, error = 'completed', None
        try:
            sys.settrace(trace)
            if job['resume'] is None:
                F.main(['--arm', job['arm'], '--seed', str(job['seed']), '--run_dir', str(run_dir),
                        '--device', 'cpu', '--max_steps', '1218'])
            else:
                F.execute(job['arm'], job['seed'], run_dir, 'cpu', 1218, _resume=job['resume'])
        except BaseException as exc:
            status = 'paused' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else '实现失败'
            error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
            traceback.print_exc()
        finally:
            sys.settrace(None)
        result_path = run_dir / '347_run_result.json'
        if status == 'completed':
            result = C.read(result_path)
            C.check(result['full_budget_completed'] is True and result['step'] == 1218, 'full budget completion')
            observed.update(completed_steps=1218,
                            new_optimizer_steps_observed=result['counts_this_process']['optimizer_steps'])
            publish({'event': 'completed', 'completed_steps': 1218, 'last_checkpoint': result['last_checkpoint']})
        C.write(outcome_path, {'status': status, 'run_id': identity, 'pid': os.getpid(),
                              'job_path': str(job_path), 'run_dir': str(run_dir), 'observed': observed,
                              'error': error, 'wall_seconds': time.perf_counter() - started,
                              'completed_utc': C.now()})
        C.emit({'run_id': identity, 'status': status, **observed})
        if status != 'completed':
            raise SystemExit(75 if status == 'paused' else 1)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
