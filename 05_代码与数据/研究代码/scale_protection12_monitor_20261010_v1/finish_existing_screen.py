"""Wait on an owned screen, then run frozen reload/collector once. No training."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().split()
    return dict(pid=pid, start_ticks=fields[21], state=fields[2])


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def inspect(bundle, tools, receipt_sha):
    assert sha(bundle / 'receipt.json') == receipt_sha
    sys.path.insert(0, str(tools))
    from verify_candidate_reload import verify_tools, bound_runtime
    from collect_terminal import terminal_state
    protocol = verify_tools()
    _, receipt, plan, paths = bound_runtime(bundle)
    assert plan['max_epochs'] == 12 and plan['recipe_epochs'] == 132
    assert paths['screen_root'] == bundle.parent
    assert protocol['candidate_forward_batches'] == 137 and protocol['validation_images'] == 548
    assert protocol['reload_timeout_seconds'] == 1800
    assert not any(n == 'torch' or n.startswith('engine.') for n in sys.modules)
    return receipt, plan, paths, terminal_state, protocol


def finish(bundle, tools, receipt_sha, check_only):
    bundle, tools = Path(bundle).resolve(), Path(tools).resolve()
    receipt, plan, paths, terminal_state, protocol = inspect(bundle, tools, receipt_sha)
    screen = bundle.parent
    state = terminal_state(screen)
    if check_only:
        return dict(status='CHECK_ONLY_NO_ACTION', seed=plan['seed'], arm=plan['arm'],
            actual_terminal_state=state, source_sha256=sha(__file__),
            receipt_sha256=receipt_sha, new_model_forwards=0, epochs_added=0,
            process_records={name: read(screen / name) for name in
                             ('candidate_start.json', 'controller_start.json')})
    assert state in ('OWNED_TRAINING_RUNNING', 'FIXED12_RELOAD_PENDING', 'FAILED_NO_FIXED12_ACCEPTANCE'), state
    assert not (screen / 'candidate_reload_v1').exists()
    out = paths['experiment_root'] / f"post_training_seed{plan['seed']}_{plan['arm']}_v1"
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    write(out / 'start.json', dict(**identity(os.getpid()), source_sha256=sha(__file__),
        bundle=str(bundle), tools=str(tools), identity=receipt['identity'],
        receipt_sha256=receipt_sha, command=sys.argv, started_remote_UTC=now(),
        observation_timeout_seconds=64800, automatic_retry=False, training_commands=0,
        note='Only frozen candidate reload and terminal collector after actual owned training processes stop. No promotion or random-control launch.'))

    def live(phase, **fields):
        temporary = out / 'live.tmp.json'
        temporary.write_text(json.dumps(dict(phase=phase, elapsed_seconds=time.monotonic()-started,
            observed_remote_UTC=now(), **fields)), encoding='utf8')
        temporary.replace(out / 'live.json')

    def execute(phase, command):
        with (out / f'{phase}.log').open('xb') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                         OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'))
            record = dict(**identity(child.pid), command=command, started_remote_UTC=now())
            write(out / f'{phase}_start.json', record)
            while child.poll() is None:
                live(phase, child=record)
                time.sleep(5)
            code = child.wait()
            write(out / f'{phase}_completion.json', dict(exit_code=code,
                completed_remote_UTC=now(), automatic_retry=False))
            return code

    try:
        while state == 'OWNED_TRAINING_RUNNING':
            live('waiting_owned_training', actual_terminal_state=state)
            if time.monotonic()-started >= 64800:
                raise TimeoutError('Observation expired. Existing trainer is not restarted or terminated.')
            time.sleep(15)
            state = terminal_state(screen)
        reload_exit = None
        if state == 'FIXED12_RELOAD_PENDING':
            reload_exit = execute('candidate_reload', [sys.executable, '-B',
                str(tools / 'verify_candidate_reload.py'), '--bundle', str(bundle)])
            state = terminal_state(screen)
        assert state in ('FAILED_NO_FIXED12_ACCEPTANCE', 'FIXED12_RELOAD_FAILED', 'FIXED12_AND_RELOAD_COMPLETE'), state
        collection_exit = execute('terminal_collection', [sys.executable, '-B',
            str(tools / 'collect_terminal.py'), '--bundle', str(bundle)])
        assert collection_exit == 0, 'Collector failed; preserve output, no automatic retry'
        metadata = paths['experiment_root'] / f"terminal_{plan['arm']}_seed{plan['seed']}_v1_metadata.json"
        result = dict(status='POST_TRAINING_RELOAD_AND_COLLECTION_FINISHED', seed=plan['seed'],
            arm=plan['arm'], terminal_state=state, reload_exit_code=reload_exit,
            collection_exit_code=collection_exit, archive_metadata=read(metadata),
            archive_metadata_sha256=sha(metadata), completed_remote_UTC=now(),
            automatic_retry=False, training_commands=0, baseline_model_forwards=0,
            epochs_added=0, random_controls_launched=0, automatic_promotion=False)
        write(out / 'completion.json', result)
        live('terminal', terminal_state=state)
        return result
    except BaseException:
        write(out / 'failure.json', dict(status='POST_TRAINING_PIPELINE_FAILED_NO_RETRY',
            traceback=traceback.format_exc(), observed_remote_UTC=now(), training_commands=0,
            baseline_model_forwards=0, epochs_added=0, automatic_retry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('bundle', 'tools', 'receipt-sha'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(finish(args.bundle, args.tools, args.receipt_sha, args.check_only)), flush=True)
