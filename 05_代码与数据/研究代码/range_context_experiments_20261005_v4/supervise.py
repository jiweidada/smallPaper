"""Bounded capacity OR train24 stage; no retries, 96 runs or old queues."""
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    with path.open('x') as handle:
        json.dump(data, handle, indent=2)


def ticks(pid):
    return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]


def run(root, stage):
    jobs, code = root / 'jobs', root / 'code'
    lock = (jobs / 'stage.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if stage not in ('capacity', 'train24'):
        raise ValueError(stage)
    runtime_path = root / 'runtime.json'
    runtime = json.loads(runtime_path.read_text())
    runtime_hash = sha(runtime_path)
    manifest_path = root / 'bundle_manifest.json'
    manifest_hash = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    protocol = json.loads((code / 'protocol.json').read_text())
    variant = protocol['planned_accounts'][runtime['account']][0]
    assert variant == runtime['variant']
    uuid, python = runtime['GPU_uuid'], runtime['python']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONHASHSEED='1',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    env.pop('PYTHONPATH', None)
    current = [None]
    def stopped(signum, frame):
        p = current[0]
        if p is not None and p.poll() is None:
            p.terminate()
        raise RuntimeError('Explicit stop received; preserve partial outputs')
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    def verify():
        assert sha(manifest_path) == manifest_hash and sha(runtime_path) == runtime_hash
        for row in manifest['files']:
            assert sha(root / row['path']) == row['sha256'], row['path']
    def child(name, command, gpu=True):
        verify()
        if shutil.disk_usage(root).free < 40 * 2**30:
            raise RuntimeError('Insufficient disk; preserve all old/new material')
        if gpu and subprocess.check_output(['nvidia-smi', '--id', uuid, '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU occupied; do not stop or compete with another process')
        with (jobs / (name + '.log')).open('x') as log:
            p = subprocess.Popen(command, cwd=code, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            current[0] = p
            identity = {'pid': p.pid, 'start_ticks': ticks(p.pid), 'argv': command,
                        'time': time.time(), 'runtime_sha256': runtime_hash, 'bundle_manifest_sha256': manifest_hash}
            save(jobs / (name + '_started.json'), identity)
            (jobs / 'status.json').write_text(json.dumps({'status': 'running', 'stage': name, **identity}))
            rc = p.wait()
            current[0] = None
        save(jobs / (name + '_exit.json'), {'exit_code': rc, 'time': time.time()})
        if rc:
            raise RuntimeError(name + ' failed; no automatic retry or next stage')
    common = ['--common-init', runtime['init'], '--train-images', runtime['train_images'],
              '--train-annotations', runtime['train_ann'], '--val-images', runtime['val_images'],
              '--val-annotations', runtime['val_ann'], '--gpu-uuid', uuid]
    cfg = code / 'configs' / (variant + '.yml')
    capdir = root / 'outputs' / ('capacity_' + variant)
    if stage == 'capacity':
        child('preflight', [python, '-u', str(code / 'preflight.py'), str(root)], gpu=False)
        child('adapter_cpu', [python, '-u', str(code / 'verify_adapter_cpu.py'), '--config', str(cfg),
                             '--init', runtime['init'], '--images', runtime['val_images'],
                             '--annotations', runtime['val_ann'], '--output', str(jobs / 'adapter_cpu.json')], gpu=False)
        child('capacity', [python, '-u', str(code / 'run_candidate.py'), 'capacity', '--config', str(cfg),
                           *common, '--output', str(capdir)])
        cap = json.loads((capdir / 'capacity.json').read_text())
        assert cap['status'] == 'passed_capacity_only'
        save(jobs / 'capacity_stage_complete.json', {'status': 'capacity_passed_waiting_for_pair_audit',
              'capacity_sha256': sha(capdir / 'capacity.json'), 'variant': variant,
              'bundle_manifest_sha256': manifest_hash, 'runtime_sha256': runtime_hash, 'time': time.time()})
        (jobs / 'status.json').write_text(json.dumps({'status': 'capacity_passed_waiting_for_pair_audit', 'time': time.time()}))
        return
    authorization = json.loads((jobs / 'train24_authorization.json').read_text())
    assert authorization['status'] == 'both_capacity_and_common_stream_accepted'
    assert authorization['epochs'] == 24 and authorization['bundle_manifest_sha256'] == manifest_hash
    assert authorization['capacity_sha256'] == sha(capdir / 'capacity.json')
    out = root / 'outputs' / (variant + '_24')
    child('train24', [python, '-u', str(code / 'run_candidate.py'), 'train', '--config', str(cfg), *common,
                      '--capacity-receipt', str(capdir / 'capacity.json'), '--stop-after-epochs', '24', '--output', str(out)])
    complete = json.loads((out / 'complete.json').read_text())
    records = []
    for line in (out / 'training_audit.jsonl').open():
        row = json.loads(line)
        assert row['batches'] == len(row['steps']) == 808
        assert row['unique_images'] == row['consumed_originals'] == 6464
        assert row['optimizer_steps'] == sum(s['optimizer_step_calls'] for s in row['steps'])
        assert sha(Path(row['checkpoint'])) == row['checkpoint_sha256']
        records.append(row)
    assert [r['epoch'] for r in records] == list(range(24))
    assert complete['completed_epochs'] == 24 and not complete['full96_complete']
    assert complete['primary_checkpoint'] == records[-1]['checkpoint']
    checkpoint = out / 'checkpoints' / 'epoch_023.pth'
    evaluated = root / 'outputs' / (variant + '_fixed24_separate_eval')
    child('fixed24_eval', [python, '-u', str(code / 'evaluate.py'), '--config', str(cfg), '--checkpoint', str(checkpoint),
                          '--images', runtime['val_images'], '--annotations', runtime['val_ann'], '--output', str(evaluated), '--device', 'cuda:0'])
    result = json.loads((evaluated / 'evaluation.json').read_text())
    assert result['images'] == 548 and result['forward_views'] == 3390
    assert all(abs(result['metrics'][k] - records[-1]['metrics'][k]) < 1e-8 for k in result['metrics'])
    assert all(abs(result['mechanism_metrics'][k] - records[-1]['mechanism_metrics'][k]) < 1e-8 for k in result['mechanism_metrics'])
    save(jobs / 'fixed24_verified.json', {'status': '24_epochs_and_fixed24_separate_eval_verified',
          'variant': variant, 'metrics': result['metrics'], 'mechanism_metrics': result['mechanism_metrics'],
          'checkpoint_sha256': result['checkpoint_sha256'], 'evaluation_sha256': sha(evaluated / 'evaluation.json'),
          'single_seed_exploratory_not_full_innovation_acceptance': True, 'time': time.time()})
    (jobs / 'status.json').write_text(json.dumps({'status': 'bounded24_finished_no_followup_queue', 'time': time.time()}))


if __name__ == '__main__':
    root, stage = Path(sys.argv[1]), sys.argv[2]
    try:
        run(root, stage)
    except BaseException as exc:
        error = {'error': str(exc), 'traceback': traceback.format_exc(), 'time': time.time(), 'no_automatic_retry': True}
        (root / 'jobs' / (stage + '_failed_preserved.json')).write_text(json.dumps(error, indent=2))
        (root / 'jobs' / 'status.json').write_text(json.dumps({'status': 'technical_failure_stopped', 'stage': stage, **error}))
        raise
