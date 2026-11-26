"""Launch exactly one authorized seed1 fresh132 arm, with no automatic retry."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from screen_runtime import file_sha, source_manifest


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    root = Path(sys.argv[1]); work = root / 'code'; arm = root / 'p2_seed1'
    auth = json.loads((root / 'authorization.json').read_text())
    assert auth['human_authorization'] == '两张显卡，你都用起来跑实验'
    assert auth['seed'] == 1 and auth['epoches'] == 132 and not auth['auto_retry']
    if (arm / 'launch.json').exists() or (arm / 'run').exists():
        raise RuntimeError('Seed1 already launched; automatic retry prohibited')
    assert json.loads((root / 'full132_amp_preflight.json').read_text())['passed']
    assert json.loads((root / 'full132_cpu_checks.json').read_text())['passed']
    alignment = json.loads((root / 'full132_configuration_alignment.json').read_text())
    assert file_sha(root / 'p2_full132.yml') == alignment['full_config_sha256']
    assert hashlib.sha256(json.dumps(source_manifest(work), sort_keys=True).encode()).hexdigest() == alignment['source_digest']
    assert file_sha(auth['common_init']) == auth['common_init_sha256']
    assert file_sha(auth['baseline_args_path']) == auth['baseline_args_sha256']
    for item in auth['annotations'].values():
        assert file_sha(item['path']) == item['sha256']
    snapshot = json.loads((root / 'baseline_A_snapshot.json').read_text())
    baseline_work = Path(snapshot['baseline_cwd'])
    for rel, sha in auth['native_sources'].items():
        assert file_sha(baseline_work / rel) == sha
        assert file_sha(work / rel) == sha
    proc = Path(f"/proc/{auth['baseline_PID']}/stat")
    if proc.exists():
        fields = proc.read_text().split(') ', 1)[1].split()
        assert int(fields[19]) == auth['baseline_start_ticks'] and fields[0] not in ('Z', 'T')
    else:
        status = Path(auth['baseline_args_path']).parents[1] / 'status.txt'
        assert 'exit 0' in status.read_text(), 'Baseline unavailable without successful completion'
    arm.mkdir(exist_ok=True)
    command = [sys.executable, '-u', 'train_screen.py', '-c', str(root / 'p2_full132.yml'),
               '-t', auth['common_init'], '--seed', '1', '--use-amp', '-d', 'cuda:0',
               '--output-dir', str(arm / 'run')]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=auth['GPU_UUID'], OMP_NUM_THREADS='6',
               MKL_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6', PYTHONDONTWRITEBYTECODE='1',
               SCREEN_RECEIPT_DIR=str(arm / 'audit'))
    start = time.monotonic(); log = arm / 'train.log'; hard_timeout = False
    with log.open('wb') as output:
        child = subprocess.Popen(command, cwd=work, env=env, stdout=output,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        ticks = int(Path(f'/proc/{child.pid}/stat').read_text().split(') ', 1)[1].split()[19])
        supervisor_ticks = int(Path(f'/proc/{os.getpid()}/stat').read_text().split(') ', 1)[1].split()[19])
        receipt = {'start_UTC': utc(), 'supervisor_PID': os.getpid(),
                   'supervisor_start_ticks': supervisor_ticks, 'trainer_PID': child.pid,
                   'trainer_start_ticks': ticks, 'command': command, 'cwd': str(work),
                   'authorization_sha256': file_sha(root / 'authorization.json'),
                   'human_authorization': auth['human_authorization'],
                   'common_init_fresh_start': True, 'seed': 1, 'epoches': 132,
                   'config_sha256': alignment['full_config_sha256'],
                   'source_digest': alignment['source_digest'], 'timeout_hours': 72}
        (arm / 'launch.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
        previous = -1; stalled = 0
        while child.poll() is None:
            size = log.stat().st_size; stalled = stalled + 1 if size == previous else 0
            previous = size; elapsed = time.monotonic() - start
            state = {'time_UTC': utc(), 'alive': True, 'elapsed_seconds': elapsed, 'log_bytes': size,
                     'advisory': 'output_stall_suspected' if stalled >= 3 else None}
            with (arm / 'monitor.jsonl').open('a') as f:
                f.write(json.dumps(state) + '\n')
            if elapsed > 72 * 3600:
                (arm / 'hard_timeout_notice.json').write_text(json.dumps(state, indent=2))
                os.killpg(child.pid, signal.SIGTERM); hard_timeout = True
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(60)
        code = child.wait()
    (arm / 'exit.json').write_text(json.dumps({'end_UTC': utc(), 'exit_code': code,
                  'hard_timeout': hard_timeout, 'elapsed_seconds': time.monotonic() - start}, indent=2))
    sys.exit(code if code >= 0 else 1)


if __name__ == '__main__':
    main()
