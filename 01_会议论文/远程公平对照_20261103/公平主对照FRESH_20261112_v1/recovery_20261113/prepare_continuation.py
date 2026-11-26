"""Verify both accepted prefixes and stage isolated continuation directories."""
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
GOAL = PROJECT / 'monitor_goal_20261112'
BACKUPS = Path('D:/MPSMA_DEIM_experiment_backups/BASE_FULL_FRESH_20261112_v1')
RUNTIMES = {30344: '/25085400120/envs/deim/bin/python',
            30048: '/24085403014/deim_migration_runtime/envs/deim/bin/python'}
CODE = ('continuation_guard.py', 'continuation_launch.py', 'continuation_train.py')
REPLAY = {'BASE_FRESH': (165, 1601, 1617), 'FULL_FRESH': (140, 0, 0)}

sys.path.insert(0, str(PROJECT))
from ssh_control import connect, run_script


def sha(path):
    with path.open('rb') as stream:
        h = hashlib.sha256()
        while block := stream.read(4 * 1024**2):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def check_lr(path, count):
    with gzip.open(path, 'rt', newline='') as stream:
        reader = csv.reader(stream)
        assert next(reader) == ['attempt', 'actual_epoch', 'batch', 'group0', 'group1', 'group2', 'group3']
        rows = 0
        for i, row in enumerate(reader):
            epoch, batch = divmod(i, 1617)
            assert len(row) == 7 and row[:3] == [str(i), str(epoch), str(batch)]
            assert all(math.isfinite(float(lr)) for lr in row[3:])
            rows += 1
    assert rows == count


def prepare(remote, audit, runtime):
    port = remote['port']
    arm = remote['arm']
    expected, lost_lower, lost_upper = REPLAY[arm]
    assert runtime['port'] == port and runtime['status'] == 'VERIFIED'
    assert audit['arm'] == arm and audit['epoch_metrics_rows'] == audit['last_accepted_round'] == expected
    assert audit['data_file_count'] == 7021 and not audit['data_file_errors']
    assert audit['frozen_file_count'] == 494 and not audit['frozen_file_errors']
    assert not audit['next_gate_exists'] and not audit['gpu_compute_apps']
    assert not any(audit['original_processes_present'].values())
    assert audit['lr_history']['complete_gzip_rows_with_header'] == expected * 1617 + 1
    assert remote['receipt']['optimizer_step_attempts'] - expected * 1617 == lost_lower
    original = PurePosixPath(remote['root'])
    destination = original.parent / 'recovery_20261113' / arm
    archive = BACKUPS / arm / 'backups'
    assert not (archive / 'epochs' / f'epoch{expected:04d}').exists()
    stage = archive / 'epochs' / f'epoch{expected-1:04d}'
    manifest_path = stage / 'manifest.json'
    manifest_sha = sha(manifest_path)
    manifest = read(manifest_path)
    assert manifest == remote['last_gate_manifest'] and manifest_sha == audit['gate_manifest_sha256']
    assert manifest['protocol_sha256'] == audit['protocol_sha256'] == sha(BACKUPS / arm / 'execution_protocol.json')
    assert manifest['next_epoch'] == expected and manifest['optimizer_step_attempts'] == expected * 1617
    payload = read(stage / 'payload_validation.json')
    assert payload['status'] == 'CHECKPOINT_PAYLOADS_VERIFIED_CPU_ONLY' and payload['manifest_sha256'] == manifest_sha
    for item in manifest['files']:
        path = stage / item['name']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], path
        if item['source'] in audit['anchor_files']:
            assert audit['anchor_files'][item['source']]['actual'] == item['sha256']
    check_lr(stage / 'executed_learning_rates.csv.gz', expected * 1617)
    with (stage / 'epochs.jsonl').open(encoding='utf-8') as stream:
        metrics = [json.loads(line) for line in stream]
    assert [row['epoch'] for row in metrics] == list(range(expected))
    assert all(row['optimizer_step_attempts'] == (row['epoch'] + 1) * 1617 for row in metrics)
    assert max(metrics, key=lambda row: row['metrics']['AP'])['epoch'] == manifest['best']['epoch']
    local = HERE / arm
    local.mkdir(exist_ok=True)
    with connect(port) as client:
        script = """from pathlib import Path
import hashlib, json
root = Path(%r)
names = ('supervisor.json', 'run/receipt.json', 'training.log', 'supervisor.log', 'launch.lock')
print(json.dumps({name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                  for name in names if (root/name).is_file()}))
""" % str(original)
        answer = run_script(client, script, python=RUNTIMES[port], timeout=60)
        assert answer['exit_code'] == 0, answer['stderr']
        failure_hashes = json.loads(answer['stdout'])
        assert {'supervisor.json', 'run/receipt.json', 'training.log', 'launch.lock'} <= set(failure_hashes)
        deadline = datetime.fromisoformat(remote['supervisor']['started_utc']) + timedelta(days=7)
        plan = {'status': 'PREPARED_FOR_EXPLICIT_CONTINUATION', 'arm': arm, 'port': port,
                'original_root': str(original), 'continuation_root': str(destination),
                'original_protocol_sha256': manifest['protocol_sha256'], 'boot_id': remote['boot_id'],
                'gpu_uuid': runtime['current_snapshot']['cuda_available'] and audit['gpu'].split(', ')[0],
                'runtime': RUNTIMES[port], 'deadline_remote_utc': deadline.isoformat(),
                'accepted_rounds': expected, 'first_round_1based': expected + 1, 'target_rounds': 320,
                'retained_attempts': expected * 1617, 'remaining_attempts': (320 - expected) * 1617,
                'remaining_full_validations': 320 - expected,
                'lost_attempts_lower_bound': lost_lower, 'replayed_attempts_upper_bound': lost_upper,
                'physical_attempts_upper_bound': 517440 + lost_upper,
                'completed_full_validations_before_recovery': expected,
                'checkpoint_sha256': manifest['rolling_sha256'], 'checkpoint_manifest_sha256': manifest_sha,
                'best': manifest['best'], 'stage_one': manifest['stage_one'],
                'original_failure_records': failure_hashes,
                'continuation_code': {name: sha(HERE / name) for name in CODE},
                'local_backup_root': str(archive), 'accepted_checkpoint': str(stage / 'next_epoch.pt')}
        plan_path = local / 'continuation_plan.json'
        if plan_path.exists():
            assert read(plan_path) == plan, 'Existing continuation plan differs'
        else:
            save(plan_path, plan)
        pending = destination.parent / (arm + '.staging')
        with client.open_sftp() as sftp:
            sftp.get_channel().settimeout(30)
            for folder in (destination.parent, pending):
                try:
                    sftp.stat(str(folder))
                except FileNotFoundError:
                    sftp.mkdir(str(folder))
            for source in [plan_path, *(HERE / name for name in CODE)]:
                path = str(pending / source.name)
                data = source.read_bytes()
                try:
                    with sftp.open(path, 'wx') as stream:
                        stream.write(data)
                except OSError:
                    with sftp.open(path, 'rb') as stream:
                        assert stream.read() == data, path
        script = 'plan = ' + repr(plan) + '\n' + (HERE / 'stage_continuation_remote.py').read_text(encoding='utf-8')
        result = run_script(client, script, python=RUNTIMES[port], timeout=300)
        save(local / 'remote_stage_transport.json', result)
        assert result['exit_code'] == 0, result['stderr']
        receipt = json.loads(result['stdout'])
        assert receipt['plan_sha256'] == sha(plan_path)
        assert receipt['accepted_backup_gate_prefix'] == expected and receipt['optimizer_updates'] == 0
        return {'port': port, 'arm': arm, 'plan': str(plan_path), 'remote_stage': receipt}


def main():
    original = read(GOAL / 'fresh_pair_recovery_readonly_070143.json')['observations']
    checks = read(GOAL / 'recovery_preflight_detailed_20261113.json')['observations']
    runtimes = read(GOAL / 'recovery_torch_environment_20261113.json')['observations']
    work = [(arm, next(x['observation'] for x in checks if x['port'] == arm['port']),
             next(x for x in runtimes if x['port'] == arm['port'])) for arm in original]
    with ThreadPoolExecutor(2) as pool:
        pending = [pool.submit(prepare, *entry) for entry in work]
        results = []
        for entry, task in zip(original, pending):
            try:
                results.append(task.result())
            except Exception as error:
                results.append({'port': entry['port'], 'arm': entry['arm'],
                                'error': f'{type(error).__name__}: {error}'})
    report = {'recorded_local_utc': datetime.now(timezone.utc).isoformat(), 'arms': results,
              'model_forwards': 0, 'optimizer_updates': 0, 'evaluator_runs': 0}
    save(HERE / 'continuation_staging.json', report)
    print(json.dumps(report, ensure_ascii=True), flush=True)
    return int(any('error' in result for result in results))


if __name__ == '__main__':
    raise SystemExit(main())
