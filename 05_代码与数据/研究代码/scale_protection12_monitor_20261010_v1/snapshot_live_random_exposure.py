"""Copy actual complete log-line prefixes from a live owned random control."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import zipfile


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def actual_process(record, pid_key, tick_key):
    stat = Path('/proc/' + str(record[pid_key]) + '/stat')
    assert stat.exists(), 'Original process is not live; inspect terminal state instead'
    fields = stat.read_text().split(') ', 1)[1].split()
    assert fields[19] == str(record[tick_key]) and fields[0] not in ('Z', 'X')
    return dict(pid=record[pid_key], start_ticks=fields[19], state=fields[0], same_owned_process=True)


def snapshot(root, seed, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    screen = root / f'seed{seed}_random_protect'
    run = screen / 'run'
    receipt = read(screen / 'bundle/receipt.json')
    profile = Path(receipt['paths']['profile'])
    assert receipt['identity']['arm'] == 'random_protect' and receipt['identity']['seed'] == seed
    candidate, controller = read(screen / 'candidate_start.json'), read(screen / 'controller_start.json')

    def processes():
        return dict(trainer=actual_process(candidate, 'child_pid', 'child_start_ticks'),
                    controller=actual_process(controller, 'pid', 'start_ticks'))

    before = processes()
    observed_start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    destination.mkdir(parents=True, exist_ok=False)
    events = sorted((run / 'augmentation_events').glob('*.jsonl'))
    assert events, 'No actual active Mosaic events yet'
    files = []
    paths = [(profile, 'training_profile.json', False),
             (run / 'training_audit_epochs.jsonl', 'training_audit_epochs.jsonl', True),
             (run / 'sampled_training_target_sizes.jsonl', 'sampled_training_target_sizes.jsonl', True)]
    paths += [(path, 'augmentation_events/' + path.name, True) for path in events]
    for source, relative, jsonl in paths:
        raw = source.read_bytes()
        captured = raw
        tail = b''
        if jsonl and raw and not raw.endswith(b'\n'):
            split = raw.rfind(b'\n')
            captured, tail = (raw[:split + 1], raw[split + 1:]) if split >= 0 else (b'', raw)
        if jsonl:
            for row in captured.splitlines():
                json.loads(row)
        else:
            json.loads(captured)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(captured)
        files.append(dict(relative_path=relative, source_path=str(source),
            snapshot_sha256=sha_bytes(captured), captured_bytes=len(captured),
            source_read_bytes=len(raw), source_read_sha256=sha_bytes(raw),
            incomplete_trailing_bytes_omitted=len(tail),
            incomplete_tail_sha256=sha_bytes(tail) if tail else None,
            complete_rows=len(captured.splitlines()) if jsonl else None))
    assert sha_bytes((destination / 'training_profile.json').read_bytes()) == receipt['identity']['training_profile_sha256']
    after = processes()
    assert before['trainer']['pid'] == after['trainer']['pid'] and before['controller']['pid'] == after['controller']['pid']
    value = dict(schema='live_random_exposure_log_prefix_snapshot_v1', seed=seed,
        status='ACTUAL_LIVE_OWNED_LOG_PREFIXES_SNAPSHOTTED', identity=receipt['identity'],
        observed_start_remote_UTC=observed_start,
        observed_end_remote_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        processes_before=before, processes_after=after, files=files,
        coherent_atomic_training_state_snapshot_not_claimed=True,
        complete_line_prefixes_only=True, current_epoch_may_be_incomplete=True,
        completed_epoch_event_coverage_not_assumed=True,
        model_forwards=0, evaluator_calls=0, training_epochs_added=0, optimizer_updates_added=0,
        original_training_files_written=False, source_sha256=sha_bytes(Path(__file__).read_bytes()))
    with (destination / 'snapshot_manifest.json').open('x') as stream:
        json.dump(value, stream, indent=2)
    archive = destination.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(destination.rglob('*')):
            if path.is_file():
                bundle.write(path, path.relative_to(destination).as_posix())
    print(json.dumps(dict(status=value['status'], seed=seed, archive=str(archive),
        bytes=archive.stat().st_size, archive_sha256=sha_bytes(archive.read_bytes()),
        manifest_sha256=sha_bytes((destination / 'snapshot_manifest.json').read_bytes()),
        actual_processes=after, files=len(files) + 1,
        incomplete_trailing_bytes_omitted=sum(row['incomplete_trailing_bytes_omitted'] for row in files),
        model_forwards=0, training_epochs_added=0)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--seed', required=True, type=int, choices=(1, 2))
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    snapshot(args.root, args.seed, args.destination)
