"""Read-only packaging after the existing one-shot selection diagnostics.

No model/runtime import, process signalling, training command or replay launch.
"""
from pathlib import Path, PurePosixPath
import argparse, datetime, hashlib, json, os, shutil, subprocess, time, traceback, zipfile

HERE = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def protocol():
    p = read(HERE / 'protocol.json')
    assert p['schema'] == 'native_selection_collection12_v1'
    for n, h in p['files'].items(): assert sha(HERE / n) == h, n
    return p


def names_valid(names):
    assert len(names) == len(set(names)) == len({n.casefold() for n in names})
    for n in names:
        path = PurePosixPath(n)
        assert n and not path.is_absolute() and '..' not in path.parts
        assert ':' not in n and '\\' not in n and path.as_posix() == n


def state(r, pid='pid', ticks='start_ticks'):
    try:
        fields = Path('/proc/' + str(r[pid]) + '/stat').read_text().split()
        return fields[2] if fields[21] == str(r[ticks]) and fields[2] != 'Z' else None
    except (FileNotFoundError, ProcessLookupError): return None


def validate_metadata(identity, plan, trained, v, done, p, arm, seed, checkpoint, training_sha):
    assert identity['arm'] == plan['arm'] == v['arm'] == arm and arm in p['arms']
    assert type(seed) is int and identity['seed'] == plan['seed'] == v['seed'] == seed in (1, 2)
    assert plan['max_epochs'] == 12 and plan['fixed_endpoint'] == 'native_epoch12_EMA'
    assert trained['status'] == 'NATIVE_FIT_FINISHED_EXACT12' and trained['identity'] == identity
    assert trained['completed_epochs'] == 12 and trained['entered_epochs'] == list(range(12))
    assert trained['automatic_retry'] is False
    assert checkpoint == trained['last_checkpoint_sha256'] == v['checkpoint_sha256']
    assert v['original_training_completion_sha256'] == training_sha and v['identity'] == identity
    assert v['status'] == 'PASS_SELECTION_DIAGNOSTIC_EXACT_SAVED_EMA_REPLAY'
    assert v['diagnostic_protocol_sha256'] == p['diagnostic_protocol_sha256']
    assert v['verification_script_sha256'] == p['diagnostic_source_sha256']['replay.py']
    assert v['original_stats'] == v['reloaded_stats'] and len(v['original_stats']) == 12
    assert v['stats_max_abs_fraction_difference'] == v['stats_tolerance'] == 0
    assert v['native_GT_hits_exact'] is True and v['all548_stage_arrays_exact'] is True
    assert v['stage_max_abs_difference'] and all(x == 0 for x in v['stage_max_abs_difference'].values())
    assert v['candidate_model_forwards'] == v['actual_selector_calls'] == v['existing_selected_bbox_head_calls'] == 137
    assert v['images'] == v['selection_images'] == 548
    assert v['additional_bbox_head_or_topk_calls'] == v['baseline_model_forwards'] == v['optimizer_updates'] == v['epochs_added'] == 0
    assert v['automatic_retry'] is False and v['all_parameters_frozen_like_native_EMA'] is True
    assert done['exit_code'] == 0 and done['timed_out'] is False and done['automatic_retry'] is False
    assert done['baseline_model_forwards'] == done['epochs_added'] == 0


def screens(base, seed):
    return {arm: base / family / f'seed{seed}_{arm}' for arm, family in (
        ('original_stem', 'original_stem12_20261011_v1'),
        ('native_stem_scratch', 'native_stem_scratch12_20261011_v3'),
        ('highfreq_rgb', 'stem_component12_20261011_v1'))}


def collect(screen, arm, seed, base, p, output):
    diagcode = base / 'native_selection_diagnostic12_20261011_v2/code'
    assert {x.name: sha(x) for x in diagcode.iterdir() if x.is_file()} == p['diagnostic_source_sha256']
    diagnostic_p = read(diagcode / 'diagnostic_protocol.json')
    native_code = screen.parent / 'code'; allowed = diagnostic_p['allowed_native_roots'][screen.parent.name]
    assert arm in allowed['arms'] and {x.name: sha(x) for x in native_code.iterdir() if x.is_file()} == allowed['files']
    receipt, plan = read(screen / 'bundle/receipt.json'), read(screen / 'bundle/plan.json')
    identity = receipt['identity']; assert identity['implementation'] == allowed['files']
    assert sha(screen / 'bundle/plan.json') == identity['plan_sha256']
    assert sha(screen / 'bundle/resolved.yml') == receipt['resolved_sha256']
    run, diag = screen / 'run', screen / 'native_selection_diagnostic12_v2'
    trained, v, done = read(run / 'training_completion.json'), read(diag / 'verification.json'), read(diag / 'completion.json')
    validate_metadata(identity, plan, trained, v, done, p, arm, seed, sha(run / 'last.pth'), sha(run / 'training_completion.json'))
    assert read(screen / 'controller_completion.json')['status'] == 'COMPLETE_FIXED12'
    assert read(screen / 'candidate_completion.json')['exit_code'] == 0
    assert state(read(screen / 'controller_start.json')) is None
    assert state(read(screen / 'candidate_start.json'), 'child_pid', 'child_start_ticks') is None
    replay_start = read(diag / 'start.json')
    assert state(replay_start) is None and state(replay_start, 'controller_pid', 'controller_start_ticks') is None
    closeout = read(screen / 'post_training/completion.json')
    assert closeout['status'] in ('STEM_FIXED12_RELOAD_AND_COLLECTION_FINISHED', 'NATIVE_STEM_SCRATCH_FIXED12_RELOAD_AND_COLLECTION_FINISHED')
    assert read(screen / 'candidate_reload_v1/verification.json')['status'] == 'PASS_SAVED_CANDIDATE_RELOAD'
    pairs = {}
    for directory in ('selection_epoch12', 'stages_epoch12', 'endpoint_epoch12'):
        folder = diag / directory; m = read(folder / 'manifest.json')
        expected = {'summary.json'} | {f'image_{i:06d}.npz' for i in range(548)} if directory != 'endpoint_epoch12' else {'summary.json', 'precision_recall.npz', 'GT_hits.npz', 'matching_all_area.json.gz'}
        assert set(m['files']) == expected
        summary = read(folder / 'summary.json'); assert summary['identity'] == identity
        assert summary['epoch_zero_based'] == 11 and summary['evaluated_weights'] == 'EMA'
        for n, h in m['files'].items():
            names_valid([n]); assert sha(folder / n) == h, n
            pairs['diagnostic/' + directory + '/' + n] = folder / n
        pairs['diagnostic/' + directory + '/manifest.json'] = folder / 'manifest.json'
    assert sha(diag / 'selection_epoch12/manifest.json') == v['selection_manifest_sha256']
    assert read(diag / 'selection_epoch12/manifest.json')['trace_code_sha256'] == p['diagnostic_source_sha256']['selection_stage_trace.py']
    assert read(run / 'endpoint_epoch12/summary.json')['native_stats'] == v['original_stats']
    for name in ('verification.json', 'completion.json', 'start.json', 'live.json', 'reload_config.yml', 'worker.log'):
        assert (diag / name).is_file(); pairs['diagnostic/' + name] = diag / name
    for name in p['diagnostic_source_sha256']: pairs['diagnostic_source/' + name] = diagcode / name
    for name, path in dict(receipt=screen / 'bundle/receipt.json', plan=screen / 'bundle/plan.json',
            native_training_completion=run / 'training_completion.json', native_closeout_completion=screen / 'post_training/completion.json',
            primary_stage_manifest=run / 'stages_epoch12/manifest.json', primary_endpoint_manifest=run / 'endpoint_epoch12/manifest.json',
            native_terminal_manifest=screen / 'terminal_manifest.json').items():
        pairs['binding/' + name + '.json'] = path
    names_valid(list(pairs))
    assert shutil.disk_usage(output).free > sum(x.stat().st_size for x in pairs.values()) + 64 * 1024**2
    files = {n: dict(bytes=f.stat().st_size, sha256=sha(f)) for n, f in sorted(pairs.items())}
    manifest = dict(schema='native_selection_diagnostic_packet_v1', identity=identity, files=files,
        checkpoint_sha256=v['checkpoint_sha256'], native_terminal_manifest_sha256=sha(screen / 'terminal_manifest.json'),
        diagnostic_verification_sha256=sha(diag / 'verification.json'), collector_source_sha256=sha(__file__),
        collection_protocol_sha256=sha(HERE / 'protocol.json'), all_observed_replay_handles_stopped=True,
        model_forwards_added=0, evaluator_calls_added=0, optimizer_updates_added=0, training_epochs_added=0,
        existing_process_signals=0, scope='Diagnostic tensors and provenance only; native checkpoint/formal terminal packet is collected separately.')
    name = f'seed{seed}_{arm}_selection_v1'
    manifest_path = output / (name + '_manifest.json'); write(manifest_path, manifest)
    archive = output / (name + '.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED) as z:
        for n, f in sorted(pairs.items()): z.write(f, n)
        z.write(manifest_path, 'packet_manifest.json')
    for n, f in pairs.items(): assert sha(f) == files[n]['sha256'], 'Source changed during collection: ' + n
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None and len(z.namelist()) == len(files) + 1
    metadata = dict(status='PASS_SELECTION_DIAGNOSTIC_READ_ONLY_COLLECTION', arm=arm, seed=seed,
        archive=str(archive), bytes=archive.stat().st_size, sha256=sha(archive), manifest_sha256=sha(manifest_path),
        entries=len(files) + 1, model_forwards_added=0, training_epochs_added=0, existing_process_signals=0,
        completed_remote_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(output / (name + '_metadata.json'), metadata)
    return metadata


def main(prefix, seed):
    assert prefix == {1: '/25085400120', 2: '/24085403014'}[seed]
    p = protocol(); base = Path(prefix) / 'DEIM/experiments'
    root = HERE.parent / f'seed{seed}_collection'; root.mkdir(exist_ok=False)
    qroot = base / 'native_selection_diagnostic12_20261011_v2'
    owner = read(qroot / f'queue_dispatch_seed{seed}.json')
    assert state(owner) is not None, 'Observe the existing diagnostic queue live; never restart it'
    write(root / 'start.json', dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21],
        source_sha256=sha(__file__), protocol_sha256=sha(HERE / 'protocol.json'), observed_queue=owner,
        model_forwards_added=0, training_epochs_added=0, existing_process_signals=0))
    begun = time.monotonic(); collected = {}
    try:
        while len(collected) != 3:
            for arm, screen in screens(base, seed).items():
                if arm in collected: continue
                diag = screen / 'native_selection_diagnostic12_v2'
                if (diag / 'completion.json').exists():
                    r = read(diag / 'start.json')
                    if state(r) is None and state(r, 'controller_pid', 'controller_start_ticks') is None:
                        collected[arm] = collect(screen, arm, seed, base, p, root)
            if len(collected) == 3: break
            queue = qroot / f'seed{seed}_queue'
            assert not (queue / 'failure.json').exists(), 'Diagnostic failed; preserve partial valid packets, no replay/retry'
            assert state(owner) is not None, 'Existing queue exited without all positive records; no relaunch'
            temporary = root / 'live.tmp'
            temporary.write_text(json.dumps(dict(phase='waiting_existing_diagnostics', UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                pending=[a for a in p['arms'] if a not in collected], collected=collected, queue_same_live=True)))
            temporary.replace(root / 'live.json')
            if time.monotonic() - begun > p['wait_timeout_seconds']: raise TimeoutError('Read-only collection wait expired; existing jobs untouched')
            time.sleep(30)
        write(root / 'completion.json', dict(status='ALL_THREE_DIAGNOSTICS_PACKAGED_NO_NEW_MODEL_CALLS', seed=seed,
            collected=collected, model_forwards_added=0, training_epochs_added=0, existing_process_signals=0))
    except BaseException:
        write(root / 'failure.json', dict(status='READ_ONLY_COLLECTION_FAILED_NO_RETRY', traceback=traceback.format_exc(),
            collected=collected, model_forwards_added=0, training_epochs_added=0, existing_process_signals=0))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prefix', required=True); parser.add_argument('--seed', type=int, choices=(1, 2), required=True)
    args = parser.parse_args(); main(args.prefix, args.seed)
