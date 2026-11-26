"""Read-only progress observer for the authorized independent fixed12 runs."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, time, traceback


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def live(record, pid='pid', ticks='start_ticks'):
    if not record:
        return False
    try:
        fields = Path('/proc/' + str(record[pid]) + '/stat').read_text().split()
        return fields[21] == str(record[ticks]) and fields[2] != 'Z'
    except (OSError, KeyError):
        return False


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def publish(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def main(prefix, seed):
    root = Path(prefix) / 'DEIM/experiments/user_backbone12_20261011_v1'
    code = root / 'code'
    out = root / 'review_execution_monitor_v1'
    out.mkdir(exist_ok=False)
    protocol = read(code / 'training_protocol.json')
    assert protocol['max_epochs'] == 12
    arms = ('backbone_silu', 'p2_skip', 'backbone_attention')
    plans = {}
    for arm in arms:
        bundle = root / f'seed{seed}_{arm}/bundle'
        receipt = read(bundle / 'receipt.json')
        plan = read(bundle / 'plan.json')
        assert receipt['identity']['plan_sha256'] == sha(bundle / 'plan.json')
        assert plan['arm'] == arm and plan['seed'] == seed and plan['max_epochs'] == 12
        assert protocol['execution_allowlist'][arm] == [1, 2]
        plans[arm] = (bundle, receipt, plan)
    cached = protocol['baseline_reuse'][str(seed)]
    for relative, digest in cached['pinned_files'].items():
        assert sha(Path(cached['source']) / relative) == digest
    baseline = plans[arms[0]][2]['historical_baseline_stats']
    assert len(baseline) == 12 and all(p[2]['historical_baseline_stats'] == baseline for p in plans.values())
    owner = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21],
                 started_remote_UTC=utc(), source_sha256=sha(__file__), seed=seed,
                 scope='No model imports, launches, stops, retries or BASE execution; observation only.')
    (out / 'start.json').write_text(json.dumps(owner))
    previous = None
    started = time.monotonic()
    while True:
        try:
            jobs = {}
            for arm, (bundle, receipt, plan) in plans.items():
                screen = root / f'seed{seed}_{arm}'
                rows = []
                log = screen / 'run/log.txt'
                if log.exists():
                    for line in log.read_text().splitlines():
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if 'test_coco_eval_bbox' in row:
                            rows.append(row)
                epochs = [int(row['epoch']) for row in rows]
                assert epochs == list(range(len(epochs))) and len(epochs) <= 12
                trainer_record = read(screen / 'candidate_start.json') if (screen / 'candidate_start.json').exists() else None
                active = live(trainer_record, 'child_pid', 'child_start_ticks')
                audit_file = screen / 'run/training_audit_live.json'
                audit = read(audit_file) if audit_file.exists() else None
                steps = audit['actual_optimizer_steps'] if audit else 0
                current = rows[-1]['test_coco_eval_bbox'] if rows else None
                preflight = screen / 'detector_preflight_v1/report.json'
                state = 'native_fixed12_complete_saved_EMA_verification_pending' if len(rows) == 12 else (
                    'training' if active and steps > 0 else 'trainer_initializing' if active else 'awaiting_GPU_slot')
                failures = {name: read(screen / name) for name in ('controller_failure.json',)
                            if (screen / name).exists()}
                jobs[arm] = dict(state=state, completed_epochs=len(rows), actual_optimizer_steps=steps,
                    native_primary_AP_points=100 * current[0] if current else None,
                    native_primary_APs_points=100 * current[3] if current else None,
                    compared_to_cached_BASE12=current is not None,
                    comparison_scope='Fixed12 screening endpoint' if len(rows) == 12 else 'Interim learning progress; not an endpoint judgment',
                    delta_AP_points=100 * (current[0] - baseline[0]) if current else None,
                    delta_APs_points=100 * (current[3] - baseline[3]) if current else None,
                    trainer=trainer_record, trainer_same_live=active, full_GPU_preflight_passed=preflight.exists(),
                    failure=failures, independent_saved_EMA_effect_not_claimed=True)
            queue_root = root / f'queue_seed{seed}'
            q = read(queue_root / 'queue_live.json')
            queue_record = read(root / f'queue_dispatch_seed{seed}.json')
            failures = read(queue_root / 'failure.json') if (queue_root / 'failure.json').exists() else None
            complete = read(queue_root / 'completion.json') if (queue_root / 'completion.json').exists() else None
            result = dict(observed_remote_UTC=utc(), seed=seed, jobs=jobs, queue_same_live=live(queue_record),
                queue_phase=q['phase'], queue_failure=failures, queue_completion=complete,
                cached_BASE12_AP_points=100 * baseline[0], cached_BASE12_APs_points=100 * baseline[3],
                per_configuration_epoch_cap=12, BASE_epochs_added=0, BASE_forwards_added=0,
                monitor_model_forwards=0, monitor_training_updates=0, automatic_retry=False)
            publish(out / 'live.json', result)
            signature = json.dumps({arm: {k: v[k] for k in ('state', 'completed_epochs', 'failure')} for arm, v in jobs.items()}, sort_keys=True)
            if signature != previous:
                with (out / 'events.jsonl').open('a') as stream:
                    stream.write(json.dumps(result) + '\n')
                previous = signature
            if complete or failures or not live(queue_record):
                (out / 'completion.json').write_text(json.dumps(dict(status='OBSERVATION_FINISHED', UTC=utc(),
                    experiments_completed=bool(complete), queue_failure=failures, no_training_process_signals=True)))
                return
            if time.monotonic() - started > 172800:
                (out / 'completion.json').write_text(json.dumps(dict(status='OBSERVATION_TIME_LIMIT', UTC=utc(),
                    no_training_process_signals=True)))
                return
        except (OSError, json.JSONDecodeError):
            # Concurrent writers may temporarily expose an incomplete observation.
            publish(out / 'read_advisory.json', dict(UTC=utc(), traceback=traceback.format_exc(),
                observation_only=True, no_process_signals=True))
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--seed', type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    main(args.prefix, args.seed)
