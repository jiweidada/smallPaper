"""Test audit rejection rules with synthetic policy records and existing saved files."""

import copy
import gzip
import json
from pathlib import Path
import shutil
import tempfile

from audit_ablation_lineage import (BATCHES, EPOCHS, HERE, read, read_rows, sha, validate_prefix_rates,
                                    validate_restores, verify_endpoint)
from summarize_completed_pair import validate_learning_rates

ROOT = HERE / 'NO_MPS'
BACKUPS = Path(r'D:\MPSMA_DEIM_experiment_backups\20261107\NO_MPS\backups')
PROTOCOL_SHA = '62c8a9012df8695b400c55b21fac6210d320492f911d551a34eb9015bd4571ad'


def policy_fixture(path, config, preflight, stage_zero_best):
    base = [group.get('lr', config['optimizer']['lr']) for group in config['optimizer']['params']]
    base.append(config['optimizer']['lr'])
    rates = list(preflight['initial_executed_lrs'])
    rows, events = [], []
    last_epoch, updates, best_ap = -1, 0, -1.0
    stage_one = None

    def restore(epoch, reason, attempts):
        nonlocal last_epoch, updates, rates
        saved = rows[stage_one]
        events.append({'actual_loop_epoch': epoch, 'reason': reason, 'checkpoint_sha256': 'a' * 64,
                       'previous_last_epoch': last_epoch, 'restored_last_epoch': saved['last_epoch'],
                       'monotonic_step_attempts': attempts, 'ema_updates': saved['ema_updates']})
        last_epoch, updates = saved['last_epoch'], saved['ema_updates']
        rates = list(saved['post_final_update_lrs'])

    with gzip.open(path, 'wt', encoding='utf-8', compresslevel=1) as stream:
        stream.write('attempt,actual_epoch,batch,group0,group1,group2,group3\n')
        for epoch in range(EPOCHS):
            if epoch == 296:
                restore(epoch, 'stage_two_entry', epoch * BATCHES)
            first = list(rates)
            for batch in range(BATCHES):
                attempt = epoch * BATCHES + batch
                stream.write(f'{attempt},{epoch},{batch},' + ','.join(repr(rate) for rate in rates) + '\n')
                rates = [rate * (attempt / config['warmup_iter']) ** 2
                         if attempt < config['warmup_iter'] else rate for rate in base]
            last_epoch += 1
            updates += BATCHES
            if stage_zero_best:
                ap = 0.30 if epoch == 0 else 0.10
                if epoch in (303, 310):
                    ap = 0.35
                if epoch == 319:
                    ap = 0.36
            else:
                ap = 0.01 + epoch / 2000 if epoch < 296 else 0.01 + 295 / 2000
            row = {'epoch': epoch, 'last_epoch': last_epoch, 'ema_updates': updates,
                   'metrics': {'AP': ap}, 'first_executed_lrs': first, 'post_final_update_lrs': list(rates)}
            rows.append(row)
            if ap > best_ap:
                best_ap = ap
                if epoch < 296:
                    stage_one = epoch
            elif epoch >= 296:
                restore(epoch, 'late_epoch_without_new_global_best', (epoch + 1) * BATCHES)
    return rows, events, stage_one


def rejected(action, label):
    try:
        action()
    except (RuntimeError, ValueError) as error:
        return {'case': label, 'rejected': True, 'reason': str(error)}
    raise AssertionError('Invalid fixture accepted: ' + label)


def main():
    assert sha(ROOT / 'execution_protocol.json') == PROTOCOL_SHA
    plan = read(ROOT / 'execution_protocol.json')
    frozen = {item['path']: item['sha256'] for item in plan['frozen_files']}
    config = read(ROOT / 'execution_configs/NO_MPS.yml')
    preflight = read(ROOT / 'inputs/preflight.json')
    scratch = HERE / 'validator_scratch'
    scratch.mkdir(exist_ok=True)
    assert scratch.resolve().is_relative_to(HERE.resolve())
    valid, invalid = [], []
    with tempfile.TemporaryDirectory(prefix='late_policy_', dir=str(scratch)) as temporary:
        folder = Path(temporary).resolve()
        assert folder.is_relative_to(scratch.resolve())
        for name, stage_zero_best, expected_count in [('all_late_rounds_restore', False, 25),
                                                      ('warmup_checkpoint_and_new_terminal_best', True, 23)]:
            path = folder / f'{name}.csv.gz'
            rows, events, stage_one = policy_fixture(path, config, preflight, stage_zero_best)
            selected, resets = validate_restores(rows, events, 'a' * 64)
            assert selected == stage_one and len(events) == expected_count
            if stage_zero_best:
                assert stage_one == 0 and 304 not in resets and 311 in resets and 320 not in resets
            else:
                assert stage_one == 295 and 320 in resets
            current = validate_prefix_rates(path, rows, config, preflight, resets, stage_one)
            native = validate_learning_rates(path, rows, config, preflight, resets, stage_one)
            assert current['recorded_attempts'] == native['recorded_attempts'] == 517440
            assert current['maximum_absolute_difference'] == native['maximum_absolute_difference'] == 0
            valid.append({'case': name, 'synthetic_restore_events': len(events),
                          'synthetic_lr_records': current['recorded_attempts'], 'native_rule_agrees': True})
            bad_rows = copy.deepcopy(rows)
            bad_rows[297]['ema_updates'] += BATCHES
            invalid.append(rejected(lambda: validate_restores(bad_rows, events, 'a' * 64), name + '_wrong_ema_count'))
            invalid.append(rejected(lambda: validate_restores(rows, events[:-1], 'a' * 64), name + '_missing_restore'))
            if stage_zero_best:
                corrupt = folder / 'wrong_stage_entry_lr.csv.gz'
                with gzip.open(path, 'rt') as source, gzip.open(corrupt, 'wt', compresslevel=1) as target:
                    target.write(source.readline())
                    for attempt, line in enumerate(source):
                        if attempt == 296 * BATCHES:
                            fields = line.rstrip('\n').split(',')
                            fields[3] = repr(config['optimizer']['params'][0]['lr'])
                            line = ','.join(fields) + '\n'
                        target.write(line)
                invalid.append(rejected(lambda: validate_prefix_rates(corrupt, rows, config, preflight, resets, stage_one),
                                        'incorrect_warmup_checkpoint_restore_lr'))
        actual = BACKUPS / 'epochs/epoch0099'
        manifest = read(actual / 'manifest.json')
        best = actual / 'selected' / f"epoch{manifest['best']['epoch']:04d}" / 'best.pt'
        row = copy.deepcopy(read_rows(actual / 'epochs.jsonl')[-1])
        row.update(epoch=319, last_epoch=319, optimizer_step_attempts=517440, ema_updates=517440)
        for label, source in [('rolling_checkpoint_as_terminal', actual / 'next_epoch.pt'),
                              ('earlier_best_checkpoint_as_terminal', best)]:
            case = folder / label
            case.mkdir()
            weight = case / 'terminal_epoch0319.pt'
            shutil.copyfile(source, weight)
            for suffix in ('predictions.json', 'evaluation_arrays.npz'):
                shutil.copyfile(actual / f'latest.{suffix}', case / f'terminal.{suffix}')
            (case / 'terminal.metrics.json').write_text(json.dumps({**row, 'checkpoint_sha256': sha(weight)}))
            outcome = rejected(lambda: verify_endpoint(ROOT, case, 'terminal', row, plan, frozen, PROTOCOL_SHA), label)
            assert ('provenance' in outcome['reason'] if label.startswith('rolling') else 'counters' in outcome['reason'])
            invalid.append(outcome)
        assert folder.is_relative_to(scratch.resolve())
    report = {'status': 'LATE_AUDIT_RULES_TESTED_WITH_SYNTHETIC_POLICY_FIXTURES',
              'actual_training_completion_claimed': False, 'valid_synthetic_cases': valid,
              'rejected_cases': invalid, 'model_forwards': 0, 'evaluator_runs': 0,
              'audit_script_sha256': sha(HERE / 'audit_ablation_lineage.py')}
    with (HERE / 'late_audit_validation_tests.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
