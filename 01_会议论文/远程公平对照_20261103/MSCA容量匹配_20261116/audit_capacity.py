"""Audit the archived capacity comparison without rerunning the evaluator."""

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

from pair import BACKUP, HERE, SERVERS
from collect_metrics import atomic_write
from audit_accepted_fresh_evidence import (
    artifact, audit_endpoint, read, read_rows, require, sha, validate_prefix_rates,
    validate_restores, verify_exit, verify_full_archive,
)


def recovery_lineage(server, archive, count, original_launch):
    from audit_recovery_lineage import verify
    return verify(server, archive, count, original_launch)


def audit_arm(server, sources, launch):
    control = BACKUP / server.arm
    archive = control / 'backups'
    count = len(sources)
    if not count:
        return {'arm': server.arm, 'accepted_rounds': 0}
    require(list(sources) == list(range(1, count + 1)), 'Missing accepted metric row')
    for name in ('backup_monitor.py', 'backup_protocol.py', 'checkpoint_validator.py'):
        require(sha(control/name) == sha(HERE/'driver'/name), f'Archived verification code differs: {name}')
    spec = importlib.util.spec_from_file_location('capacity_audit_' + server.arm, control/'backup_protocol.py')
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    plan = read(control/'execution_protocol.json')
    require(protocol.ARM == server.arm == plan['arm'], 'Wrong archived arm')
    require(protocol.PROTOCOL == launch['protocol_sha256'], 'Wrong run protocol')
    final_stage = archive/'epochs'/f'epoch{count - 1:04d}'
    rows = read_rows(final_stage/'epochs.jsonl')
    require(len(rows) == count, 'Saved cumulative metric history has a different length')
    for i, row in enumerate(rows):
        source = sources[i + 1]
        p = archive/'epochs'/f'epoch{i:04d}'/'epoch.metrics.json'
        require(Path(source['metrics_file']) == p and sha(p) == source['metrics_sha256'] and read(p) == row
                and row['epoch'] == i and row['optimizer_step_attempts'] == (i + 1) * protocol.BATCHES,
                f'Cumulative row does not match accepted source at round {i + 1}')
    manifest_data = (final_stage/'manifest.json').read_bytes()
    digest = hashlib.sha256(manifest_data).hexdigest()
    require(digest == sources[count]['manifest_sha256'], 'Last manifest hash differs')
    manifest = protocol.parse_manifest(manifest_data, server.arm, count - 1)
    proof = read(final_stage/'payload_validation.json')
    require(sha(final_stage/'payload_validation.json') == sources[count]['validation_sha256'],
            'Last checkpoint validation differs')
    best = max(range(count), key=lambda i: rows[i]['metrics']['AP'])
    stage_one = max(range(min(count, 296)), key=lambda i: rows[i]['metrics']['AP'])
    require(manifest['best']['epoch'] == best and manifest['stage_one']['epoch'] == stage_one,
            'Selected AP checkpoint differs from the accepted metric history')
    if count > 296:
        events = read_rows(final_stage/'restore_events.jsonl')
        returned, resets = validate_restores(rows, events, manifest['stage_one']['checkpoint_sha256'])
        require(returned == stage_one, 'Recorded stage-one checkpoint differs')
    else:
        require(all(r['last_epoch'] == r['epoch'] and r['ema_updates'] == r['optimizer_step_attempts'] for r in rows),
                'Pre-stage-two counters differ')
        resets = set()
    frozen = {entry['path']:entry['sha256'] for entry in plan['frozen_files']}
    evidence = HERE.parent/'remote_evidence'
    config = 'execution_configs/FULL.yml'
    preflight = 'preflight/FULL.json'
    first100 = evidence/'preflight/FULL.first100.json'
    require(all(sha(evidence/key) == frozen['frozen_parent/' + key] for key in (config, preflight)),
            'Frozen FULL recipe differs')
    require(sha(first100) == frozen['inputs/first100.json']
            and read_rows(final_stage/'actual_first100.jsonl') == read(first100),
            'Executed first 100 input batches differ from the frozen recipe')
    rates = validate_prefix_rates(final_stage/'executed_learning_rates.csv.gz', rows,
                                 read(evidence/config), read(evidence/preflight), resets, stage_one)
    result = {'arm': server.arm, 'accepted_rounds': count, 'learning_rates': rates,
              'latest': audit_endpoint(protocol, final_stage, manifest, proof, rows[-1], '', 'latest'),
              'best': audit_endpoint(protocol, final_stage, manifest, proof, rows[best],
                                     protocol.selected_folder(manifest['best']), 'best')}
    lineage = recovery_lineage(server, archive, count, launch)
    if lineage is not None:
        result['recovery'] = lineage
    monitor = read(archive/'monitor_status.json')
    if count == 320 and monitor['status'] == 'COMPLETED_AND_BACKED_UP':
        first_receipt = read(archive/'epochs/epoch0000/receipt.json')
        final_receipt = read(archive/'completed_run_receipt.json')
        require(first_receipt['first_epoch_in_this_segment'] == 0, 'Initial run segment differs')
        expected_supervisor = launch
        if lineage is None:
            require(final_receipt['first_epoch_in_this_segment'] == 0, 'Unrecorded continuation at endpoint')
        else:
            directory = Path(lineage['segments'][-1]['directory'])
            first_live = read(directory/'first_resumed_live.json')
            expected_supervisor = first_live['supervisor']
            recovery_launch = read(directory/'launch.json')
            require(expected_supervisor['supervisor_pid'] == recovery_launch['supervisor_pid']
                    and expected_supervisor['supervisor_start_ticks'] == recovery_launch['start_ticks']
                    and expected_supervisor['recovery_plan_sha256'] == recovery_launch['recovery_plan_sha256']
                    and final_receipt['first_epoch_in_this_segment'] == lineage['boundary_round']
                    and final_receipt['resumed_from_checkpoint_sha256']
                        == read(directory/'recovery_plan.json')['checkpoint_sha256']
                    and final_receipt['recovery_plan_sha256'] == recovery_launch['recovery_plan_sha256'],
                    'Final resumed process identity differs')
        binding = {'records':{'supervisor.json':expected_supervisor, 'run/receipt.json':first_receipt}}
        result['terminal'] = audit_endpoint(protocol, final_stage, manifest, proof, rows[-1], '', 'terminal')
        result['exit'] = verify_exit(archive, plan, rows, manifest, digest, binding)
        result['backup'] = verify_full_archive(control)
    return result


def main():
    export_path = HERE/'逐轮导出核对.json'
    csv_path = HERE/'逐轮指标_MSCA容量匹配.csv'
    report = read(export_path)
    require(report['csv_sha256'] == sha(csv_path), 'Verified metric export has changed')
    setup = read(HERE/'监控状态.json')
    launch = {r['arm']:r['supervisor'] for r in setup['records']}
    results = []
    for s in SERVERS:
        sources = {r['round']:r for r in report['sources'] if r['arm'] == s.arm}
        require(len(sources) == report['accepted_rounds'][s.arm], 'Accepted row count differs')
        results.append(audit_arm(s,sources,launch[s.arm]))
    result = {'status':'ACCEPTED_PREFIX_VERIFIED', 'observed_utc':datetime.now(timezone.utc).isoformat(),
              'metric_export':artifact(export_path), 'csv':artifact(csv_path), 'arms':results}
    output = HERE/'当前指标核验.json'
    if all('exit' in arm and 'backup' in arm and 'terminal' in arm for arm in results):
        full, matched = results
        result['status'] = 'COMPLETED_PAIR_VERIFIED'
        result['primary_FULL_minus_MATCHED_percentage_points'] = {
            k:100*(full['terminal']['metrics'][k]-matched['terminal']['metrics'][k]) for k in ('AP','APs')}
        output = HERE/'第320轮主结果.json'
    atomic_write(output,(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps({'file':str(output),'status':result['status'],
                      'accepted_rounds':{a['arm']:a['accepted_rounds'] for a in results}},ensure_ascii=False))


if __name__ == '__main__':
    main()
