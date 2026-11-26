"""Verify every accepted run segment, including preserved hardware migrations."""

import ast
from pathlib import Path

from pair import HERE, recovery_segments
from audit_accepted_fresh_evidence import artifact, read, require, sha


def epoch_loop(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_run')
    block = next(n for n in function.body if isinstance(n, ast.Try))
    return ast.dump(next(n for n in block.body if isinstance(n, ast.For)
                         and isinstance(n.target, ast.Name) and n.target.id == 'epoch'), include_attributes=False)


def verify(server, archive, count, original_launch):
    directories = recovery_segments(server.arm)
    if not directories:
        return None
    expected_supervisor = original_launch
    previous_plan_sha = None
    expected_gpu = server.gpu
    segments = []
    initial_runner = initial_pid = None
    baseline_loop = epoch_loop(HERE / 'driver/train_runner.py')
    for directory in directories:
        plan_path = directory / 'recovery_plan.json'
        plan = read(plan_path)
        launch = read(directory / 'launch.json')
        prior = read(directory / 'original_supervisor.json')
        failed = read(directory / 'original_failed_receipt.json')
        boundary = plan['accepted_rounds']
        require(boundary <= count and (not segments or boundary > segments[-1]['boundary_round']),
                'Recovery boundaries are not increasing accepted epochs')
        require(sha(plan_path) == launch['recovery_plan_sha256']
                and sha(directory / 'original_supervisor.json') == plan['original_supervisor_sha256']
                and sha(directory / 'original_failed_receipt.json') == plan['original_failed_receipt_sha256'],
                'Recovery evidence binding differs')
        expected_ticks = expected_supervisor.get('supervisor_start_ticks', expected_supervisor.get('start_ticks'))
        require(prior['supervisor_pid'] == expected_supervisor['supervisor_pid']
                and prior['supervisor_start_ticks'] == expected_ticks
                and prior['status'] == 'STOPPED' and prior['exit_code'] == 1
                and prior['final_receipt'] == failed,
                'Recovery does not continue the preceding failed supervisor')
        require(plan['arm'] == failed['arm'] == server.arm
                and prior['protocol_sha256'] == failed['protocol_sha256'] == plan['protocol_sha256']
                and plan['total_optimizer_update_limit'] == 517440 and plan['total_evaluation_limit'] == 320,
                'Recovery changed run identity or experimental budget')
        if previous_plan_sha is not None:
            require(plan['previous_recovery_plan_sha256'] == previous_plan_sha == prior['recovery_plan_sha256'],
                    'Recovery chain does not bind its preceding plan')
            require(plan['deadline_utc'] == prior['original_deadline_utc'] == segments[-1]['deadline_utc'],
                    'Recovery extended the original deadline')
        if initial_runner is None:
            initial_runner, initial_pid = prior['runner_sha256'], prior['pid']
            require(initial_runner == sha(HERE / 'driver/train_runner.py'), 'Original runner differs')
        else:
            require(prior['runner_sha256'] == segments[-1]['runner_sha256'], 'Preceding runner differs')
        manifest = read(archive / 'epochs' / f'epoch{boundary - 1:04d}' / 'manifest.json')
        require(failed['phase'] == 'awaiting_local_backup' and failed['next_epoch'] == boundary
                and failed['optimizer_step_attempts'] == boundary * 1617
                and failed['rolling_checkpoint_sha256'] == plan['checkpoint_sha256'] == manifest['rolling_sha256']
                and plan['best'] == manifest['best'] and plan['stage_one'] == manifest['stage_one']
                and plan['repeated_updates'] == plan['repeated_evaluations'] == 0,
                'Recovery boundary differs from complete saved state')
        name = directory.parent.name
        require(name in ('recovery_20261118', 'recovery_20261122'), 'Unknown recovery source layout')
        suffix = '_20261122' if name == 'recovery_20261122' else ''
        runner = directory.parent / f'resume_train{suffix}.py'
        supervisor = directory.parent / f'resume_launch{suffix}.py'
        require(sha(runner) == plan['runner_sha256'] and sha(supervisor) == plan['supervisor_sha256']
                and epoch_loop(runner) == baseline_loop, 'Recovery source or epoch loop changed')
        segment = {'directory': str(directory), 'plan': artifact(plan_path), 'boundary_round': boundary,
                   'runner_sha256': plan['runner_sha256'], 'deadline_utc': plan['deadline_utc'],
                   'checkpoint_sha256': plan['checkpoint_sha256'], 'repeated_updates': 0, 'repeated_evaluations': 0,
                   'training_evaluation_loop_AST_identical': True}
        if 'hardware_migration' in plan:
            migration = plan['hardware_migration']
            require(migration['previous_gpu_uuid'] == expected_gpu
                    and migration['gpu_name'] == 'NVIDIA GeForce RTX 3090', 'Unexpected hardware migration')
            expected_gpu = migration['current_gpu_uuid']
            current = read(directory / 'environment.json')
            original = read(HERE / f'environment_{server.port}.json')
            frozen = {entry['path']: entry['sha256'] for entry in read(archive.parent / 'execution_protocol.json')['frozen_files']}
            require(sha(HERE / f'environment_{server.port}.json') == frozen['environment.current.json']
                    and sha(directory / 'environment.json') == plan['environment_snapshot_sha256']
                    and all(current[key] == original[key] for key in ('torch', 'torchvision', 'cuda', 'cudnn', 'packages')),
                    'Migration scientific runtime differs from frozen environment')
            dataset = read(directory / 'dataset_bundle_manifest.json')
            data = read(directory / 'data_preparation.json')
            require(sha(directory / 'dataset_bundle_manifest.json') == frozen['dataset_bundle_manifest.json']
                    and sha(directory / 'data_preparation.json') == plan['data_preparation_sha256']
                    and len(dataset['files']) == data['verified_files'] == 7021
                    and data['archive_sha256'] == dataset['zip_sha256'], 'Migration dataset evidence differs')
            segment['hardware_migration'] = migration
        first_path = directory / 'first_resumed_live.json'
        if count > boundary:
            require(first_path.is_file(), 'No observed live worker evidence for resumed updates')
        if first_path.is_file():
            first = read(first_path)
            observed = first['observed_live']
            sup = first['supervisor']
            require(sup['supervisor_pid'] == launch['supervisor_pid']
                    and sup['supervisor_start_ticks'] == launch['start_ticks']
                    and sup['recovery_plan_sha256'] == launch['recovery_plan_sha256']
                    and first['receipt']['pid'] == sup['pid'] == observed['worker_pid']
                    and observed['worker_live'] and observed['receipt_worker_bound']
                    and first['receipt']['optimizer_step_attempts'] > boundary * 1617,
                    'Live resumed worker does not bind its launch and actual updates')
            segment['worker_pid'] = sup['pid']
        segments.append(segment)
        expected_supervisor = launch
        previous_plan_sha = sha(plan_path)
    for i in range(count):
        receipt = read(archive / 'epochs' / f'epoch{i:04d}' / 'receipt.json')
        preceding = [segment for segment in segments if i >= segment['boundary_round']]
        segment = preceding[-1] if preceding else None
        require(receipt['runner_sha256'] == (segment['runner_sha256'] if segment else initial_runner)
                and receipt['first_epoch_in_this_segment'] == (segment['boundary_round'] if segment else 0)
                and receipt['pid'] == (segment['worker_pid'] if segment else initial_pid),
                f'Accepted round {i + 1} has unexpected process lineage')
        if segment:
            require(receipt['recovery_plan_sha256'] == segment['plan']['sha256']
                    and receipt['resumed_from_checkpoint_sha256'] == segment['checkpoint_sha256'],
                    'Accepted resumed receipt binds another checkpoint or plan')
            if 'hardware_migration' in segment:
                require(receipt['hardware_migration'] == segment['hardware_migration'], 'Receipt hardware migration differs')
    latest = segments[-1]
    return {'plan': latest['plan'], 'boundary_round': latest['boundary_round'], 'segments': segments,
            'repeated_updates': 0, 'repeated_evaluations': 0, 'training_evaluation_loop_AST_identical': True}
