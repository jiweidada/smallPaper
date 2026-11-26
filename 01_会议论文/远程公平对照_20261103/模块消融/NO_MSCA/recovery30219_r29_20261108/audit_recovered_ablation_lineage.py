"""Audit only the explicitly bound NO_MSCA round-29 recovery, preserving the old audit."""

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile


HERE = Path(__file__).resolve().parent
ARM_ROOT = HERE.parent
MODULE_ROOT = ARM_ROOT.parent
REMOTE = MODULE_ROOT.parent
PROTOCOL_SHA = '3e5e8937e0889aa360ffe3f7acb722ad1ab5f8e6c66cd6b3991a71516212dcfc'
ADDENDUM_SHA = 'e2db4f390756e33e742684e8f1eff526047403ba2db6bad17d5c9e21132e9d54'
ANCHOR_SHA = '4a4a67671cdb6a9419ab47b11db4225cf8642a8096ca4dc6f501148d27cd92f1'
RUNNER_SHA = '7a3934c214f6b196180554c254864ecdc0555dc439bbb555c35b3e610f7e4e93'
LAUNCHER_SHA = '186b84535fdb383b8456047bacc7a41ec991af63f2cea93678a28096c482e490'
PACKAGE_SHA = '7dd93c476068cd8a85a865bac41d01223b51917c9850d9188f8a4fde62504f1b'
OLD_AUDITOR_SHA = '1da46b7e87d81f5eb49edc2de5c4534dfcb46d96ed416fc839f0e7f577d8175c'
ORIGINAL_RUNNER_SHA = '1745f922a91f2b5ee49ece47d7d942612fd636dfd9a151a3c92078fa576115eb'
GPU_UUID = 'GPU-06ccf9c8-5f92-5aad-3458-e162506dd1dc'
BOOT_ID = '2e5d4a32-c510-4f86-8b5a-eb66f03b864a'
RUN_ID = 'no_msca_20261107_v1'
WORKER_PID = 1779501
SUPERVISOR_PID = 1779500
START_EPOCH = 29
PREFIX_UPDATES = 46893
LAST_EPOCH = 319
BATCHES = 1617
REMAINING_UPDATES = 470547
DEPENDENCIES = {
    MODULE_ROOT / 'audit_ablation_lineage.py': OLD_AUDITOR_SHA,
    MODULE_ROOT / 'verify_ablation_backup.py': '47fca690c8acff98d8ba3ac09bcc6b95518d34c1f26c8d47fa38e191047cf9ad',
    REMOTE / 'summarize_completed_pair.py': '3dee0beb969c698efdf7a64b796155672ca29224dd2eb844be2e4d9c4c546615',
    REMOTE / '断点续训_20261105/verify_base_endpoint.py': '534782e8f49bde4a8aaf9a5eef8f780358652380f78d32eca0019dae05c58e5f',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def checked_addendum(path):
    require(sha(path) == ADDENDUM_SHA, 'Exact authorized recovery addendum is missing or changed')
    recovery = read(path)
    expected = {
        'schema': 'authorized_no_msca_round29_recovery_v1', 'authorized': True, 'arm': 'NO_MSCA',
        'original_protocol_sha256': PROTOCOL_SHA, 'original_protocol_unchanged': True,
        'original_run_id': RUN_ID, 'checkpoint_sha256': ANCHOR_SHA,
        'first_epoch_index': START_EPOCH, 'final_epoch_index': LAST_EPOCH,
        'retained_round_limit': 320, 'retained_optimizer_update_limit': 517440,
        'new_segment_evaluation_limit': 291, 'new_segment_optimizer_update_limit': REMAINING_UPDATES,
        'target_ssh_port': 30219, 'target_gpu_uuid': GPU_UUID, 'target_boot_id': BOOT_ID,
        'resume_runner_sha256': RUNNER_SHA, 'automatic_training_retry': False,
        'additional_arms': [], 'additional_seeds': [], 'tuning': False,
    }
    require(all(recovery.get(key) == value and type(recovery.get(key)) is type(value)
                for key, value in expected.items()), 'Recovery authorization or budget differs')
    require(recovery['physical_compute_limitation'].strip(), 'Physical-compute limitation is missing')
    require(PREFIX_UPDATES == START_EPOCH * BATCHES
            and PREFIX_UPDATES + REMAINING_UPDATES == 517440, 'Recovery arithmetic differs')
    return recovery


def function_node(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def verify_runner_ast(root):
    original = root / 'train_runner.py'
    resumed = HERE / 'resume_train_runner.py'
    require(sha(original) == ORIGINAL_RUNNER_SHA and sha(resumed) == RUNNER_SHA,
            'Original or resumed runner changed')
    trees = [ast.parse(path.read_bytes()) for path in (original, resumed)]
    functions = [function_node(tree, 'train_run') for tree in trees]
    for name in ('state', 'load_state', 'restore', 'commit'):
        matches = [[node for node in ast.walk(function) if isinstance(node, ast.FunctionDef)
                    and node.name == name] for function in functions]
        require(all(len(nodes) == 1 for nodes in matches), 'Expected one nested function: ' + name)
        require(ast.dump(matches[0][0]) == ast.dump(matches[1][0]), 'Changed resumed nested function: ' + name)
    loops = [[node for node in ast.walk(function) if isinstance(node, ast.For)
              and isinstance(node.target, ast.Name) and node.target.id == 'epoch'] for function in functions]
    require(all(len(nodes) == 1 for nodes in loops), 'Expected one complete epoch loop')
    require(ast.dump(loops[0][0]) == ast.dump(loops[1][0]), 'Changed resumed epoch loop')
    return {'entire_epoch_loop_identical': True, 'nested_functions_identical': ['state', 'load_state', 'restore', 'commit']}


def patched_auditor_tree():
    path = MODULE_ROOT / 'audit_ablation_lineage.py'
    require(sha(path) == OLD_AUDITOR_SHA, 'Original auditor changed')
    original = ast.parse(path.read_bytes())
    changed = copy.deepcopy(original)
    replacements = {
        "receipt['first_epoch_in_this_segment'] == 0": START_EPOCH,
        "receipt['runner_sha256'] == frozen['train_runner.py']": RUNNER_SHA,
    }
    modifications = []
    audit = function_node(changed, 'audit')
    for node in ast.walk(audit):
        if isinstance(node, ast.Compare) and ast.unparse(node) in replacements:
            before = ast.unparse(node)
            modifications.append({'original_line': node.lineno, 'before': before,
                                  'after_value': replacements[before]})
            node.comparators = [ast.copy_location(ast.Constant(replacements[before]), node.comparators[0])]
    require(len(modifications) == 2 and {item['before'] for item in modifications} == set(replacements),
            'Original final receipt comparisons are not the two exact expected expressions')
    restored = copy.deepcopy(changed)
    for node in ast.walk(function_node(restored, 'audit')):
        if isinstance(node, ast.Compare):
            text = ast.unparse(node)
            if text == f"receipt['first_epoch_in_this_segment'] == {START_EPOCH}":
                node.comparators = [ast.Constant(0)]
            elif text == f"receipt['runner_sha256'] == '{RUNNER_SHA}'":
                node.comparators = [ast.Subscript(ast.Name('frozen', ast.Load()),
                                               ast.Constant('train_runner.py'), ast.Load())]
    require(ast.dump(restored) == ast.dump(original), 'Auditor changed outside the two approved comparisons')
    ast.fix_missing_locations(changed)
    return changed, modifications


def load_auditor():
    for path, expected in DEPENDENCIES.items():
        require(sha(path) == expected, 'Audit dependency changed: ' + str(path))
    sys.path.insert(0, str(MODULE_ROOT))
    tree, modifications = patched_auditor_tree()
    namespace = {'__name__': '_exact_recovery_auditor', '__file__': str(MODULE_ROOT / 'audit_ablation_lineage.py')}
    exec(compile(tree, namespace['__file__'], 'exec'), namespace)
    helper = sys.modules['verify_ablation_backup']
    require(Path(helper.__file__).resolve() == (MODULE_ROOT / 'verify_ablation_backup.py').resolve(),
            'Unexpected backup audit helper import')
    return namespace, modifications


def check_launch_record(record, remote_root):
    command = ['/25085400120/envs/deim/bin/python', '-B', '-u', str(remote_root / 'resume_train_runner.py'),
               '--protocol-sha256', PROTOCOL_SHA, '--recovery-sha256', ADDENDUM_SHA]
    require(record['arm'] == 'NO_MSCA' and record['protocol_sha256'] == PROTOCOL_SHA
            and record['runner_sha256'] == RUNNER_SHA and record['recovery_addendum_sha256'] == ADDENDUM_SHA
            and record['boot_id'] == BOOT_ID and record['command'] == command
            and record['pid'] == WORKER_PID and record['supervisor_pid'] == SUPERVISOR_PID
            and record['start_ticks'] == 18098010 and record['supervisor_start_ticks'] == 18097864
            and record['automatic_training_retry'] is False and record['hard_timeout_seconds'] == 604800,
            'Recovered supervisor identity, command, boot or budget differs')


def validate_local_plan(root):
    root = root.resolve(strict=True)
    require(root == ARM_ROOT.resolve(strict=True) and sha(root / 'execution_protocol.json') == PROTOCOL_SHA,
            'This auditor accepts only the exact NO_MSCA recovery root and protocol')
    recovery = checked_addendum(HERE / 'recovery_addendum.json')
    plan = read(root / 'execution_protocol.json')
    require(plan['arm'] == 'NO_MSCA' and plan['authorized'] is True and plan['epochs'] == 320
            and plan['seed'] == 0 and plan['batch_size'] == 4
            and plan['optimizer_update_limit_per_arm'] == 517440
            and plan['configuration_sha256'] == recovery['configuration_sha256'], 'Original plan differs')
    for item in plan['frozen_files']:
        path = (root / item['path']).resolve(strict=True)
        require(path.is_relative_to(root) and sha(path) == item['sha256'], 'Frozen recovery audit input differs')
    require(sha(HERE / 'resume_launch_arm.py') == LAUNCHER_SHA, 'Resume supervisor launcher changed')
    ast_checks = verify_runner_ast(root)
    preflight_path = HERE / 'target_preflight_readonly.json'
    require(sha(preflight_path) == recovery['runtime_and_data_audit_sha256'], 'Target preflight record changed')
    observed = read(preflight_path)
    environment = read(root / 'environment.current.json')
    require(observed['boot_id'] == BOOT_ID and observed['gpu'].split(', ')[0] == GPU_UUID
            and observed['gpu'].split(', ')[1] == 'NVIDIA GeForce RTX 3090'
            and observed['target_exists'] is False and not observed['gpu_processes']
            and observed['python'] == environment['python'] and observed['executable'] == environment['executable'],
            'Target host or original runtime differs')
    require(all(observed['runtime'][key] == environment[key] for key in ('torch', 'cuda', 'cudnn', 'torch_build')),
            'Target runtime build differs')
    normalize = lambda name: name.lower().replace('_', '-').replace('.', '-')
    installed = {normalize(key): value for key, value in observed['runtime']['packages'].items()}
    require(all(installed.get(normalize(key)) == value for key, value in environment['packages'].items()),
            'Target package versions differ')
    require(observed['dataset_current_byte_verification']['files'] == 7021
            and observed['dataset_current_byte_verification']['mismatches'] == [], 'Target dataset proof differs')
    config = read(root / 'execution_configs/NO_MSCA.yml')
    remote_root = PurePosixPath(config['output_dir']).parent
    require(remote_root.name == RUN_ID, 'Original backup identity changed')
    launch = read(HERE / 'recovery_launch_result.json')
    require(launch['exit_code'] == 0, 'Recovery launch command failed')
    launch_record = json.loads(launch['stdout'])
    require(launch_record['status'] == 'SUPERVISOR_SPAWNED' and launch_record['protocol_sha256'] == PROTOCOL_SHA
            and launch_record['supervisor_pid'] == SUPERVISOR_PID and launch_record['start_ticks'] == 18097864,
            'Launch result differs')
    live = read(HERE / 'live_after_launch.json')
    require(live['exit_code'] == 0, 'Initial remote process observation failed')
    observation = json.loads(live['stdout'])
    require(observation['boot_id'] == BOOT_ID and observation['gpu'].split(', ')[0] == GPU_UUID,
            'Observed launch host differs')
    check_launch_record(observation['records']['supervisor.json'], remote_root)
    worker_record = read(HERE / 'live_first_update.json')
    require(worker_record['exit_code'] == 0, 'Initial new-worker observation failed')
    worker_observation = json.loads(worker_record['stdout'])
    require(worker_observation['boot_id'] == BOOT_ID and worker_observation['gpu'].split(', ')[0] == GPU_UUID,
            'Observed new-worker host differs')
    check_launch_record(worker_observation['records']['supervisor.json'], remote_root)
    require(worker_observation['records']['run/receipt.json']['pid'] == WORKER_PID,
            'Observed new-worker receipt differs')
    return recovery, plan, remote_root, ast_checks


def verify_prefix(root, backups, namespace, recovery):
    package = HERE / 'no_msca_round29_recovery_package.zip'
    require(sha(package) == PACKAGE_SHA, 'Exact installed recovery package is missing or changed')
    spec = importlib.util.spec_from_file_location('_recovery_backup_protocol', root / 'backup_protocol.py')
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    count = 0
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) == 359, 'Unexpected recovery archive member set')
        for name in names:
            relative = PurePosixPath(name)
            require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe recovery archive member')
        for item in recovery['recovery_files']:
            require(hashlib.sha256(archive.read(item['path'])).hexdigest() == item['sha256'],
                    'Recovery package input differs')
        for epoch in range(-1, START_EPOCH):
            phase = 'probe' if epoch == -1 else f'epoch{epoch:04d}'
            folder = backups / 'probe' if epoch == -1 else backups / 'epochs' / phase
            payload = (folder / 'manifest.json').read_bytes()
            manifest = protocol.parse_manifest(payload, RUN_ID, epoch)
            expected_ack = protocol.ack_bytes(payload, manifest)
            require((folder / 'ack.json').read_bytes() == expected_ack
                    and (folder / 'accepted.json').read_bytes() == protocol.accepted_bytes(payload, manifest),
                    'Original prefix backup acceptance differs')
            for name in ('manifest.json', 'ack.json', 'accepted.json'):
                require((folder / name).read_bytes() == archive.read(f'backup_gate/{phase}/{name}'),
                        'Original prefix gate identity differs from the exact recovery package')
            for item in manifest['files']:
                path = protocol.checked_local(folder / item['name'], folder)
                require(protocol.regular(path).st_size == item['bytes'] and sha(path) == item['sha256'],
                        'Original prefix backup file differs')
                count += 1
            protocol.check_metric_files(manifest, lambda name: protocol.local_metadata(folder / name))
    last = backups / 'epochs/epoch0028'
    payload = (last / 'manifest.json').read_bytes()
    manifest = protocol.parse_manifest(payload, RUN_ID, 28)
    require(manifest['rolling_sha256'] == ANCHOR_SHA and manifest['next_epoch'] == START_EPOCH
            and manifest['optimizer_step_attempts'] == PREFIX_UPDATES, 'Original prefix recovery cursor differs')
    rows = read_rows(last / 'epochs.jsonl')
    require(len(rows) == START_EPOCH and rows[-1] == read(last / 'epoch.metrics.json')
            and all(row['epoch'] == index and row['optimizer_step_attempts'] == (index + 1) * BATCHES
                    for index, row in enumerate(rows)), 'Original 29-round prefix counters differ')
    require(sha(last / 'next_epoch.pt') == ANCHOR_SHA, 'Original round-29 checkpoint changed')
    import torch
    state = torch.load(last / 'next_epoch.pt', map_location='cpu', weights_only=True)
    plan = read(root / 'execution_protocol.json')
    frozen = {item['path']: item['sha256'] for item in plan['frozen_files']}
    require(state['resume_format'] == 'fair_runner_post_policy_epoch_v1' and state['arm'] == 'NO_MSCA'
            and state['protocol_sha256'] == PROTOCOL_SHA and state['configuration_sha256'] == protocol.CONFIG
            and state['environment_sha256'] == frozen['environment.current.json']
            and state['initialization_sha256'] == plan['probe_files']['initial.pt']
            and state['parent_protocol_sha256'] == plan['parent_protocol_sha256']
            and state['weight_source'] == 'ema.module'
            and state['actual_loop_epoch'] == state['last_epoch'] == 28 and state['next_epoch'] == 29
            and state['optimizer_step_attempts'] == state['ema']['updates'] == PREFIX_UPDATES
            and state['latest_evaluated_row'] == rows[-1]
            and state['best'] == manifest['best'] and state['stage_one'] == manifest['stage_one'],
            'Original checkpoint state or provenance differs')
    require({'model', 'ema', 'optimizer', 'criterion', 'postprocessor', 'lr_warmup_scheduler',
             'ema_decay_attribute', 'rng_state'} <= state.keys(), 'Incomplete recovery state')
    namespace['verify_tensor_schema'](root, state, read(root / 'execution_protocol.json'))
    namespace['verify_rng_states'](state['rng_state'])
    require(namespace['tensor_sha'](state['ema']['module']) == rows[-1]['ema_tensor_sha256']
            and [group['lr'] for group in state['optimizer']['param_groups']] == rows[-1]['post_final_update_lrs'],
            'Original checkpoint EMA or optimizer LR differs')
    selected_path = last / protocol.selected_folder(manifest['best']) / 'best.pt'
    require(manifest['best']['epoch'] == 28 and manifest['stage_one'] == manifest['best']
            and sha(selected_path) == manifest['best']['checkpoint_sha256'], 'Original selected boundary differs')
    selected = torch.load(selected_path, map_location='cpu', weights_only=True)

    def same_state(left, right):
        if torch.is_tensor(left):
            return torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left, right)
        if isinstance(left, dict):
            return isinstance(right, dict) and left.keys() == right.keys() and all(
                same_state(value, right[key]) for key, value in left.items())
        if isinstance(left, (tuple, list)):
            return type(left) is type(right) and len(left) == len(right) and all(
                same_state(a, b) for a, b in zip(left, right))
        return type(left) is type(right) and left == right

    sections = ('model', 'ema', 'optimizer', 'criterion', 'postprocessor', 'lr_warmup_scheduler')
    require(all(same_state(state[key], selected[key]) for key in sections),
            'Rolling anchor training state differs from the selected round-29 checkpoint')
    return {'retained_rounds': START_EPOCH, 'retained_update_attempts': PREFIX_UPDATES,
            'prefix_manifest_and_acknowledgement_records': 30, 'prefix_file_hashes_checked': count,
            'checkpoint_sha256': ANCHOR_SHA, 'all_saved_rng_sections_valid': True,
            'anchor_state_sections_equal_selected_checkpoint': list(sections)}, rows


def verify_recovery_contract(root, backups):
    recovery, plan, remote_root, ast_checks = validate_local_plan(root)
    namespace, modifications = load_auditor()
    root, backups = root.resolve(strict=True), backups.resolve(strict=True)
    prefix, rows = verify_prefix(root, backups, namespace, recovery)
    result = {'status': 'EXACT_RECOVERY_CONTRACT_AND_29_ROUND_PREFIX_VERIFIED',
              'protocol_sha256': PROTOCOL_SHA, 'recovery_addendum_sha256': ADDENDUM_SHA,
              'package_sha256': PACKAGE_SHA, 'original_auditor_sha256': OLD_AUDITOR_SHA,
              'resume_runner_sha256': RUNNER_SHA, 'target_gpu_uuid': GPU_UUID, 'target_boot_id': BOOT_ID,
              'ast_checks': ast_checks, 'only_old_audit_changes': modifications, 'prefix': prefix,
              'remaining_evaluations': 291, 'remaining_optimizer_update_attempts': REMAINING_UPDATES,
              'physical_compute_limitation': recovery['physical_compute_limitation'],
              'live_recovery_training_progress_verified': False,
              'model_forwards': 0, 'evaluator_runs': 0}
    return result, namespace, rows, remote_root


def audit(root, backups, epoch, expected_protocol):
    require(expected_protocol == PROTOCOL_SHA and START_EPOCH <= epoch <= LAST_EPOCH,
            'Expected an accepted recovered epoch index 29 through 319 and the exact original protocol')
    contract, namespace, prefix_rows, remote_root = verify_recovery_contract(root, backups)
    folder = backups / 'epochs' / f'epoch{epoch:04d}'
    original = backups / 'epochs/epoch0028'
    for name in ('epochs.jsonl', 'executed_learning_rates.csv.gz'):
        prefix = (original / name).read_bytes()
        with (folder / name).open('rb') as stream:
            require(stream.read(len(prefix)) == prefix, 'Recovered cumulative history prefix changed: ' + name)
    rows = read_rows(folder / 'epochs.jsonl')
    require(rows[:START_EPOCH] == prefix_rows and len(rows) == epoch + 1
            and rows[START_EPOCH]['first_executed_lrs'] == prefix_rows[-1]['post_final_update_lrs'],
            'Recovered rows or first executed LR changed the original boundary')
    receipt = read(folder / 'receipt.json')
    require(receipt['pid'] == WORKER_PID and receipt['first_epoch_in_this_segment'] == START_EPOCH
            and receipt['runner_sha256'] == RUNNER_SHA and receipt['recovery_addendum_sha256'] == ADDENDUM_SHA
            and receipt['resumed_from_checkpoint_sha256'] == ANCHOR_SHA,
            'Saved snapshot is not from this exact recovered worker')
    resumed_rounds = epoch + 1 - START_EPOCH
    require(1 <= resumed_rounds <= 291 and resumed_rounds * BATCHES <= REMAINING_UPDATES,
            'Recovered segment exceeds the authorized remaining budget')
    if epoch == LAST_EPOCH:
        final = read(backups / 'completed_run_receipt.json')
        supervisor = read(backups / 'completed_supervisor.json')
        require(final['recovery_addendum_sha256'] == ADDENDUM_SHA
                and final['resumed_from_checkpoint_sha256'] == ANCHOR_SHA and final['pid'] == WORKER_PID,
                'Final receipt lacks the exact recovery linkage')
        check_launch_record(supervisor, remote_root)
    result = namespace['audit'](root, backups, epoch, expected_protocol)
    result['original_analysis_script_sha256'] = result['analysis_script_sha256']
    result['analysis_script_sha256'] = sha(Path(__file__))
    result['recovery'] = {**contract, 'new_completed_rounds': resumed_rounds,
                          'new_optimizer_update_attempts': resumed_rounds * BATCHES,
                          'retained_plus_new_rounds': epoch + 1}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm-root', type=Path, required=True)
    parser.add_argument('--backup-root', type=Path, required=True)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--verify-recovery-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.protocol_sha256 == PROTOCOL_SHA, 'Incorrect original protocol')
    require(args.output.resolve().is_relative_to(MODULE_ROOT.resolve()),
            'Audit output must stay within the writable local ablation workspace')
    if args.verify_recovery_only:
        require(args.epoch is None, 'Recovery-only verification does not accept a later epoch')
        result = verify_recovery_contract(args.arm_root, args.backup_root)[0]
    else:
        require(args.epoch is not None, 'A recovered epoch is required')
        result = audit(args.arm_root, args.backup_root, args.epoch, args.protocol_sha256)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
