"""Strict local saved-state audit of the second NO_MSCA recovery; never run a model.

The external binding SHA must be supplied by the controlling agent after it seals
segment30715_audit_binding.json exclusively. No PID or artifact SHA override is
accepted on the command line. Missing preflight, archive, anchor, transport,
launch, or actual new-worker evidence is a failure, including contract-only mode.
"""

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
import zipfile


HERE = Path(__file__).resolve().parent
ARM = HERE.parent
MODULE = ARM.parent
PREVIOUS = ARM / 'recovery30219_r29_20261108'
BACKUPS = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA/backups')
REMOTE_ROOT = PurePosixPath('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1/no_msca_20261107_v1')
RUN_ID = REMOTE_ROOT.name
PYTHON = '/25085400120/envs/deim/bin/python'
PROTOCOL = '3e5e8937e0889aa360ffe3f7acb722ad1ab5f8e6c66cd6b3991a71516212dcfc'
CONFIG = '8b32bc2c5a620dbed5a758a64c3e891f76acf78ef8deab615da7146f6c83a438'
ORIGINAL_RUNNER = '1745f922a91f2b5ee49ece47d7d942612fd636dfd9a151a3c92078fa576115eb'
OLD_AUDITOR = '1da46b7e87d81f5eb49edc2de5c4534dfcb46d96ed416fc839f0e7f577d8175c'
PREVIOUS_AUDITOR = '1b8aa1fac1089bc8a75dcc4e4cfd2f51705876a060b90663f467c19bdcd58425'
PREVIOUS_ADDENDUM = 'e2db4f390756e33e742684e8f1eff526047403ba2db6bad17d5c9e21132e9d54'
PREVIOUS_RUNNER = '7a3934c214f6b196180554c254864ecdc0555dc439bbb555c35b3e610f7e4e93'
RUNNER = 'f5f7ef88ace026c5236734dfeeb29cdbc4688617f90fc32c3d6abb70b3fe0e20'
LAUNCHER = '2bfa6a7f8db57e5e62d09e9a83d03d6f92e1d6464275108dc949dc304580f99a'
ANCHOR = '512dfe64c5b99479ffc2235054ee1bc1b2b5378ac776e7bead4d57395b3e545d'
PREFIX_MANIFEST = '8bf322a46106ffc2113e65ddc4710ef5e280f43b58466be18866ebf2dae9e8a0'
PREFIX_CERTIFICATE = 'e8bfae7b6c31111d225bdabe07eabcff6bd2aa18d8398f00deb8515f9bfba53a'
PREFIX_REFERENCE = 'b36b5591326e6e9dbfbf24a0262f75dc6ac6560d863fa6375e6a86460cfd4fee'
BACKUP_CLIENT = '4e5c38bb7dcf4dbd8131157b34f2637de7e45379796b75c9e211a63a9f25b37b'
BACKUP_PROTOCOL = '2a7bf275be22108cfda5753e660a78d8bbc86bbecc37bd185294f1b7d6a27a8f'
ADAPTER = 'b6fbfc08b8445ba5126d370e9ab2539b3b35fed986b23105dc2707a048d887b6'
GPU = 'GPU-ca8d11de-f221-f2b9-82bf-d389523dd71a'
BOOT = '2e5d4a32-c510-4f86-8b5a-eb66f03b864a'
START = 164
BATCHES = 1617
REMAINING = 252252
RNG_KEYS = {'python', 'numpy', 'torch_cpu', 'torch_cuda_all', 'train_loader', 'train_sampler', 'validation_loader'}
ARTIFACTS = {
    'addendum': 'recovery30715_addendum.json',
    'runtime_preflight_script': 'read_runtime_and_state_preflight.py',
    'runtime_preflight': 'runtime_and_state_preflight.json',
    'data_restore_script': 'restore_exact_data_cache.py',
    'restored_source_records': 'restored_source_records.json',
    'archive_script': 'archive_prior_segment.py',
    'archive_manifest': 'prior_segment_archive_manifest.json',
    'archive_zip': 'prior_segment_archive.zip',
    'transport_adapter': 'backup_monitor_30715_adapter.py',
    'transport_verification': 'backup_port_adapter_verified.json',
    'launch_result': 'launch_result.json',
    'new_worker_observation': 'new_worker_observation.json',
}
HEX = re.compile(r'[0-9a-f]{64}\Z')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, 'Duplicate JSON key in ' + str(path))
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def exact(value, expected, context):
    require(type(value) is type(expected) and value == expected, context)


def local(relative):
    value = PurePosixPath(relative)
    require(type(relative) is str and not value.is_absolute() and '..' not in value.parts
            and value.as_posix() == relative, 'Unsafe evidence path')
    path = HERE.joinpath(*value.parts).resolve(strict=True)
    require(path.is_relative_to(HERE.resolve()) and path.is_file(), 'Evidence outside the recovery directory')
    return path


def function(tree, name):
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(found) == 1, 'Expected one top-level function: ' + name)
    return found[0]


def ast_sha(tree):
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode('utf-8')).hexdigest()


def import_checked(name, path, expected):
    require(sha(path) == expected, 'Frozen audit dependency changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def binding_record(expected_sha):
    require(type(expected_sha) is str and HEX.fullmatch(expected_sha), 'An externally sealed binding SHA is required')
    path = local('segment30715_audit_binding.json')
    require(sha(path) == expected_sha, 'Controlling-agent binding file is missing or changed')
    record = read(path)
    exact(record.get('schema'), 'second_no_msca_recovery_audit_binding_v1', 'Unexpected binding schema')
    require(record.get('auditor_sha256') == sha(Path(__file__))
            and record.get('bootstrap_sha256') == sha(local('audit_resume30715_bootstrap.py')),
            'Binding is not sealed to this exact auditor and bootstrap')
    artifacts = record.get('artifacts')
    require(isinstance(artifacts, dict) and set(artifacts) == set(ARTIFACTS), 'Missing or unexpected bound evidence')
    paths = {}
    for key, filename in ARTIFACTS.items():
        item = artifacts[key]
        require(isinstance(item, dict) and set(item) == {'relative_path', 'sha256'}
                and item['relative_path'] == filename and HEX.fullmatch(item['sha256']), 'Unrecognized bound artifact: ' + key)
        paths[key] = local(filename)
        require(sha(paths[key]) == item['sha256'], 'Bound evidence changed: ' + filename)
    target = record.get('target')
    fields = {'gpu_uuid', 'boot_id', 'supervisor_pid', 'supervisor_start_ticks', 'worker_pid', 'worker_start_ticks'}
    require(isinstance(target, dict) and set(target) == fields, 'Missing exact new process identity')
    exact(target['gpu_uuid'], GPU, 'New GPU binding differs')
    exact(target['boot_id'], BOOT, 'New boot binding differs')
    require(all(type(target[key]) is int and target[key] > 0 for key in fields - {'gpu_uuid', 'boot_id'}),
            'New PID/tick identity must contain positive integers')
    require(target['worker_pid'] != target['supervisor_pid']
            and (target['worker_pid'], target['worker_start_ticks']) != (1779501, 18098010), 'Old worker identity cannot be reused')
    return record, paths


def verify_scientific_ast():
    sources = [(ARM / 'train_runner.py', ORIGINAL_RUNNER),
               (PREVIOUS / 'resume_train_runner.py', PREVIOUS_RUNNER),
               (local('resume30715_train_runner.py'), RUNNER)]
    require(all(sha(path) == expected for path, expected in sources), 'One scientific runner source changed')
    require(sha(local('resume30715_launch_arm.py')) == LAUNCHER, 'Second launcher changed')
    trees = [ast.parse(path.read_bytes()) for path, _ in sources]
    functions = [function(tree, 'train_run') for tree in trees]
    checks = {}
    for name in ('state', 'load_state', 'restore', 'commit'):
        found = [[node for node in ast.walk(fn) if isinstance(node, ast.FunctionDef) and node.name == name]
                 for fn in functions]
        require(all(len(items) == 1 for items in found), 'Expected one nested function: ' + name)
        hashes = [ast_sha(items[0]) for items in found]
        require(len(set(hashes)) == 1, 'Changed scientific nested AST: ' + name)
        checks[name] = hashes[0]
    loops = [[node for node in ast.walk(fn) if isinstance(node, ast.For)
              and isinstance(node.target, ast.Name) and node.target.id == 'epoch'] for fn in functions]
    require(all(len(items) == 1 for items in loops), 'Expected one whole epoch loop')
    hashes = [ast_sha(items[0]) for items in loops]
    require(len(set(hashes)) == 1, 'Changed complete scientific epoch loop')
    checks['entire_epoch_loop'] = hashes[0]
    require(len({ast_sha(function(tree, 'capture_rng')) for tree in trees}) == 1, 'capture_rng changed')
    require(ast_sha(function(trees[1], 'restore_rng')) == ast_sha(function(trees[2], 'restore_rng')), 'restore_rng changed')
    return checks


def patched_original_auditor(previous):
    path = MODULE / 'audit_ablation_lineage.py'
    require(sha(path) == OLD_AUDITOR, 'Original full-lineage auditor changed')
    for dependency, expected in previous.DEPENDENCIES.items():
        require(sha(dependency) == expected, 'Original full-lineage audit dependency changed')
    original = ast.parse(path.read_bytes())
    changed = copy.deepcopy(original)
    wanted = {"receipt['first_epoch_in_this_segment'] == 0": START,
              "receipt['runner_sha256'] == frozen['train_runner.py']": RUNNER}
    modifications = []
    for node in ast.walk(function(changed, 'audit')):
        if isinstance(node, ast.Compare) and ast.unparse(node) in wanted:
            text = ast.unparse(node)
            modifications.append({'original_line': node.lineno, 'before': text, 'after_value': wanted[text]})
            node.comparators = [ast.copy_location(ast.Constant(wanted[text]), node.comparators[0])]
    require(len(modifications) == 2 and {item['before'] for item in modifications} == set(wanted),
            'Expected exactly the two original final-receipt comparisons')
    restored = copy.deepcopy(changed)
    for node in ast.walk(function(restored, 'audit')):
        if isinstance(node, ast.Compare):
            text = ast.unparse(node)
            if text == "receipt['first_epoch_in_this_segment'] == 164":
                node.comparators = [ast.Constant(0)]
            elif text == f"receipt['runner_sha256'] == '{RUNNER}'":
                node.comparators = [ast.Subscript(ast.Name('frozen', ast.Load()), ast.Constant('train_runner.py'), ast.Load())]
    require(ast.dump(restored) == ast.dump(original), 'Original auditor changed outside the two comparisons')
    ast.fix_missing_locations(changed)
    namespace = {'__name__': '_exact_second_recovery_original_audit', '__file__': str(path)}
    sys.path.insert(0, str(MODULE))
    exec(compile(changed, str(path), 'exec'), namespace)
    helper = sys.modules['verify_ablation_backup']
    require(Path(helper.__file__).resolve() == (MODULE / 'verify_ablation_backup.py').resolve(), 'Wrong original audit helper')
    return namespace, modifications


def verify_prefix(previous, backups):
    # Explicitly retain the historical29 package, preflight and real first launch proof.
    historical_contract, _, prefix29_rows, old_remote_root = previous.verify_recovery_contract(ARM, backups)
    require(PurePosixPath(old_remote_root) == REMOTE_ROOT, 'Original remote backup identity changed')
    certificate_path = PREVIOUS / 'lineage_recovered_through_round164.json'
    require(sha(certificate_path) == PREFIX_CERTIFICATE, 'Original independently accepted164 certificate changed')
    certificate = read(certificate_path)
    require(certificate['status'] == 'VERIFIED_PARTIAL_ABLATION_LINEAGE' and certificate['completed_rounds'] == START
            and certificate['learning_rates']['recorded_attempts'] == 265188
            and certificate['learning_rates']['maximum_absolute_difference'] == 0
            and certificate['maximum_latest_metric_error'] == 0
            and certificate['model_forwards'] == certificate['evaluator_runs'] == 0, 'Original164 certificate differs')
    reference_path = local('prefix164_metadata_reference.json')
    require(sha(reference_path) == PREFIX_REFERENCE, '495-entry prefix reference changed')
    reference = read(reference_path)
    exact(reference['retained_rounds'], START, 'Prefix reference round count differs')
    entries = reference['entries']
    expected_paths = {f'backup_gate/{"probe" if epoch == -1 else f"epoch{epoch:04d}"}/{name}'
                      for epoch in range(-1, START) for name in ('manifest.json', 'ack.json', 'accepted.json')}
    require(len(entries) == 495 and {item['path'] for item in entries} == expected_paths, 'Incomplete or duplicated495 metadata reference')
    protocol = import_checked('_second_recovery_backup_protocol', ARM / 'backup_protocol.py', BACKUP_PROTOCOL)
    references = {item['path']: item for item in entries}
    hash_cache, file_count = {}, 0
    for epoch in range(-1, START):
        phase = protocol.phase_name(epoch)
        folder = backups / 'probe' if epoch == -1 else backups / 'epochs' / phase
        for name in ('manifest.json', 'ack.json', 'accepted.json'):
            path, item = folder / name, references[f'backup_gate/{phase}/{name}']
            require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Accepted prefix metadata changed')
        payload = (folder / 'manifest.json').read_bytes()
        manifest = protocol.parse_manifest(payload, RUN_ID, epoch)
        require((folder / 'ack.json').read_bytes() == protocol.ack_bytes(payload, manifest)
                and (folder / 'accepted.json').read_bytes() == protocol.accepted_bytes(payload, manifest), 'Accepted prefix nonce/ACK changed')
        for item in manifest['files']:
            path = protocol.checked_local(folder / item['name'], folder)
            info = protocol.regular(path)
            require(info.st_size == item['bytes'], 'Accepted prefix file size differs')
            signature = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, item['sha256'])
            if signature not in hash_cache:
                require(sha(path) == item['sha256'], 'Accepted prefix file SHA differs: ' + str(path))
                hash_cache[signature] = True
            file_count += 1
        protocol.check_metric_files(manifest, lambda name: protocol.local_metadata(folder / name))
    folder = backups / 'epochs/epoch0163'
    require(sha(folder / 'manifest.json') == PREFIX_MANIFEST and sha(folder / 'next_epoch.pt') == ANCHOR, 'Accepted164 cursor changed')
    manifest = read(folder / 'manifest.json')
    prefix_rows = rows(folder / 'epochs.jsonl')
    require(len(prefix_rows) == START and prefix_rows[:29] == prefix29_rows, 'Original29/164 cumulative row linkage changed')
    # This calls only the unmodified saved-state/array arithmetic audit, never an evaluator.
    recomputed = previous.audit(ARM, backups, 163, PROTOCOL)
    require(recomputed == certificate, 'Independent164 certificate no longer reproduces from preserved saved evidence')
    import torch
    original = torch.load(folder / 'next_epoch.pt', map_location='cpu', weights_only=True)
    anchor_path = local('recovery30715_anchor/next_epoch.pt')
    require(sha(anchor_path) == ANCHOR, 'New isolated anchor differs from accepted164 bytes')
    anchor = torch.load(anchor_path, map_location='cpu', weights_only=True)

    def same(left, right):
        if torch.is_tensor(left):
            return torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
        if isinstance(left, dict):
            return isinstance(right, dict) and left.keys() == right.keys() and all(same(value, right[key]) for key, value in left.items())
        if isinstance(left, (list, tuple)):
            return type(left) is type(right) and len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
        return type(left) is type(right) and left == right

    require(same(original, anchor), 'Full raw/EMA/optimizer/warmup/RNG anchor state differs')
    require(anchor['next_epoch'] == 164 and anchor['actual_loop_epoch'] == anchor['last_epoch'] == 163
            and anchor['optimizer_step_attempts'] == anchor['ema']['updates'] == 265188
            and anchor['latest_evaluated_row'] == prefix_rows[-1], 'New anchor boundary differs')
    require(set(anchor['rng_state']) == RNG_KEYS, 'New anchor lost saved RNG sections')
    receipt = read(folder / 'receipt.json')
    require(receipt['pid'] == 1779501 and receipt['first_epoch_in_this_segment'] == 29
            and receipt['runner_sha256'] == PREVIOUS_RUNNER and receipt['recovery_addendum_sha256'] == PREVIOUS_ADDENDUM,
            'Accepted164 prefix does not belong to the original first recovery')
    return protocol, prefix_rows, manifest, reference, {
        'historical_round29_contract': historical_contract, 'retained_rounds': START,
        'retained_optimizer_updates': 265188, 'metadata_records_verified': 495,
        'all_prefix_manifest_files_verified': file_count, 'distinct_file_hash_checks': len(hash_cache),
        'independent164_certificate_sha256': PREFIX_CERTIFICATE, 'prefix_reference_sha256': PREFIX_REFERENCE,
        'anchor_sha256': ANCHOR, 'full_saved_state_including_all_rng_equal': True,
    }


def verify_addendum(path, artifact_sha):
    value = read(path)
    expected = {'authorized': True, 'arm': 'NO_MSCA', 'original_protocol_sha256': PROTOCOL,
                'original_protocol_unchanged': True, 'configuration_sha256': CONFIG, 'original_run_id': RUN_ID,
                'checkpoint_sha256': ANCHOR, 'first_epoch_index': 164, 'final_epoch_index': 319,
                'retained_round_limit': 320, 'retained_optimizer_update_limit': 517440,
                'new_segment_evaluation_limit': 156, 'new_segment_optimizer_update_limit': REMAINING,
                'target_ssh_port': 30715, 'target_gpu_uuid': GPU, 'target_boot_id': BOOT,
                'resume_runner_sha256': RUNNER, 'prior_recovery_addendum_sha256': PREVIOUS_ADDENDUM,
                'automatic_training_retry': False, 'additional_arms': [], 'additional_seeds': [], 'tuning': False}
    require(all(type(value.get(key)) is type(item) and value.get(key) == item for key, item in expected.items()),
            'Second recovery authorization, identity, anchor or budget differs')
    require(sha(path) == artifact_sha and sha(PREVIOUS / 'recovery_addendum.json') == PREVIOUS_ADDENDUM,
            'Second/first recovery addendum chain differs')
    require(type(value.get('physical_compute_limitation')) is str and value['physical_compute_limitation'].strip(),
            'Physical-compute disclosure is missing')
    files = value['recovery_files']
    require(isinstance(files, list) and len({item['path'] for item in files}) == len(files), 'Duplicated recovery inputs')
    for item in files:
        relative = PurePosixPath(item['path'])
        require(not relative.is_absolute() and '..' not in relative.parts and HEX.fullmatch(item['sha256']), 'Unsafe recovery file binding')
        # Original29 files remain rooted at ARM; new files and anchor are prepared in HERE.
        previous_names = {'recovery_addendum.json', 'resume_train_runner.py', 'resume_launch_arm.py', 'resume_ast_equivalence.json'}
        candidate = PREVIOUS / item['path'] if item['path'] in previous_names else HERE.joinpath(*relative.parts)
        require(candidate.resolve(strict=True).is_relative_to(ARM.resolve()) and sha(candidate) == item['sha256'],
                'Bound new recovery input missing or changed: ' + item['path'])
    required = {'recovery30715_anchor/next_epoch.pt': ANCHOR, 'resume30715_launch_arm.py': LAUNCHER}
    by_path = {item['path']: item['sha256'] for item in files}
    require(all(by_path.get(name) == digest for name, digest in required.items()), 'New recovery input set omits anchor or launcher')
    return value


def verify_preflight(paths, manifest, reference):
    report = read(paths['runtime_preflight'])
    require(report['boot_id'] == BOOT and report['gpu_raw'].split(', ')[0] == GPU
            and 'NVIDIA GeForce RTX 3090' in report['gpu_raw'] and not report['compute_processes']
            and report['project_processes'] == [], 'Target preflight lacks the exact idle host/GPU')
    require(int(report['gpu_raw'].split(', ')[2]) >= 20000, 'Insufficient idle target memory')
    environment = read(ARM / 'environment.current.json')
    require(all(report['runtime'][key] == environment[key]
                for key in ('python', 'executable', 'torch', 'cuda', 'cudnn', 'torch_build', 'packages')),
            'Target runtime/package build differs from frozen environment')
    require(report['preflight_source_sha256'] == sha(paths['runtime_preflight_script']), 'Preflight script identity differs')
    plan = read(ARM / 'execution_protocol.json')
    expected = {item['path']: {'path': item['path'], 'sha256': item['sha256'], 'bytes': (ARM / item['path']).stat().st_size}
                for item in plan['frozen_files']}
    actual = report['frozen_files_actual']
    require(len(actual) == len(expected) and {item['path']: item for item in actual} == expected, 'Remote frozen file bytes differ')
    require(all(sha(ARM / name) == item['sha256'] for name, item in expected.items()), 'Local frozen inputs changed')
    actual = report['prefix_metadata_actual']
    require(len(actual) == 495 and {item['path']: item for item in actual}
            == {item['path']: item for item in reference['entries']}, 'Remote accepted prefix495 identity differs')
    metadata = report['rolling_metadata']
    require(report['rolling_checkpoint_sha256'] == ANCHOR
            and metadata['next_epoch'] == 164 and metadata['actual_loop_epoch'] == metadata['last_epoch'] == 163
            and metadata['optimizer_step_attempts'] == 265188 and metadata['resume_format'] == 'fair_runner_post_policy_epoch_v1'
            and set(metadata['rng_keys']) == RNG_KEYS and metadata['cuda_rng_state_count'] == 1,
            'Remote saved rolling state differs from accepted164')
    cache = read(ARM / 'cache_verification.json')
    restoration = report['data_restoration']
    require(restoration['status'] == 'ORIGINAL_CACHE_PATH_7021_FROZEN_FILES_RESTORED_AND_VERIFIED'
            and all(restoration[key] == cache[key] for key in ('archive_sha256', 'manifest_sha256', 'cache_root', 'files', 'train_images', 'validation_images'))
            and restoration['files'] == 7021 and restoration['source_config_changed'] is False
            and restoration['original_cache_verification_record_changed'] is False and restoration['training_started'] is False
            and restoration['restoration_source_sha256'] == sha(paths['data_restore_script']), 'Exact original cache restoration proof differs')
    expected_sources = {item['source']: item for item in manifest['files']}
    actual = report['manifest_source_actual']
    require(len(actual) == len(expected_sources) == 12 and {item['source'] for item in actual} == set(expected_sources), 'Preflight omits protected source records')
    mismatches = []
    for item in actual:
        expected_item = expected_sources[item['source']]
        require(item['name'] == expected_item['name'] and item['expected_sha256'] == expected_item['sha256']
                and item['expected_bytes'] == expected_item['bytes'] and HEX.fullmatch(item['sha256'])
                and type(item['bytes']) is int and item['bytes'] > 0, 'Invalid pre-restore source identity')
        if item['sha256'] != expected_item['sha256'] or item['bytes'] != expected_item['bytes']:
            mismatches.append(item)
    restored = read(paths['restored_source_records'])
    records = restored['files']
    require(len(records) == len(expected_sources) and {item['source'] for item in records} == set(expected_sources), 'Restored source proof is incomplete')
    for item in records:
        expected_item = expected_sources[item['source']]
        path = PurePosixPath(item['path'])
        actual_path = path if path.is_absolute() else REMOTE_ROOT / path
        require(actual_path == REMOTE_ROOT / expected_item['source'] and item['sha256'] == expected_item['sha256']
                and item['bytes'] == expected_item['bytes'], 'A working source was not restored to accepted164 bytes')
    require(report['model_forwards'] == report['optimizer_updates'] == report['evaluator_runs'] == 0,
            'Preflight performed unrequested ML computation')
    return mismatches, report, restored


def verify_archive(paths, mismatches, preflight):
    manifest = read(paths['archive_manifest'])
    require(manifest['project_processes_before'] == manifest['project_processes_after'] == [], 'Prior archive lacks before/after no-worker proof')
    entries = manifest['files']
    require(isinstance(entries, list) and len({item['original_path'] for item in entries}) == len(entries), 'Duplicated archived source')
    root = REMOTE_ROOT / 'segment_history/segment30219_until164'
    by_original, archived = {}, {}
    for entry in entries:
        original, saved = PurePosixPath(entry['original_path']), PurePosixPath(entry['archive_path'])
        original = original if original.is_absolute() else REMOTE_ROOT / original
        saved = saved if saved.is_absolute() else REMOTE_ROOT / saved
        require(original.is_relative_to(REMOTE_ROOT) and saved.is_relative_to(root)
                and '..' not in original.parts and '..' not in saved.parts
                and HEX.fullmatch(entry['sha256']) and type(entry['bytes']) is int and entry['bytes'] >= 0, 'Unsafe archived path or digest')
        relative = saved.relative_to(root).as_posix()
        require(relative not in archived and '..' not in PurePosixPath(relative).parts, 'Duplicated archived ZIP member')
        require(original.as_posix() not in by_original, 'Duplicate normalized archived source')
        archived[relative] = entry
        by_original[original.as_posix()] = entry
    required = ['launch.lock', 'supervisor.json', 'training.log', 'supervisor.log',
                'run/receipt.json', 'run/executed_learning_rates.csv.gz', 'run/epochs.jsonl',
                'run/actual_first100.jsonl', 'run/latest.predictions.json', 'run/latest.evaluation_arrays.npz', 'run/next_epoch.pt']
    require(all(str(REMOTE_ROOT / name) in by_original for name in required), 'Prior mutable history or four launcher records omitted')
    for item in mismatches:
        old = by_original.get(str(REMOTE_ROOT / item['source']))
        require(old is not None and old['sha256'] == item['sha256'] and old['bytes'] == item['bytes'], 'Discarded partial165 source was not preserved before restoration')
    for package in preflight['extra_selected_packages']:
        require(any(name.startswith(str(REMOTE_ROOT / 'run/selected' / package) + '/') for name in by_original), 'Unaccepted selected package was omitted from history')
    with zipfile.ZipFile(paths['archive_zip']) as archive:
        infos = archive.infolist()
        require(len({item.filename for item in infos}) == len(infos), 'Duplicate archive ZIP names')
        files = {item.filename: item for item in infos if not item.is_dir()}
        prefixes = ['', 'segment30219_until164/', 'segment_history/segment30219_until164/']
        matches = [prefix for prefix in prefixes if set(files) == {prefix + name for name in archived}]
        require(len(matches) == 1, 'Archive ZIP member set does not match its complete hash-bound manifest')
        prefix = matches[0]
        for name, item in archived.items():
            info = files[prefix + name]
            relative = PurePosixPath(info.filename)
            require(not relative.is_absolute() and '..' not in relative.parts and not ((info.external_attr >> 16) & 0o170000) == 0o120000,
                    'Unsafe archived ZIP entry')
            digest = hashlib.sha256()
            size = 0
            with archive.open(info) as stream:
                while block := stream.read(4 * 1024**2):
                    digest.update(block)
                    size += len(block)
            require(size == item['bytes'] and digest.hexdigest() == item['sha256'], 'Archive ZIP bytes differ from preserved original')
    return {'archived_files': len(entries), 'zip_bytes_hash_checked': True, 'pre_restore_mismatched_sources_preserved': len(mismatches)}


def verify_transport(paths):
    source_path = BACKUPS.parent / 'backup_monitor.py'
    require(sha(source_path) == BACKUP_CLIENT and sha(BACKUPS.parent / 'backup_protocol.py') == BACKUP_PROTOCOL,
            'D frozen client or protocol changed')
    require(sha(paths['transport_adapter']) == ADAPTER, 'Reviewed transport adapter changed')
    tree = ast.parse(paths['transport_adapter'].read_bytes())
    pure = copy.deepcopy(function(tree, 'patched_tree'))
    require(not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(pure)), 'Transport transformer imports runtime effects')
    ast.fix_missing_locations(pure)
    namespace = {'ast': ast}
    exec(compile(ast.Module(body=[pure], type_ignores=[]), '<pure_port_AST_transform>', 'exec'), namespace)
    original = ast.parse(source_path.read_bytes())
    changed = namespace['patched_tree'](source_path.read_bytes())
    restored = copy.deepcopy(changed)
    matches = []
    for node in ast.walk(restored):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'
                and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == '--port'):
            choices = [item for item in node.keywords if item.arg == 'choices']
            require(len(choices) == 1 and ast.literal_eval(choices[0].value) == [30219, 30848, 30715], 'Unexpected transport port transform')
            choices[0].value.elts.pop()
            matches.append(node)
    require(len(matches) == 1 and ast.dump(restored) == ast.dump(original), 'Transport changed outside the one port append')
    report = read(paths['transport_verification'])
    require(report['status'] == 'PORT30715_ONLY_ADAPTER_STATIC_VERIFIED'
            and report['original_backup_monitor_sha256'] == BACKUP_CLIENT and report['adapter_sha256'] == ADAPTER
            and report['patched_ast_sha256'] == ast_sha(changed) and report['reverse_ast_sha256'] == ast_sha(original)
            and Path(report['original_file_path']).resolve() == source_path.resolve()
            and report['only_change'] == {'argument': '--port', 'before': [30219, 30848], 'after': [30219, 30848, 30715]},
            'Transport AST report differs from independent source transformation')
    return {'adapter_sha256': ADAPTER, 'patched_ast_sha256': ast_sha(changed), 'reverse_ast_sha256': ast_sha(original), 'only_port30715_append': True}


def check_supervisor(record, target, addendum_sha):
    command = [PYTHON, '-B', '-u', str(REMOTE_ROOT / 'resume30715_train_runner.py'),
               '--protocol-sha256', PROTOCOL, '--recovery-sha256', addendum_sha]
    require(record['arm'] == 'NO_MSCA' and record['protocol_sha256'] == PROTOCOL
            and record['runner_sha256'] == RUNNER and record['recovery_addendum_sha256'] == addendum_sha
            and record['boot_id'] == BOOT and record['command'] == command
            and record['pid'] == target['worker_pid'] and record['start_ticks'] == target['worker_start_ticks']
            and record['supervisor_pid'] == target['supervisor_pid'] and record['supervisor_start_ticks'] == target['supervisor_start_ticks']
            and record['automatic_training_retry'] is False and record['hard_timeout_seconds'] == 604800,
            'Supervisor argv, new worker, boot, code or remaining-scope identity differs')


def check_receipt(record, target, addendum_sha):
    require(record['arm'] == 'NO_MSCA' and record['protocol_sha256'] == PROTOCOL
            and record['configuration_sha256'] == CONFIG and record['pid'] == target['worker_pid']
            and record['first_epoch_in_this_segment'] == START and record['runner_sha256'] == RUNNER
            and record['recovery_addendum_sha256'] == addendum_sha and record['resumed_from_checkpoint_sha256'] == ANCHOR,
            'Receipt is not from this exact second recovery')


def verify_new_launch(paths, binding, addendum_sha):
    target = binding['target']
    launch = read(paths['launch_result'])
    require(launch['status'] == 'SUPERVISOR_SPAWNED' and launch['protocol_sha256'] == PROTOCOL
            and launch['supervisor_pid'] == target['supervisor_pid'] and launch['start_ticks'] == target['supervisor_start_ticks'],
            'Bound second launch result differs')
    observation = read(paths['new_worker_observation'])
    gpu = observation.get('gpu_raw', observation.get('gpu'))
    require(observation['boot_id'] == BOOT and isinstance(gpu, str) and gpu.split(', ')[0] == GPU, 'New worker observed on a different host/GPU')
    supervisor, receipt = observation['records']['supervisor.json'], observation['records']['run/receipt.json']
    check_supervisor(supervisor, target, addendum_sha)
    check_receipt(receipt, target, addendum_sha)
    require(supervisor['status'] == 'RUNNING' and receipt['status'] == 'RUNNING'
            and receipt['optimizer_step_attempts'] > 265188 and receipt['actual_loop_epoch'] >= 164,
            'A real new recovery update has not been observed; initialization is not proof of resumed training')
    require(receipt['optimizer_step_attempts'] <= 517440 and receipt['actual_loop_epoch'] <= 319, 'New worker observation exceeds total budget')
    require(receipt['actual_loop_epoch'] * BATCHES <= receipt['optimizer_step_attempts']
            <= (receipt['actual_loop_epoch'] + 1) * BATCHES, 'Observed epoch/update counters disagree')
    processes = observation['processes']
    for key, pid_key, ticks_key in [('supervisor_pid', 'supervisor_pid', 'supervisor_start_ticks'), ('pid', 'worker_pid', 'worker_start_ticks')]:
        process = processes[key]
        require(process['exists'] is True and process['pid'] == target[pid_key]
                and process['start_ticks'] == target[ticks_key], 'Actual /proc PID/start-tick identity was not verified')
        expected_command = ([PYTHON, '-B', '-u', str(REMOTE_ROOT / 'resume30715_launch_arm.py'), 'supervise', PROTOCOL, addendum_sha]
                            if key == 'supervisor_pid' else supervisor['command'])
        require(isinstance(process['command'], str) and shlex.split(process['command']) == expected_command,
                'Actual /proc command differs from the exact new launcher/worker argv')
    return {'new_worker_observed_with_updates': True, 'worker_pid': target['worker_pid'],
            'supervisor_pid': target['supervisor_pid'], 'observed_server_utc': observation['observed_server_utc']}


def verify_contract(backups, expected_binding):
    binding, paths = binding_record(expected_binding)
    require(sha(ARM / 'execution_protocol.json') == PROTOCOL, 'Original protocol changed')
    previous = import_checked('_preserved_original29_recovery_audit', PREVIOUS / 'audit_recovered_ablation_lineage.py', PREVIOUS_AUDITOR)
    ast_checks = verify_scientific_ast()
    protocol, prefix_rows, manifest, reference, prefix_report = verify_prefix(previous, backups)
    addendum_sha = binding['artifacts']['addendum']['sha256']
    addendum = verify_addendum(paths['addendum'], addendum_sha)
    mismatches, preflight, restored_sources = verify_preflight(paths, manifest, reference)
    archive = verify_archive(paths, mismatches, preflight)
    transport = verify_transport(paths)
    launch = verify_new_launch(paths, binding, addendum_sha)
    namespace, changes = patched_original_auditor(previous)
    contract = {'status': 'EXACT_SECOND_RECOVERY_CONTRACT_AND164_PREFIX_VERIFIED',
                'binding_sha256': expected_binding, 'recovery_addendum_sha256': addendum_sha,
                'original_protocol_sha256': PROTOCOL, 'original29_recovery_addendum_sha256': PREVIOUS_ADDENDUM,
                'prefix': prefix_report, 'scientific_ast_checks': ast_checks, 'only_original_auditor_changes': changes,
                'archive': archive, 'transport': transport, 'new_launch': launch,
                'retained_rounds': 164, 'retained_optimizer_updates': 265188,
                'remaining_evaluations': 156, 'remaining_optimizer_updates': REMAINING,
                'physical_compute_limitation': addendum['physical_compute_limitation'],
                'verification_scope': 'Saved-state, accepted-history, recorded-LR and saved-array checks only. Unretained work after the earlier loss and after accepted164 remains physically unquantified; uninterrupted bitwise equivalence is not established.',
                'model_forwards': 0, 'evaluator_runs': 0, 'remote_connections': 0}
    return contract, binding, paths, protocol, prefix_rows, namespace


def audit(backups, epoch, expected_binding):
    require(type(epoch) is int and START <= epoch <= 319, 'Expected a second-segment accepted epoch164 through319')
    contract, binding, paths, protocol, prefix_rows, namespace = verify_contract(backups, expected_binding)
    addendum_sha = binding['artifacts']['addendum']['sha256']
    for index in range(START, epoch + 1):
        folder = backups / 'epochs' / f'epoch{index:04d}'
        payload = (folder / 'manifest.json').read_bytes()
        manifest = protocol.parse_manifest(payload, RUN_ID, index)
        require((folder / 'ack.json').read_bytes() == protocol.ack_bytes(payload, manifest)
                and (folder / 'accepted.json').read_bytes() == protocol.accepted_bytes(payload, manifest), 'Second-segment acceptance differs')
        for item in manifest['files']:
            path = protocol.checked_local(folder / item['name'], folder)
            require(protocol.regular(path).st_size == item['bytes'] and sha(path) == item['sha256'], 'Second-segment protected artifact changed')
        protocol.check_metric_files(manifest, lambda name: protocol.local_metadata(folder / name))
        check_receipt(read(folder / 'receipt.json'), binding['target'], addendum_sha)
    final_folder = backups / 'epochs' / f'epoch{epoch:04d}'
    for name in ('epochs.jsonl', 'executed_learning_rates.csv.gz'):
        prefix = (backups / 'epochs/epoch0163' / name).read_bytes()
        with (final_folder / name).open('rb') as stream:
            require(stream.read(len(prefix)) == prefix, 'Accepted164 cumulative byte prefix changed: ' + name)
    current_rows = rows(final_folder / 'epochs.jsonl')
    require(current_rows[:START] == prefix_rows and len(current_rows) == epoch + 1
            and current_rows[START]['first_executed_lrs'] == prefix_rows[-1]['post_final_update_lrs'],
            'Second segment cumulative rows or LR boundary differ')
    completed = epoch + 1 - START
    require(1 <= completed <= 156 and completed * BATCHES <= REMAINING, 'Second segment exceeds its remaining budget')
    if epoch == 319:
        receipt, supervisor = read(backups / 'completed_run_receipt.json'), read(backups / 'completed_supervisor.json')
        check_receipt(receipt, binding['target'], addendum_sha)
        check_supervisor(supervisor, binding['target'], addendum_sha)
    # Unchanged original full-prefix, LR, late-policy, prediction/array, terminal and successful-exit checks.
    result = namespace['audit'](ARM, backups, epoch, PROTOCOL)
    result['original_analysis_script_sha256'] = result['analysis_script_sha256']
    result['analysis_script_sha256'] = sha(Path(__file__))
    result['recovery'] = {**contract, 'new_completed_rounds': completed,
                          'new_optimizer_update_attempts': completed * BATCHES,
                          'retained_plus_new_rounds': epoch + 1}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm-root', required=True, type=Path)
    parser.add_argument('--backup-root', required=True, type=Path)
    parser.add_argument('--binding-sha256', required=True)
    parser.add_argument('--protocol-sha256', default=PROTOCOL)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--verify-contract-only', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(args.arm_root.resolve(strict=True) == ARM.resolve(), 'Only the exact original C arm root is allowed')
    backups = args.backup_root.resolve(strict=True)
    require(backups == BACKUPS.resolve(strict=True) and args.protocol_sha256 == PROTOCOL, 'Only the original D backup store/protocol is allowed')
    require(args.output.resolve().is_relative_to(HERE.resolve()) and not args.output.exists(), 'Audit output must be a new file inside this recovery directory')
    if args.verify_contract_only:
        require(args.epoch is None, 'Contract-only mode cannot accept an epoch')
        result = verify_contract(backups, args.binding_sha256)[0]
    else:
        require(args.epoch is not None, 'An accepted second-segment epoch is required')
        result = audit(backups, args.epoch, args.binding_sha256)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
