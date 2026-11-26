"""Verify preserved evidence and prepare one bounded, explicitly requested continuation."""
import ast
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import zipfile

here = Path(__file__).resolve().parent
arm = here.parent
backups = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA/backups')
stage = backups / 'epochs/epoch0028'
expected_protocol = '3e5e8937e0889aa360ffe3f7acb722ad1ab5f8e6c66cd6b3991a71516212dcfc'
def sha(path):
    h = sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024**2): h.update(block)
    return h.hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')

current = read(here / 'target_preflight_readonly.json')
expected = read(arm / 'environment.current.json')
assert current['target_exists'] is False and not current['gpu_processes']
assert current['boot_id'] == '2e5d4a32-c510-4f86-8b5a-eb66f03b864a'
assert current['gpu'].startswith('GPU-06ccf9c8-5f92-5aad-3458-e162506dd1dc, NVIDIA GeForce RTX 3090, 2, 24576, 0')
assert current['python'] == expected['python'] and current['executable'] == expected['executable']
for key in ['torch', 'cuda', 'cudnn', 'torch_build']:
    assert current['runtime'][key] == expected[key], key
normalize = lambda name: name.lower().replace('_', '-').replace('.', '-')
actual_packages = {normalize(k): v for k, v in current['runtime']['packages'].items()}
package_mismatches = {name: [version, actual_packages.get(normalize(name))]
                      for name, version in expected['packages'].items()
                      if actual_packages.get(normalize(name)) != version}
assert not package_mismatches, package_mismatches
assert current['dataset_current_byte_verification']['files'] == 7021
assert not current['dataset_current_byte_verification']['mismatches']
assert current['existing_no_mps']['supervisor.json']['content']['status'] == 'COMPLETED'
assert current['existing_no_mps']['supervisor.json']['content']['exit_code'] == 0
assert current['existing_no_mps']['run/receipt.json']['content']['completed_epochs'] == 320
protocol = read(arm / 'execution_protocol.json')
assert sha(arm / 'execution_protocol.json') == expected_protocol
manifest = read(stage / 'manifest.json')
assert manifest['epoch'] == 28 and manifest['next_epoch'] == 29
assert manifest['optimizer_step_attempts'] == 46893
for item in manifest['files']:
    path = stage / item['name']
    assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], item['name']
assert manifest['rolling_sha256'] == '4a4a67671cdb6a9419ab47b11db4225cf8642a8096ca4dc6f501148d27cd92f1'
assert read(arm / 'lineage_through_round029.json')['status'] == 'VERIFIED_PARTIAL_ABLATION_LINEAGE'

launcher = (arm / 'launch_arm.py').read_text(encoding='utf-8')
def replace(old, new):
    global launcher
    assert launcher.count(old) == 1, old[:100]
    launcher = launcher.replace(old, new)
replace('def ready(expected):', 'def ready(expected, recovery_expected):')
replace("    if (HERE / 'run').exists():\n        raise RuntimeError('Run output already exists')\n",
        "    if sha(HERE / 'recovery_addendum.json') != recovery_expected:\n"
        "        raise RuntimeError('Recovery addendum differs')\n"
        "    recovery = read(HERE / 'recovery_addendum.json')\n"
        "    if recovery['authorized'] is not True or recovery['first_epoch_index'] != 29:\n"
        "        raise RuntimeError('Unexpected recovery scope')\n"
        "    for item in recovery['recovery_files']:\n"
        "        path = (HERE / item['path']).resolve(strict=True)\n"
        "        if not path.is_relative_to(HERE) or sha(path) != item['sha256']:\n"
        "            raise RuntimeError('Recovery launch input differs')\n"
        "    if sha(HERE / 'resume_train_runner.py') != recovery['resume_runner_sha256']:\n"
        "        raise RuntimeError('Resume runner differs')\n"
        "    if sha(HERE / 'run/next_epoch.pt') != recovery['checkpoint_sha256']:\n"
        "        raise RuntimeError('Restored rolling checkpoint differs')\n"
        "    if sys.executable != '/25085400120/envs/deim/bin/python':\n"
        "        raise RuntimeError('Unexpected recovery interpreter')\n")
replace('def supervise(expected):\n    plan = ready(expected)\n',
        'def supervise(expected, recovery_expected):\n    plan = ready(expected, recovery_expected)\n')
replace("    command = [sys.executable, '-B', '-u', str(HERE / 'train_runner.py'), '--protocol-sha256', expected]\n",
        "    command = [sys.executable, '-B', '-u', str(HERE / 'resume_train_runner.py'),\n"
        "               '--protocol-sha256', expected, '--recovery-sha256', recovery_expected]\n")
replace("             'runner_sha256': sha(HERE / 'train_runner.py'), 'command': command,\n",
        "             'runner_sha256': sha(HERE / 'resume_train_runner.py'), 'command': command,\n"
        "             'recovery_addendum_sha256': recovery_expected,\n")
replace('def launch(expected):\n    ready(expected)\n',
        'def launch(expected, recovery_expected):\n    ready(expected, recovery_expected)\n')
replace("[sys.executable, '-B', '-u', str(Path(__file__)), 'supervise', expected]",
        "[sys.executable, '-B', '-u', str(Path(__file__)), 'supervise', expected, recovery_expected]")
replace("    args = parser.parse_args()\n    {'launch': launch, 'supervise': supervise}[args.mode](args.protocol_sha256)\n",
        "    parser.add_argument('recovery_sha256')\n"
        "    args = parser.parse_args()\n"
        "    {'launch': launch, 'supervise': supervise}[args.mode](args.protocol_sha256, args.recovery_sha256)\n")
ast.parse(launcher)
with (here / 'resume_launch_arm.py').open('x', encoding='utf-8', newline='\n') as handle:
    handle.write(launcher)

files = {}
def add(path, source):
    assert path not in files and not PurePosixPath(path).is_absolute() and '..' not in PurePosixPath(path).parts
    files[path] = {'source': source, 'sha256': sha(source), 'bytes': source.stat().st_size}

# Keep the original historical gate metadata byte-for-byte; no acknowledgements are invented.
for epoch in range(-1, 29):
    part = backups / 'probe' if epoch == -1 else backups / f'epochs/epoch{epoch:04d}'
    phase = 'probe' if epoch == -1 else f'epoch{epoch:04d}'
    for name in ['manifest.json', 'ack.json', 'accepted.json']:
        add(f'backup_gate/{phase}/{name}', part / name)
for item in manifest['files']:
    if item['name'] == 'next_epoch.pt':
        add('recovery_anchor/next_epoch.pt', stage / item['name'])
    else:
        add(item['source'], stage / item['name'])
for name in ['manifest.json', 'ack.json', 'accepted.json', 'receipt.json']:
    add('recovery_anchor/original_' + name, stage / name)
for name in ['resume_train_runner.py', 'resume_launch_arm.py', 'resume_ast_equivalence.json',
             'target_preflight_readonly.json']:
    add(name, here / name)

addendum = {'schema': 'authorized_no_msca_round29_recovery_v1', 'authorized': True,
    'arm': 'NO_MSCA', 'created_local_utc': datetime.now(timezone.utc).isoformat(),
    'user_authorization': '跑实验  并且把数据整理一下',
    'authorization_context': 'User was told both provided instances have no training, MSCA is not resumed, and a full local round29 backup is retained.',
    'original_protocol_sha256': expected_protocol, 'original_protocol_unchanged': True,
    'configuration_sha256': protocol['configuration_sha256'],
    'original_run_id': 'no_msca_20261107_v1',
    'checkpoint_sha256': manifest['rolling_sha256'], 'first_epoch_index': 29, 'final_epoch_index': 319,
    'retained_round_limit': 320, 'retained_optimizer_update_limit': 517440,
    'new_segment_evaluation_limit': 291, 'new_segment_optimizer_update_limit': 470547,
    'target_ssh_port': 30219, 'target_gpu_uuid': 'GPU-06ccf9c8-5f92-5aad-3458-e162506dd1dc',
    'target_boot_id': current['boot_id'], 'original_gpu_uuid': protocol['gpu_uuid'],
    'resume_runner_sha256': sha(here / 'resume_train_runner.py'),
    'runtime_and_data_audit_sha256': sha(here / 'target_preflight_readonly.json'),
    'recovery_files': [{'path': key, 'sha256': value['sha256']} for key, value in files.items()
                       if key.startswith('recovery_anchor/') or key in ['resume_launch_arm.py',
                          'resume_ast_equivalence.json', 'target_preflight_readonly.json']],
    'unchanged': ['seed0', 'batch4', 'FP32', 'TF32off', 'all frozen neural/data/loss/evaluator sources',
                  'dataset bytes', 'config', 'optimizer/EMA/update/LR/restore policy',
                  'original backup protocol and per-round local verification'],
    'physical_compute_limitation': 'Unretained work after accepted round29 on the lost original volume cannot be quantified. This authorized segment resumes from round29 and does not claim equal physical compute or bitwise equivalence to an uninterrupted run.',
    'automatic_training_retry': False, 'additional_arms': [], 'additional_seeds': [], 'tuning': False}
write(here / 'recovery_addendum.json', addendum)
add('recovery_addendum.json', here / 'recovery_addendum.json')

frozen = arm / 'frozen_inputs_20261108.zip'
assert sha(frozen) == '446613010ba761885ddbaf54d2026f8b7e10aaa0bba6e05be91a8194026a8c75'
archive = here / 'no_msca_round29_recovery_package.zip'
with zipfile.ZipFile(frozen) as original_zip, zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as dest:
    assert len(original_zip.infolist()) == 248
    for item in original_zip.infolist():
        data = original_zip.read(item)
        if item.filename == 'execution_protocol.json':
            assert sha256(data).hexdigest() == expected_protocol
        else:
            expected_item = next(x for x in protocol['frozen_files'] if x['path'] == item.filename)
            assert sha256(data).hexdigest() == expected_item['sha256'], item.filename
        assert item.filename not in files
        dest.writestr(item.filename, data)
    for name, item in files.items():
        dest.write(item['source'], name)
report = {'status': 'BOUND_ROUND29_RECOVERY_PACKAGE_PREPARED',
    'matching_original_dependency_versions': len(expected['packages']),
    'all_runtime_build_fields_equal': True, 'dataset_files_byte_verified': 7021,
    'frozen_original_files': 248, 'retained_backup_rounds': 29,
    'package_bytes': archive.stat().st_size, 'package_sha256': sha(archive),
    'recovery_addendum_sha256': sha(here / 'recovery_addendum.json'),
    'resume_runner_sha256': sha(here / 'resume_train_runner.py'),
    'restored_files': {k: {'sha256': v['sha256'], 'bytes': v['bytes']} for k, v in files.items()},
    'physical_compute_limitation': addendum['physical_compute_limitation'],
    'training_started': False}
write(here / 'recovery_package_verified.json', report)
print(json.dumps({k:v for k,v in report.items() if k != 'restored_files'}))
