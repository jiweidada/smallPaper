"""Build an isolated, hash-bound round164 resume scaffold; never execute training."""

import ast
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARM = HERE.parent
PREVIOUS = ARM / 'recovery30219_r29_20261108'
ORIGINAL_SHA = '1745f922a91f2b5ee49ece47d7d942612fd636dfd9a151a3c92078fa576115eb'
PREVIOUS_RUNNER_SHA = '7a3934c214f6b196180554c254864ecdc0555dc439bbb555c35b3e610f7e4e93'
PREVIOUS_LAUNCHER_SHA = '186b84535fdb383b8456047bacc7a41ec991af63f2cea93678a28096c482e490'
PREVIOUS_ADDENDUM_SHA = 'e2db4f390756e33e742684e8f1eff526047403ba2db6bad17d5c9e21132e9d54'
PROTOCOL_SHA = '3e5e8937e0889aa360ffe3f7acb722ad1ab5f8e6c66cd6b3991a71516212dcfc'
ANCHOR_SHA = '512dfe64c5b99479ffc2235054ee1bc1b2b5378ac776e7bead4d57395b3e545d'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def function(tree, name):
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1, name
    return matches[0]


def nested(function_node, name):
    matches = [node for node in ast.walk(function_node)
               if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1, name
    return matches[0]


def epoch_loop(function_node):
    matches = [node for node in ast.walk(function_node) if isinstance(node, ast.For)
               and isinstance(node.target, ast.Name) and node.target.id == 'epoch']
    assert len(matches) == 1
    return matches[0]


def adapt(source, changes):
    changed = source
    locations = []
    for label, before, after in changes:
        assert changed.count(before) == 1, label
        locations.append({'label': label, 'before': before, 'after': after,
                          'source_line_before_change': changed[:changed.index(before)].count('\n') + 1})
        changed = changed.replace(before, after)
    restored = changed
    for _, before, after in reversed(changes):
        assert restored.count(after) == 1
        restored = restored.replace(after, before)
    assert restored == source
    assert ast.dump(ast.parse(restored)) == ast.dump(ast.parse(source))
    return changed, locations


def main():
    inputs = {
        ARM / 'train_runner.py': ORIGINAL_SHA,
        ARM / 'execution_protocol.json': PROTOCOL_SHA,
        PREVIOUS / 'resume_train_runner.py': PREVIOUS_RUNNER_SHA,
        PREVIOUS / 'resume_launch_arm.py': PREVIOUS_LAUNCHER_SHA,
        PREVIOUS / 'recovery_addendum.json': PREVIOUS_ADDENDUM_SHA,
    }
    assert all(digest(path) == expected for path, expected in inputs.items())
    runner_source = (PREVIOUS / 'resume_train_runner.py').read_text(encoding='utf-8')
    launcher_source = (PREVIOUS / 'resume_launch_arm.py').read_text(encoding='utf-8')
    runner_changes = [
        ('separate immutable anchor path',
         "    resume_path = out.parent / 'recovery_anchor/next_epoch.pt'\n",
         "    resume_path = out.parent / 'recovery30715_anchor/next_epoch.pt'\n"),
        ('exact accepted164 boundary',
         "    require(initial['resume_format'] == ROLLING_FORMAT and first_epoch == 29\n"
         "            and initial['actual_loop_epoch'] == 28 and initial['last_epoch'] == 28\n"
         "            and initial['optimizer_step_attempts'] == 46893, 'Unexpected recovery boundary')\n",
         "    require(initial['resume_format'] == ROLLING_FORMAT and first_epoch == 164\n"
         "            and initial['actual_loop_epoch'] == 163 and initial['last_epoch'] == 163\n"
         "            and initial['optimizer_step_attempts'] == 265188, 'Unexpected recovery boundary')\n"),
        ('second segment receipt scope',
         "        'recovery_scope': 'Explicitly authorized continuation from retained round29 on another matched RTX3090',\n",
         "        'recovery_scope': 'Explicitly authorized continuation from accepted round164 after instance expiry; original round29 recovery provenance retained',\n"),
        ('separate addendum file',
         "    recovery_path = here / 'recovery_addendum.json'\n",
         "    recovery_path = here / 'recovery30715_addendum.json'\n"),
        ('exact remaining scope and chosen anchor',
         "            and args.recovery['first_epoch_index'] == 29 and args.recovery['final_epoch_index'] == 319\n"
         "            and args.recovery['new_segment_evaluation_limit'] == 291\n"
         "            and args.recovery['new_segment_optimizer_update_limit'] == 470547\n",
         "            and args.recovery['first_epoch_index'] == 164 and args.recovery['final_epoch_index'] == 319\n"
         "            and args.recovery['new_segment_evaluation_limit'] == 156\n"
         "            and args.recovery['new_segment_optimizer_update_limit'] == 252252\n"
         f"            and args.recovery['checkpoint_sha256'] == '{ANCHOR_SHA}'\n"),
        ('immutable original29 addendum chain',
         "    for item in args.recovery['recovery_files']:\n",
         f"    require(args.recovery['prior_recovery_addendum_sha256'] == '{PREVIOUS_ADDENDUM_SHA}'\n"
         f"            and fair.sha(here / 'recovery_addendum.json') == '{PREVIOUS_ADDENDUM_SHA}',\n"
         "            'Original round29 recovery addendum chain differs')\n"
         "    for item in args.recovery['recovery_files']:\n"),
    ]
    runner, runner_locations = adapt(runner_source, runner_changes)
    # This filename occurs twice in ready(): bind the two occurrences as a single exact block.
    launcher_changes = [
        ('separate addendum file name in launcher',
         "    if sha(HERE / 'recovery_addendum.json') != recovery_expected:\n"
         "        raise RuntimeError('Recovery addendum differs')\n"
         "    recovery = read(HERE / 'recovery_addendum.json')\n",
         "    if sha(HERE / 'recovery30715_addendum.json') != recovery_expected:\n"
         "        raise RuntimeError('Recovery addendum differs')\n"
         "    recovery = read(HERE / 'recovery30715_addendum.json')\n"),
        ('launcher exact164 budget and original29 chain',
         "    if recovery['authorized'] is not True or recovery['first_epoch_index'] != 29:\n"
         "        raise RuntimeError('Unexpected recovery scope')\n",
         "    if (recovery['authorized'] is not True or recovery['arm'] != 'NO_MSCA'\n"
         "            or recovery['first_epoch_index'] != 164 or recovery['final_epoch_index'] != 319\n"
         "            or recovery['new_segment_evaluation_limit'] != 156\n"
         "            or recovery['new_segment_optimizer_update_limit'] != 252252\n"
         "            or recovery['retained_optimizer_update_limit'] != 517440\n"
         "            or recovery['original_protocol_sha256'] != expected\n"
         f"            or recovery['checkpoint_sha256'] != '{ANCHOR_SHA}'):\n"
         "        raise RuntimeError('Unexpected recovery scope')\n"
         f"    if (recovery['prior_recovery_addendum_sha256'] != '{PREVIOUS_ADDENDUM_SHA}'\n"
         f"            or sha(HERE / 'recovery_addendum.json') != '{PREVIOUS_ADDENDUM_SHA}'):\n"
         "        raise RuntimeError('Original round29 recovery addendum chain differs')\n"),
        ('runner self hash filename',
         "    if sha(HERE / 'resume_train_runner.py') != recovery['resume_runner_sha256']:\n",
         "    if sha(HERE / 'resume30715_train_runner.py') != recovery['resume_runner_sha256']:\n"),
        ('worker argv filename',
         "    command = [sys.executable, '-B', '-u', str(HERE / 'resume_train_runner.py'),\n",
         "    command = [sys.executable, '-B', '-u', str(HERE / 'resume30715_train_runner.py'),\n"),
        ('supervisor bound runner filename',
         "             'runner_sha256': sha(HERE / 'resume_train_runner.py'), 'command': command,\n",
         "             'runner_sha256': sha(HERE / 'resume30715_train_runner.py'), 'command': command,\n"),
    ]
    launcher, launcher_locations = adapt(launcher_source, launcher_changes)
    compile(runner, str(HERE / 'resume30715_train_runner.py'), 'exec')
    compile(launcher, str(HERE / 'resume30715_launch_arm.py'), 'exec')
    trees = [ast.parse((ARM / 'train_runner.py').read_bytes()), ast.parse(runner_source), ast.parse(runner)]
    train_functions = [function(tree, 'train_run') for tree in trees]
    loop_dumps = [ast.dump(epoch_loop(node)) for node in train_functions]
    assert loop_dumps[0] == loop_dumps[1] == loop_dumps[2]
    nested_checks = []
    for name in ('state', 'load_state', 'restore', 'commit'):
        dumps = [ast.dump(nested(node, name)) for node in train_functions]
        assert dumps[0] == dumps[1] == dumps[2], name
        nested_checks.append(name)
    helper_checks = []
    for name in ('require', 'sync_dir', 'write_atomic', 'copy_atomic', 'write_checkpoint',
                 'write_json', 'package_path', 'capture_rng'):
        dumps = [ast.dump(function(tree, name)) for tree in trees]
        assert dumps[0] == dumps[1] == dumps[2], name
        helper_checks.append(name)
    assert ast.dump(function(trees[1], 'restore_rng')) == ast.dump(function(trees[2], 'restore_rng'))
    helper_checks.append('restore_rng (previous scaffold versus new)')
    old_constants = [ast.dump(node) for node in trees[1].body if isinstance(node, ast.Assign)]
    new_constants = [ast.dump(node) for node in trees[2].body if isinstance(node, ast.Assign)]
    assert old_constants == new_constants
    assert (164 * 1617, 156 * 1617, 320 * 1617) == (265188, 252252, 517440)
    targets = [HERE / 'resume30715_train_runner.py', HERE / 'resume30715_launch_arm.py',
               HERE / 'resume30715_ast_equivalence.json']
    assert all(not path.exists() for path in targets), 'Never overwrite previous generated outputs'
    for path, source in zip(targets[:2], (runner, launcher)):
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(source)
    assert all(digest(path) == expected for path, expected in inputs.items())
    report = {
        'status': 'ROUND164_RESUME_SCAFFOLD_CREATED_AND_STATIC_AST_VERIFIED',
        'original_protocol_sha256': PROTOCOL_SHA, 'original_protocol_unchanged': True,
        'source_hashes': {str(path): expected for path, expected in inputs.items()},
        'builder_sha256': digest(Path(__file__)),
        'resume_runner_sha256': digest(targets[0]), 'resume_launcher_sha256': digest(targets[1]),
        'exact_checkpoint_sha256': ANCHOR_SHA, 'first_epoch_index': 164,
        'actual_loop_epoch_at_anchor': 163, 'last_epoch_at_anchor': 163,
        'retained_optimizer_updates': 265188, 'remaining_evaluations': 156,
        'remaining_optimizer_updates': 252252, 'final_epoch_index': 319,
        'total_optimizer_update_limit': 517440,
        'whole_epoch_loop_AST_identical_original_r29_and_new': True,
        'identical_nested_functions': nested_checks, 'identical_helpers': helper_checks,
        'top_level_constants_AST_identical': True,
        'full_module_restores_source_AST_after_reversing_only_listed_changes': True,
        'runner_changes': runner_locations, 'launcher_changes': launcher_locations,
        'anchor_relative_path': 'recovery30715_anchor/next_epoch.pt',
        'new_addendum_relative_path': 'recovery30715_addendum.json',
        'required_addendum_chain': {'prior_recovery_addendum_sha256': PREVIOUS_ADDENDUM_SHA,
                                    'original_addendum_relative_path': 'recovery_addendum.json'},
        'target_boot_and_gpu_bound_by_new_addendum_only': True,
        'all_recovery_files_individual_sha_guard_preserved': True,
        'standard_launcher_paths': ['supervisor.json', 'training.log', 'launch.lock', 'supervisor.log'],
        'launcher_deletes_or_overwrites_old_standard_records': False,
        'notes': [
            'Old standard launcher records must be archived and hashes bound before a new launch.',
            'Restore all accepted164 manifest source bytes, including cumulative gzip LR history; preserve discarded partial165 separately.',
            'Native backup probe identity and run basename must remain unchanged.',
            'No new addendum, remote staging, process, model or evaluation is created by this builder.',
        ],
        'model_forwards': 0, 'optimizer_updates': 0, 'evaluations': 0, 'remote_connections': 0,
    }
    with targets[2].open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': report['status'], 'builder_sha256': report['builder_sha256'],
                      'resume_runner_sha256': report['resume_runner_sha256'],
                      'resume_launcher_sha256': report['resume_launcher_sha256'],
                      'ast_report_sha256': digest(targets[2])}))


if __name__ == '__main__':
    main()
