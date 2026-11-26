"""Derive the ablation runtime from the already validated training and backup code."""

import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
PROTECTED = PARENT / '断点续训_20261105/FULL_30848_20261106'
OUTPUT = HERE / 'runtime'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Reference code changed near ' + old[:80])
    return source.replace(old, new, 1)


def function_source(source, node):
    first = min([node.lineno] + [item.lineno for item in node.decorator_list]) - 1
    return '\n'.join(source.splitlines()[first:node.end_lineno]) + '\n'


def helpers():
    source = (PARENT / 'fair_runner.py').read_text(encoding='utf-8')
    stop = source.index('def shared_key(')
    tree = ast.parse(source)
    evaluate = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'evaluate')
    result = source[:stop] + function_source(source, evaluate)
    result = replace_once(result, 'model=cfg.model; criterion=cfg.criterion; post=cfg.postprocessor',
                          "model=cfg.model\n    from ablation_model import apply\n"
                          "    assert cfgdata['ablation']['arm'] == arm\n"
                          "    model.ablation_initialization = apply(model, arm, ROOT/'reference')\n"
                          "    criterion=cfg.criterion; post=cfg.postprocessor")
    result = replace_once(result, "{'BASE':3737611,'FULL':7540942}[arm]",
                          "{'NO_MPS':7535617,'NO_MSCA':3699662}[arm]")
    return result


TRAIN_PREFIX = '''def train_run(fair, args, bindings, gate):
    torch = fair.torch
    out = args.output
    arm = args.arm
    first_epoch = 0
    out.mkdir(parents=True, exist_ok=False)
    sync_dir(out.parent)
    attempts = 0
    last_epoch = -1
    best_ap = -math.inf
    best = stage_one = None
    origin = fair.readj(out.parent / 'inputs/preflight.json')['initialization']
    receipt = {
        **bindings, 'config_sha256': bindings['configuration_sha256'],
        'status': 'INITIALIZING', 'pid': os.getpid(), 'started_utc': fair.now(),
        'actual_loop_epoch': -1, 'completed_epochs': 0, 'next_epoch': 0,
        'first_epoch_in_this_segment': 0, 'optimizer_step_attempts': 0,
        'initialization_origin': origin, 'runner_sha256': fair.sha(Path(__file__)),
        'torch_version': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
    }
'''


TRAIN_MAIN = '''
def main():
    import fair_helpers as fair
    import backup_protocol as backup
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    plan = fair.readj(here / 'execution_protocol.json')
    require(fair.sha(here / 'execution_protocol.json') == args.protocol_sha256 == backup.PROTOCOL,
            'Execution protocol differs')
    require(plan['authorized'] is True and plan['epochs'] == 320 and plan['arm'] in ('NO_MPS', 'NO_MSCA'),
            'Unexpected authorization or experiment budget')
    args.arm = plan['arm']
    args.output = here / 'run'
    for item in plan['frozen_files']:
        path = (here / item['path']).resolve(strict=True)
        require(path.is_relative_to(here) and fair.sha(path) == item['sha256'], 'Frozen input changed')
    preflight = fair.readj(here / 'inputs/preflight.json')
    require(preflight['status'] == 'PREFLIGHT_PASSED_NO_OPTIMIZER_UPDATE' and preflight['optimizer_updates'] == 0,
            'Preflight did not pass')
    environment = fair.readj(here / 'environment.current.json')
    require(str(fair.torch.__version__) == environment['torch'] and fair.torch.version.cuda == environment['cuda']
            and fair.torch.backends.cudnn.version() == environment['cudnn'], 'Training runtime changed')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    require(gpu[0] == plan['gpu_uuid'] and int(gpu[1]) >= 20000, 'GPU differs or is occupied')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                      '--format=csv,noheader,nounits'], text=True).strip()
    require(not active, 'GPU has another compute process')
    cache = fair.readj(here / 'cache_verification.json')
    cache_root = Path(cache['cache_root']).resolve(strict=True)
    for item in fair.readj(here / 'dataset_bundle_manifest.json')['files']:
        path = (cache_root / item['path']).resolve(strict=True)
        require(path.is_relative_to(cache_root) and fair.sha(path) == item['sha256'], 'Dataset changed')
    gate = backup.EpochGate(args.output)
    gate.require_probe()
    bindings = {'arm': args.arm, 'protocol_sha256': backup.PROTOCOL,
                'configuration_sha256': backup.CONFIG,
                'environment_sha256': fair.sha(here / 'environment.current.json'),
                'initialization_sha256': preflight['initial_file_sha256'],
                'parent_protocol_sha256': plan['parent_protocol_sha256'], 'weight_source': 'ema.module'}
    train_run(fair, args, bindings, gate)


if __name__ == '__main__':
    main()
'''


def trainer():
    source = (PROTECTED / 'full_resume_runner.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    common = ('require', 'sync_dir', 'write_atomic', 'copy_atomic', 'write_checkpoint',
              'write_json', 'package_path', 'capture_rng')
    prefix = ('import argparse, gzip, json, math, os, shutil, subprocess, time, traceback\n'
              'from pathlib import Path\n'
              'EPOCHS = 320\nBATCHES = 1617\nSTAGE_TWO = 296\n'
              "ROLLING_FORMAT = 'fair_runner_post_policy_epoch_v1'\n"
              "BEST_FILES = ('best.pt', 'best.metrics.json', 'best.predictions.json', 'best.evaluation_arrays.npz')\n\n")
    prefix += '\n'.join(function_source(source, functions[name]) for name in common)
    body = function_source(source, functions['train_run'])
    body = TRAIN_PREFIX + body[body.index("    fair.savej(out / 'receipt.json', receipt)"):]
    body = replace_once(body, "        copy_package(best_source, package_path(out, best))\n"
                              "        if stage_one != best:\n"
                              "            copy_package(stage_one_source, package_path(out, stage_one))\n", '')
    body = replace_once(body, '        model.cuda(); criterion.cuda(); post.cuda()',
                         "        initial = torch.load(out.parent / 'inputs/initial.pt', map_location='cpu', weights_only=True)\n"
                         "        model.load_state_dict(initial, strict=True)\n"
                         "        require(fair.tensor_hash(model.state_dict()) == fair.readj(out.parent / 'inputs/preflight.json')['initial_tensor_sha256'],\n"
                         "                'Training initial tensors differ')\n"
                         "        del initial\n"
                         "        model.cuda(); criterion.cuda(); post.cuda()")
    body = replace_once(body, "'historical_origin': origin", "'initialization_origin': origin")
    first = body.index('        load_state(initial)\n')
    last = body.index("        lrpath = out / 'executed_learning_rates.csv.gz'", first)
    body = body[:first] + "        fair.seeds(0)\n        expected = fair.readj(out.parent / 'inputs/first100.json')\n\n" + body[last:]
    insertion = "                for step, (samples, targets) in enumerate(train):\n"
    body = replace_once(body, insertion, insertion +
        "                    if epoch == 0 and step < 100:\n"
        "                        observed = {'step': step, 'sha256': fair.batch_hash(samples, targets),\n"
        "                                    'image_ids': [int(target['image_id']) for target in targets]}\n"
        "                        require(observed == expected[step], 'Actual training inputs differ')\n"
        "                        fair.appendj(out / 'actual_first100.jsonl', observed)\n")
    body = replace_once(body, '            if previous_best != best and previous_best != stage_one:',
                         '            if previous_best is not None and previous_best != best and previous_best != stage_one:')
    original_loop = next(node for node in ast.walk(functions['train_run'])
                         if isinstance(node, ast.For) and ast.unparse(node.target) == '(step, (samples, targets))')
    generated = ast.parse(body)
    new_loop = next(node for node in ast.walk(generated)
                   if isinstance(node, ast.For) and ast.unparse(node.target) == '(step, (samples, targets))')
    assert ast.dump(ast.Module(body=original_loop.body, type_ignores=[])) == ast.dump(
        ast.Module(body=new_loop.body[1:], type_ignores=[])), 'Per-batch training math changed'
    return prefix + '\n' + body + TRAIN_MAIN


def backup_protocol():
    source = (PROTECTED / 'full_backup_protocol.py').read_text(encoding='utf-8')
    first = source.index("SCHEMA = 'full_resume_backup_v1'")
    last = source.index('TERMINAL_FILES =', first)
    block = '''_PLAN_PATH = Path(__file__).resolve().with_name('execution_protocol.json')
_PLAN = json.loads(_PLAN_PATH.read_text(encoding='utf-8'))
SCHEMA = 'module_ablation_backup_v1'
PROTOCOL = hashlib.sha256(_PLAN_PATH.read_bytes()).hexdigest()
CONFIG = _PLAN['configuration_sha256']
ARM = _PLAN['arm']
FINAL_EPOCH = 319
BATCHES = 1617
METRICS = frozenset(('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl'))
PROBE_FILES = _PLAN['probe_files']
BEST_FILES = ('best.pt', 'best.metrics.json', 'best.predictions.json', 'best.evaluation_arrays.npz')
'''
    source = source[:first] + block + source[last:]
    source = replace_once(source, "'arm': 'FULL'", "'arm': ARM")
    source = replace_once(source, "'historical_checkpoint_sha256': ANCHOR_CHECKPOINT",
                          "'initialization_checkpoint_sha256': PROBE_FILES['initial.pt']")
    source = replace_once(source, "receipt['arm'] == 'FULL'", "receipt['arm'] == ARM")
    source = replace_once(source, '    if epoch > 0:\n', '    if epoch >= 0:\n')
    source = replace_once(source, "        for name in ('epochs.jsonl', 'executed_learning_rates.csv.gz'):",
                          "        sources['actual_first100.jsonl'] = 'run/actual_first100.jsonl'\n"
                          "        for name in ('epochs.jsonl', 'executed_learning_rates.csv.gz'):")
    source = replace_once(source,
        "        checked_row('best.metrics.json', 0, 'best.predictions.json', 'best.evaluation_arrays.npz', 'best.pt')\n",
        "        report = decode(read('preflight.json'))\n"
        "        require(report['arm'] == ARM and report['status'] == 'PREFLIGHT_PASSED_NO_OPTIMIZER_UPDATE'\n"
        "                and report['optimizer_updates'] == 0\n"
        "                and report['initial_file_sha256'] == by_name['initial.pt'], 'Unbound preflight transport probe')\n")
    return source


def monitor():
    source = (PROTECTED / 'full_backup_monitor.py').read_text(encoding='utf-8')
    source = replace_once(source, 'import full_backup_protocol as protocol', 'import backup_protocol as protocol')
    source = replace_once(source, "receipt['arm'] == 'FULL'", "receipt['arm'] == protocol.ARM")
    source = replace_once(source, 'default=30848, choices=[30848]', 'default=30219, choices=[30219, 30848]')
    source = replace_once(source, 'MIN_FREE_BYTES = 2 * 1024**3', 'MIN_FREE_BYTES = 32 * 1024**3')
    return source


def main():
    OUTPUT.mkdir(exist_ok=False)
    files = {'fair_helpers.py': helpers(), 'train_runner.py': trainer(),
             'backup_protocol.py': backup_protocol(), 'backup_monitor.py': monitor()}
    for name, content in files.items():
        ast.parse(content)
        (OUTPUT / name).write_text(content, encoding='utf-8')
    report = {'status': 'GENERATED_AND_AST_CHECKED', 'per_batch_math_matches_protected_full_runner': True,
              'first100_training_binding_added': True, 'backups_include_round_one_learning_rates': True,
              'local_backup_free_reserve_gib': 32, 'generated_files': sorted(files)}
    (HERE / 'runtime_generation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
