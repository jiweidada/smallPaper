"""Adapt the frozen runner for continuation and verify its training loop."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVER = HERE.parent / 'driver'
ORIGINAL_SHA = '75f6826ecc7b3416422e82ae12ce34f60564854ec44171fde2ac0cca3c783ed5'

RNG_RESTORE = """def restore_rng(fair, saved, train, val):
    torch = fair.torch
    require(not train.persistent_workers and not val.persistent_workers
            and isinstance(val.sampler, torch.utils.data.SequentialSampler), 'Unexpected loader workers or sampler')
    rng = saved['rng_state']
    require(set(rng) == {'python', 'numpy', 'torch_cpu', 'torch_cuda_all', 'train_loader',
                        'train_sampler', 'validation_loader'}, 'Incomplete saved RNG state')
    fair.random.setstate(rng['python'])
    name, keys, position, has_gauss, cached = rng['numpy']
    fair.np.random.set_state((name, fair.np.asarray(keys, dtype=fair.np.uint32), position, has_gauss, cached))
    torch.set_rng_state(rng['torch_cpu'])
    require(len(rng['torch_cuda_all']) == torch.cuda.device_count() == 1, 'CUDA device count changed')
    torch.cuda.set_rng_state_all(rng['torch_cuda_all'])
    train.generator.set_state(rng['train_loader'])
    train.sampler.generator.set_state(rng['train_sampler'])
    val.generator.set_state(rng['validation_loader'])
    current = capture_rng(fair, train, val)
    require(current['python'] == rng['python'] and current['numpy'] == rng['numpy'], 'Python or NumPy RNG differs')
    for key in ('torch_cpu', 'train_loader', 'train_sampler', 'validation_loader'):
        require(torch.equal(current[key], rng[key]), 'Restored RNG differs: ' + key)
    require(all(torch.equal(a, b) for a, b in zip(current['torch_cuda_all'], rng['torch_cuda_all'])),
            'Restored CUDA RNG differs')

"""

INITIAL = """    first_epoch = args.recovery['accepted_rounds']
    anchor = out / 'next_epoch.pt'
    saved = torch.load(anchor, map_location='cpu', weights_only=True)
    require(saved['resume_format'] == ROLLING_FORMAT and saved['next_epoch'] == first_epoch
            and saved['actual_loop_epoch'] == saved['last_epoch'] == first_epoch - 1
            and saved['optimizer_step_attempts'] == first_epoch * BATCHES, 'Unexpected recovery boundary')
    require(all(saved[k] == v for k, v in bindings.items()), 'Recovery checkpoint bindings differ')
    require(saved['best'] == args.recovery['best'] and saved['stage_one'] == args.recovery['stage_one'],
            'Recovery checkpoint selection differs')
    attempts = saved['optimizer_step_attempts']
    last_epoch = saved['last_epoch']
    best, stage_one = saved['best'], saved['stage_one']
    best_ap = fair.readj(package_path(out, best) / 'best.metrics.json')['metrics']['AP']
"""

RESTORE = """        load_state(saved)
        ema.decay = saved['ema_decay_attribute']
        require(ema.updates == saved['ema']['updates'] and ema.decay == saved['ema_decay_attribute'],
                'Restored EMA state differs')
        restore_rng(fair, saved, train, val)
        receipt.update(status='RUNNING', phase='epoch_complete', best_epoch=best['epoch'], best_AP=best_ap,
                       latest_metrics=saved['latest_evaluated_row']['metrics'], updated_utc=fair.now())
        fair.savej(out / 'receipt.json', receipt)
        del saved
        expected = fair.readj(out.parent / 'inputs/first100.json')
        lrpath = out / 'executed_learning_rates.csv.gz'
"""

OBSOLETE_CLEANUP = """            if previous_best is not None and previous_best != best and previous_best != stage_one:
                obsolete = package_path(out, previous_best)
                for name in BEST_FILES:
                    (obsolete / name).unlink()
                obsolete.rmdir()
                sync_dir(obsolete.parent)
"""


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def function(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def epoch_loop(tree):
    return next(n for n in function(tree, 'train_run').body
                if isinstance(n, ast.Try)).body


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    original = (DRIVER / 'train_runner.py').read_bytes()
    assert digest(original) == ORIGINAL_SHA
    source = original.decode().replace('\r\n', '\n')
    runner = replace_once(source, 'def train_run(', RNG_RESTORE + 'def train_run(')
    runner = replace_once(runner, """    first_epoch = 0
    out.mkdir(parents=True, exist_ok=False)
    sync_dir(out.parent)
    attempts = 0
    last_epoch = -1
    best_ap = -math.inf
    best = stage_one = None
""", INITIAL)
    runner = replace_once(runner, """        'actual_loop_epoch': -1, 'completed_epochs': 0, 'next_epoch': 0,
        'first_epoch_in_this_segment': 0, 'optimizer_step_attempts': 0,
""", """        'actual_loop_epoch': last_epoch, 'completed_epochs': first_epoch, 'next_epoch': first_epoch,
        'first_epoch_in_this_segment': first_epoch, 'optimizer_step_attempts': attempts,
        'rolling_checkpoint_sha256': args.recovery['checkpoint_sha256'],
        'resumed_from_checkpoint_sha256': args.recovery['checkpoint_sha256'],
        'recovery_authorization_sha256': args.authorization_sha256,
        'original_segment_root': args.recovery['original_root'],
""")
    runner = replace_once(runner, "        (out / 'selected').mkdir()\n", '')
    runner = replace_once(runner, """        fair.seeds(0)
        expected = fair.readj(out.parent / 'inputs/first100.json')

        lrpath = out / 'executed_learning_rates.csv.gz'
        with gzip.open(lrpath, 'wt', compresslevel=1) as lrlog:
            lrlog.write('attempt,actual_epoch,batch,' + ','.join(f'group{i}' for i in range(len(base_lrs))) + '\\n')
""", RESTORE)
    runner = replace_once(runner, '            previous_best = best\n', '')
    runner = replace_once(runner, OBSOLETE_CLEANUP, '')
    runner = replace_once(runner, "    args = parser.parse_args()\n",
                          """    parser.add_argument('--authorization-sha256', required=True)
    args = parser.parse_args()
    from continuation_guard import verify_continuation
    args.recovery = verify_continuation(here, args.protocol_sha256, args.authorization_sha256)
""")

    old = ast.parse(source.replace('            previous_best = best\n', '').replace(OBSOLETE_CLEANUP, ''))
    new = ast.parse(runner)
    old_body, new_body = (epoch_loop(tree) for tree in (old, new))
    old_end, new_end = ([ast.dump(n) for n in nodes if isinstance(n, ast.For)] for nodes in (old_body, new_body))
    assert len(old_end) == 2 and old_end == new_end
    same = ('state', 'load_state', 'restore', 'commit', 'capture_rng', 'write_checkpoint',
            'copy_atomic', 'write_atomic', 'package_path')
    for name in same:
        assert ast.dump(function(old, name)) == ast.dump(function(new, name)), name
    for file in (HERE / 'continuation_guard.py', HERE / 'continuation_launch.py'):
        ast.parse(file.read_bytes(), filename=str(file), feature_version=(3, 10))
    ast.parse(runner, feature_version=(3, 10))
    path = HERE / 'continuation_train.py'
    data = runner.encode()
    with path.open('xb') as stream:
        stream.write(data)
    report = {'original_runner_sha256': ORIGINAL_SHA, 'continuation_runner_sha256': digest(data),
              'epoch_loop_equal_excluding_obsolete_best_cleanup': True,
              'identical_training_state_functions': same,
              'restored_components': ['model', 'EMA including decay attribute', 'optimizer', 'criterion',
                                      'postprocessor', 'warmup', 'Python RNG', 'NumPy RNG', 'torch CPU and CUDA RNG',
                                      'training loader and sampler RNG', 'validation loader RNG'],
              'model_forwards': 0, 'evaluations': 0, 'optimizer_updates': 0}
    with (HERE / 'continuation_driver_audit.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
