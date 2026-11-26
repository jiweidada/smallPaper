"""Create a separately bound resume driver; keep all frozen originals untouched."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import textwrap

here = Path(__file__).resolve().parent
arm = here.parent
source_path = arm / 'train_runner.py'
source = source_path.read_text(encoding='utf-8')
original = source

def change(old, new):
    global source
    assert source.count(old) == 1, old[:100]
    source = source.replace(old, new)

change('    first_epoch = 0\n    out.mkdir(parents=True, exist_ok=False)\n    sync_dir(out.parent)\n    attempts = 0\n    last_epoch = -1\n    best_ap = -math.inf\n    best = stage_one = None\n',
       "    resume_path = out.parent / 'recovery_anchor/next_epoch.pt'\n"
       "    require(fair.sha(resume_path) == args.recovery['checkpoint_sha256'], 'Recovery anchor changed')\n"
       "    initial = torch.load(resume_path, map_location='cpu', weights_only=True)\n"
       "    first_epoch = initial['next_epoch']\n"
       "    require(initial['resume_format'] == ROLLING_FORMAT and first_epoch == 29\n"
       "            and initial['actual_loop_epoch'] == 28 and initial['last_epoch'] == 28\n"
       "            and initial['optimizer_step_attempts'] == 46893, 'Unexpected recovery boundary')\n"
       "    require(all(initial[k] == v for k, v in bindings.items()), 'Recovery bindings differ')\n"
       "    require(out.is_dir(), 'Missing restored output directory')\n"
       "    attempts = initial['optimizer_step_attempts']\n"
       "    last_epoch = initial['last_epoch']\n"
       "    best, stage_one = initial['best'], initial['stage_one']\n"
       "    best_ap = fair.readj(package_path(out, best) / 'best.metrics.json')['metrics']['AP']\n")
change("        'actual_loop_epoch': -1, 'completed_epochs': 0, 'next_epoch': 0,\n        'first_epoch_in_this_segment': 0, 'optimizer_step_attempts': 0,\n",
       "        'actual_loop_epoch': first_epoch - 1, 'completed_epochs': first_epoch, 'next_epoch': first_epoch,\n"
       "        'first_epoch_in_this_segment': first_epoch, 'optimizer_step_attempts': attempts,\n"
       "        'resumed_from_checkpoint_sha256': args.recovery['checkpoint_sha256'],\n"
       "        'recovery_addendum_sha256': args.recovery_sha256,\n"
       "        'recovery_scope': 'Explicitly authorized continuation from retained round29 on another matched RTX3090',\n")
change("        (out / 'selected').mkdir()\n", "        require((out / 'selected').is_dir(), 'Missing restored selected packages')\n")
# Preserve the resume checkpoint in `initial`; use a separate name for the frozen initialization proof.
change("        initial = torch.load(out.parent / 'inputs/initial.pt', map_location='cpu', weights_only=True)\n        model.load_state_dict(initial, strict=True)\n",
       "        frozen_initial = torch.load(out.parent / 'inputs/initial.pt', map_location='cpu', weights_only=True)\n"
       "        model.load_state_dict(frozen_initial, strict=True)\n")
change('        del initial\n', '        del frozen_initial\n')
change('        fair.seeds(0)\n        expected = fair.readj(out.parent / \'inputs/first100.json\')\n',
       "        load_state(initial)\n"
       "        ema.decay = initial['ema_decay_attribute']\n"
       "        require(fair.tensor_hash(model.state_dict()) == fair.tensor_hash(initial['model']),\n"
       "                'Restored raw model differs')\n"
       "        require(fair.tensor_hash(ema.module.state_dict()) == fair.tensor_hash(initial['ema']['module']),\n"
       "                'Restored EMA differs')\n"
       "        restore_rng(fair, initial, train, val)\n"
       "        receipt.update(status='RUNNING', phase='epoch_complete',\n"
       "                       rolling_checkpoint_sha256=args.recovery['checkpoint_sha256'],\n"
       "                       best_epoch=best['epoch'], best_AP=best_ap, updated_utc=fair.now())\n"
       "        fair.savej(out / 'receipt.json', receipt)\n"
       "        del initial\n"
       "        expected = fair.readj(out.parent / 'inputs/first100.json')\n")
change("        with gzip.open(lrpath, 'wt', compresslevel=1) as lrlog:\n            lrlog.write('attempt,actual_epoch,batch,' + ','.join(f'group{i}' for i in range(len(base_lrs))) + '\\n')\n",
       "        require(lrpath.is_file(), 'Missing retained cumulative LR history')\n")

helper = textwrap.dedent('''
def restore_rng(fair, saved, train, val):
    torch = fair.torch
    require(not train.persistent_workers and not val.persistent_workers
            and isinstance(val.sampler, torch.utils.data.SequentialSampler), 'Unexpected loader workers/sampler')
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
    require(current['python'] == rng['python'] and current['numpy'] == rng['numpy'], 'Python/NumPy RNG differs')
    for key in ['torch_cpu', 'train_loader', 'train_sampler', 'validation_loader']:
        require(torch.equal(current[key], rng[key]), 'Restored RNG differs: ' + key)
    require(all(torch.equal(a, b) for a, b in zip(current['torch_cuda_all'], rng['torch_cuda_all'])),
            'Restored CUDA RNG differs')

''')
change('def train_run(fair, args, bindings, gate):\n', helper + 'def train_run(fair, args, bindings, gate):\n')
change("    parser.add_argument('--protocol-sha256', required=True)\n",
       "    parser.add_argument('--protocol-sha256', required=True)\n"
       "    parser.add_argument('--recovery-sha256', required=True)\n")
change("    plan = fair.readj(here / 'execution_protocol.json')\n",
       "    plan = fair.readj(here / 'execution_protocol.json')\n"
       "    recovery_path = here / 'recovery_addendum.json'\n"
       "    require(fair.sha(recovery_path) == args.recovery_sha256, 'Recovery addendum differs')\n"
       "    args.recovery = fair.readj(recovery_path)\n"
       "    require(args.recovery['authorized'] is True and args.recovery['arm'] == 'NO_MSCA'\n"
       "            and args.recovery['first_epoch_index'] == 29 and args.recovery['final_epoch_index'] == 319\n"
       "            and args.recovery['new_segment_evaluation_limit'] == 291\n"
       "            and args.recovery['new_segment_optimizer_update_limit'] == 470547\n"
       "            and args.recovery['retained_optimizer_update_limit'] == 517440\n"
       "            and args.recovery['original_protocol_sha256'] == args.protocol_sha256,\n"
       "            'Unexpected explicitly authorized recovery scope')\n"
       "    for item in args.recovery['recovery_files']:\n"
       "        path = (here / item['path']).resolve(strict=True)\n"
       "        require(path.is_relative_to(here) and fair.sha(path) == item['sha256'], 'Recovery input changed')\n"
       "    require(fair.sha(Path(__file__)) == args.recovery['resume_runner_sha256'], 'Resume runner changed')\n"
       "    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == args.recovery['target_boot_id'],\n"
       "            'Recovery target boot changed')\n")
change("    require(gpu[0] == plan['gpu_uuid'] and int(gpu[1]) >= 20000, 'GPU differs or is occupied')\n",
       "    require(gpu[0] == args.recovery['target_gpu_uuid'] and int(gpu[1]) >= 20000,\n"
       "            'Recovery GPU differs or is occupied')\n")

tree_old, tree_new = ast.parse(original), ast.parse(source)
old_fn = next(n for n in tree_old.body if isinstance(n, ast.FunctionDef) and n.name == 'train_run')
new_fn = next(n for n in tree_new.body if isinstance(n, ast.FunctionDef) and n.name == 'train_run')
old_loop = next(n for n in ast.walk(old_fn) if isinstance(n, ast.For) and ast.unparse(n.target) == 'epoch')
new_loop = next(n for n in ast.walk(new_fn) if isinstance(n, ast.For) and ast.unparse(n.target) == 'epoch')
assert ast.dump(old_loop, include_attributes=False) == ast.dump(new_loop, include_attributes=False)
nested = ['state', 'load_state', 'restore', 'commit']
for name in nested:
    first = next(n for n in ast.walk(old_fn) if isinstance(n, ast.FunctionDef) and n.name == name)
    second = next(n for n in ast.walk(new_fn) if isinstance(n, ast.FunctionDef) and n.name == name)
    assert ast.dump(first, include_attributes=False) == ast.dump(second, include_attributes=False), name
output = here / 'resume_train_runner.py'
with output.open('x', encoding='utf-8', newline='\n') as handle:
    handle.write(source)
report = {'status': 'RESUME_SCAFFOLD_CREATED_FROZEN_EPOCH_LOOP_AST_IDENTICAL',
    'original_sha256': sha256(source_path.read_bytes()).hexdigest(),
    'resume_sha256': sha256(output.read_bytes()).hexdigest(),
    'identical_nested_functions': nested, 'identical_entire_epoch_loop': True,
    'new_forwards': 0, 'optimizer_updates': 0, 'evaluations': 0,
    'changes': ['restore round29 raw/EMA/optimizer/warmup/RNG and counters',
                'append existing cumulative logs; preserve original accepted backup identity',
                'require separately hash-bound authorized target/recovery inputs',
                'retain original frozen protocol and all epoch/update/evaluation mathematics']}
(here / 'resume_ast_equivalence.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report))
