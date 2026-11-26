"""Prepare isolated FRESH driver bytes; no model imports, network or training."""
from pathlib import Path
import ast
import hashlib
import json
import tarfile

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent.parent
SOURCE=PARENT/'模块消融'/'NO_MPS'
ARCHIVE=PARENT/'secondary_frozen_inputs.tar.gz'

def sha(b):return hashlib.sha256(b).hexdigest()
def replace_once(text,old,new):
    assert text.count(old)==1,(old[:100],text.count(old))
    return text.replace(old,new,1)
def save(name,text):
    ast.parse(text)
    destination=HERE/name
    destination.write_text(text,encoding='utf-8')
    return {'path':name,'sha256':sha(destination.read_bytes())}

def main():
    assert sha(ARCHIVE.read_bytes())=='e127bf14dacd999ca3794c90dc16d778f34f22ff135c227d599ad43ac243c36d'
    with tarfile.open(ARCHIVE) as t: original=t.extractfile('fair_runner.py').read().decode()
    tree=ast.parse(original)
    tree.body=[n for n in tree.body if not (isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in ('preflight','train_run')) and not (isinstance(n,ast.If) and ast.unparse(n.test)=="__name__ == '__main__'")]
    helper=ast.unparse(tree)+'\n'
    helper=replace_once(helper, """f'Unregistered transform: {op['type']}'""", '''f"Unregistered transform: {op['type']}"''')
    helper=replace_once(helper,"os.environ['CUDA_VISIBLE_DEVICES'] = '0'","os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')")
    helper=replace_once(helper,"def build(arm):\n","def build(arm):\n    arm = arm.removesuffix('_FRESH')\n")
    runner=(SOURCE/'train_runner.py').read_text(encoding='utf-8')
    runner=replace_once(runner,"ROLLING_FORMAT = 'fair_runner_post_policy_epoch_v1'","ROLLING_FORMAT = 'fresh_pair_complete_post_policy_epoch_v1'")
    runner=replace_once(runner,"'last_epoch': last_epoch, 'actual_loop_epoch': loop_epoch, 'optimizer_step_attempts': attempts,","'last_epoch': last_epoch, 'actual_loop_epoch': loop_epoch, 'optimizer_step_attempts': attempts,\n                'rng_state': capture_rng(fair, train, val), 'ema_decay_attribute': ema.decay,")
    runner=replace_once(runner,"def commit(loop_epoch, evaluated):","def commit(loop_epoch, evaluated, evaluated_is_next_state):")
    runner=replace_once(runner,"rolling_sha = fair.sha(rolling)","rolling_sha = fair.sha(rolling)\n            evaluated['evaluated_state_is_next_state'] = evaluated_is_next_state\n            evaluated['evaluated_checkpoint_sha256'] = rolling_sha if evaluated_is_next_state else fair.sha(out / 'evaluated_epoch.pt')\n            evaluated['next_state_ema_tensor_sha256'] = fair.tensor_hash(ema.module.state_dict())\n            require(not evaluated_is_next_state or evaluated['next_state_ema_tensor_sha256'] == evaluated['ema_tensor_sha256'], 'Equivalent state roles have different EMA tensors')\n            fair.appendj(out / 'epochs.jsonl', evaluated)")
    runner=replace_once(runner,"gate.protect(loop_epoch, rolling_sha, best, stage_one, evaluated)","gate.protect(loop_epoch, rolling_sha, best, stage_one, evaluated)")
    runner=replace_once(runner,"            previous_best = best\n","            evaluated_is_next_state = not (epoch >= STAGE_TWO and metrics['AP'] <= best_ap)\n            if not evaluated_is_next_state:\n                write_checkpoint(torch, out / 'evaluated_epoch.pt', {**state(epoch), 'checkpoint_role': 'evaluated_pre_policy', 'completed_epochs': epoch + 1})\n            previous_best = best\n")
    runner=replace_once(runner,"            fair.appendj(out / 'epochs.jsonl', row)\n            rolling_sha = commit(epoch, row)","            rolling_sha = commit(epoch, row, evaluated_is_next_state)")
    runner=replace_once(runner,"plan['arm'] in ('NO_MPS', 'NO_MSCA')","plan['arm'] in ('BASE_FRESH', 'FULL_FRESH')")
    runner=replace_once(runner,"'Unexpected authorization or experiment budget')","'Unexpected authorization or experiment budget')\n    require(plan['optimizer_update_limit_per_arm'] == 517440 and plan['evaluation_limit_per_arm'] == 320\n            and plan['hard_timeout_seconds'] == 604800 and plan['seed'] == 0 and plan['batch_size'] == 4\n            and plan['precision'] == 'FP32' and plan['automatic_training_retry'] is False, 'Budget or recipe changed')")
    runner=replace_once(runner,"    gate = backup.EpochGate(args.output)","    prior_frozen = fair.readj(here / 'frozen_parent/execution_protocol.json')\n    for item in prior_frozen['frozen_files']:\n        path = (here / 'frozen_parent' / item['path']).resolve(strict=True)\n        require(path.is_relative_to(here / 'frozen_parent') and fair.sha(path) == item['sha256'], 'Original frozen247 file changed')\n    require(len(prior_frozen['frozen_files']) == 247, 'Original source scope changed')\n    gate = backup.EpochGate(args.output)")
    protocol=(SOURCE/'backup_protocol.py').read_text(encoding='utf-8')
    protocol=replace_once(protocol,"SCHEMA = 'module_ablation_backup_v1'","SCHEMA = 'fresh_pair_complete_state_backup_v1'")
    protocol=replace_once(protocol,"        'next_epoch.pt': 'run/next_epoch.pt',","        'next_epoch.pt': 'run/next_epoch.pt',\n        'evaluated_epoch.pt': 'run/next_epoch.pt' if manifest['evaluated_state_is_next_state'] else 'run/evaluated_epoch.pt',")
    protocol=replace_once(protocol,"{'next_epoch', 'optimizer_step_attempts', 'rolling_sha256', 'best', 'stage_one'}","{'next_epoch', 'optimizer_step_attempts', 'rolling_sha256', 'best', 'stage_one', 'evaluated_state_is_next_state', 'evaluated_sha256', 'evaluated_ema_tensor_sha256', 'next_ema_tensor_sha256'}")
    protocol=replace_once(protocol,"    sources = expected_sources(manifest)\n","    if epoch >= 0:\n        require(type(manifest['evaluated_state_is_next_state']) is bool and type(manifest['evaluated_sha256']) is str\n                and HEX_SHA.fullmatch(manifest['evaluated_sha256']), 'Invalid evaluated state role binding')\n        for key in ['evaluated_ema_tensor_sha256', 'next_ema_tensor_sha256']:\n            require(type(manifest[key]) is str and HEX_SHA.fullmatch(manifest[key]), 'Invalid EMA role hash')\n        require(not manifest['evaluated_state_is_next_state'] or manifest['evaluated_ema_tensor_sha256'] == manifest['next_ema_tensor_sha256'], 'Equivalent roles have different EMA hashes')\n    sources = expected_sources(manifest)\n")
    protocol=replace_once(protocol,"require(by_name['next_epoch.pt'] == manifest['rolling_sha256'], 'Rolling checkpoint hash mismatch')","require(by_name['next_epoch.pt'] == manifest['rolling_sha256'], 'Rolling checkpoint hash mismatch')\n        require(by_name['evaluated_epoch.pt'] == manifest['evaluated_sha256'], 'Evaluated checkpoint hash mismatch')\n        require(not manifest['evaluated_state_is_next_state'] or manifest['evaluated_sha256'] == manifest['rolling_sha256'], 'Equivalent state roles differ')")
    protocol=replace_once(protocol,"    checked_row('epoch.metrics.json', epoch, 'latest.predictions.json', 'latest.evaluation_arrays.npz')","    evaluated = checked_row('epoch.metrics.json', epoch, 'latest.predictions.json', 'latest.evaluation_arrays.npz')\n    require(evaluated['evaluated_checkpoint_sha256'] == by_name['evaluated_epoch.pt']\n            and evaluated['evaluated_state_is_next_state'] is manifest['evaluated_state_is_next_state']\n            and evaluated['ema_tensor_sha256'] == manifest['evaluated_ema_tensor_sha256']\n            and evaluated['next_state_ema_tensor_sha256'] == manifest['next_ema_tensor_sha256'], 'Metrics not bound to actual evaluated state')")
    protocol=replace_once(protocol,"'rolling_sha256': rolling_sha, 'best': best, 'stage_one': stage_one,","'rolling_sha256': rolling_sha, 'best': best, 'stage_one': stage_one,\n            'evaluated_state_is_next_state': evaluated['evaluated_state_is_next_state'],\n            'evaluated_sha256': evaluated['evaluated_checkpoint_sha256'],\n            'evaluated_ema_tensor_sha256': evaluated['ema_tensor_sha256'],\n            'next_ema_tensor_sha256': evaluated['next_state_ema_tensor_sha256'],")
    launch=(SOURCE/'launch_arm.py').read_text(encoding='utf-8')
    launch=replace_once(launch,"plan['arm'] not in ('NO_MPS', 'NO_MSCA')","plan['arm'] not in ('BASE_FRESH', 'FULL_FRESH')")
    launch=replace_once(launch,"dict(os.environ, CUDA_VISIBLE_DEVICES='0'","dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan['cuda_visible_device'])")
    monitor=(SOURCE/'backup_monitor.py').read_text(encoding='utf-8')
    monitor=replace_once(monitor,'import shutil\n','import shutil\nimport subprocess\n')
    monitor=replace_once(monitor,'POLL_SECONDS = 2\n','POLL_SECONDS = 2\nCHECKPOINT_PYTHON = None\n\ndef payload_validation(stage, manifest_data, manifest, execute=False):\n    report_path = stage / "payload_validation.json"\n    validator = Path(__file__).resolve().with_name("checkpoint_validator.py")\n    if execute and not report_path.exists():\n        require(CHECKPOINT_PYTHON is not None, "CPU payload validator executable missing")\n        command = [str(CHECKPOINT_PYTHON), "-B", str(validator), "--stage", str(stage),\n                   "--protocol", str(LOCAL_ROOT / "execution_protocol.json")]\n        result = subprocess.run(command, capture_output=True, text=True, timeout=300,\n                                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4"))\n        require(result.returncode == 0, "CPU checkpoint payload validation failed: " + result.stderr[-2000:])\n    report = protocol.decode(protocol.local_metadata(report_path))\n    require(report["status"] == "CHECKPOINT_PAYLOADS_VERIFIED_CPU_ONLY"\n            and report["manifest_sha256"] == protocol.digest(manifest_data)\n            and report["protocol_sha256"] == protocol.PROTOCOL\n            and report["validator_sha256"] == protocol.file_sha(validator)\n            and report["epoch"] == manifest["epoch"]\n            and report["model_forwards"] == report["evaluator_runs"] == report["optimizer_updates"] == 0,\n            "Saved CPU payload validation does not bind this stage")\n\n')
    monitor=replace_once(monitor,"        protocol.check_metric_files(manifest, lambda name: protocol.local_metadata(stage / name))\n        return expected_ack","        protocol.check_metric_files(manifest, lambda name: protocol.local_metadata(stage / name))\n        payload_validation(stage, data, manifest)\n        return expected_ack")
    monitor=replace_once(monitor,"        protocol.check_metric_files(manifest, lambda name: protocol.local_metadata(temporary / name))\n        ack = protocol.ack_bytes(data, manifest)","        protocol.check_metric_files(manifest, lambda name: protocol.local_metadata(temporary / name))\n        payload_validation(temporary, data, manifest, execute=True)\n        ack = protocol.ack_bytes(data, manifest)")
    monitor=replace_once(monitor,'def main():\n','def main(password_provider=None):\n    global CHECKPOINT_PYTHON\n')
    monitor=replace_once(monitor,"parser.add_argument('--port', type=int, default=30219, choices=[30219, 30848])","parser.add_argument('--port', type=int, required=True, choices=[30715, 30048])")
    monitor=replace_once(monitor,"parser.add_argument('--remote-output', required=True, type=PurePosixPath)","parser.add_argument('--remote-root', required=True, type=PurePosixPath)\n    parser.add_argument('--local-root', required=True, type=Path)\n    parser.add_argument('--checkpoint-python', required=True, type=Path)")
    monitor=replace_once(monitor,"parser.add_argument('--backup-root', type=Path, default=LOCAL_ROOT / 'backups')","parser.add_argument('--backup-root', type=Path)")
    monitor=replace_once(monitor,"    out = args.remote_output","    out = args.remote_root / 'run'\n    CHECKPOINT_PYTHON = args.checkpoint_python.resolve(strict=True)\n    args.local_root = args.local_root.resolve(strict=True)\n    if args.backup_root is None: args.backup_root = args.local_root / 'backups'")
    monitor=replace_once(monitor,'store = BackupStore(args.backup_root, LOCAL_ROOT)','store = BackupStore(args.backup_root, args.local_root)')
    monitor=replace_once(monitor,"require(store.root != LOCAL_ROOT, 'Choose a dedicated local backup subdirectory')","require(store.root != args.local_root, 'Choose a dedicated local backup subdirectory')")
    monitor=replace_once(monitor,"    require(sys.stdin.isatty(), 'Start the backup monitor in an interactive terminal')\n    with warnings.catch_warnings():\n        warnings.simplefilter('error', getpass.GetPassWarning)\n        password = getpass.getpass('SSH password for the pinned FULL host: ')","    if password_provider is None:\n        require(sys.stdin.isatty(), 'Use a credential provider or an interactive terminal')\n        with warnings.catch_warnings():\n            warnings.simplefilter('error', getpass.GetPassWarning)\n            password = getpass.getpass('SSH password for the pinned FRESH host: ')\n    else:\n        password = password_provider(args.host, args.port, args.user)\n        require(isinstance(password, str) and bool(password), 'Credential provider returned no password')")
    monitor=replace_once(monitor,'MAX_BACKUP_BYTES = 100_000_000_000',"MAX_BACKUP_BYTES = {'BASE_FRESH':45 * 1024**3, 'FULL_FRESH':75 * 1024**3}[protocol.ARM]")
    monitor=replace_once(monitor,'CHECKPOINT_PYTHON = None\n','CHECKPOINT_PYTHON = None\nCHECKPOINT_TORCH_SITE = None\n')
    monitor=replace_once(monitor,'        result = subprocess.run(command, capture_output=True, text=True, timeout=300,','        if CHECKPOINT_TORCH_SITE is not None:\n            command.extend([\'--torch-site\', str(CHECKPOINT_TORCH_SITE)])\n        result = subprocess.run(command, capture_output=True, text=True, timeout=300,')
    monitor=replace_once(monitor,'    global CHECKPOINT_PYTHON\n','    global CHECKPOINT_PYTHON, CHECKPOINT_TORCH_SITE\n')
    monitor=replace_once(monitor,"    parser.add_argument('--checkpoint-python', required=True, type=Path)","    parser.add_argument('--checkpoint-python', required=True, type=Path)\n    parser.add_argument('--torch-site', type=Path)")
    monitor=replace_once(monitor,'    CHECKPOINT_PYTHON = args.checkpoint_python.resolve(strict=True)','    CHECKPOINT_PYTHON = args.checkpoint_python.resolve(strict=True)\n    CHECKPOINT_TORCH_SITE = args.torch_site.resolve(strict=True) if args.torch_site else None')
    outputs=[save('fair_helpers.py',helper),save('train_runner.py',runner),save('backup_protocol.py',protocol),save('launch_arm.py',launch),save('backup_monitor.py',monitor)]
    old=ast.parse((SOURCE/'train_runner.py').read_text(encoding='utf-8'));new=ast.parse(runner)
    def named_function(t,name):return next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==name)
    def training_batch_loop(t):return next(n for n in ast.walk(named_function(t,'train_run')) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and ast.unparse(n.target)=='(step, (samples, targets))')
    assert ast.dump(training_batch_loop(old),include_attributes=False)==ast.dump(training_batch_loop(new),include_attributes=False)
    for function in ['load_state','restore']:
        assert ast.dump(named_function(old,function),include_attributes=False)==ast.dump(named_function(new,function),include_attributes=False)
    for function in ['prepare_epoch','evaluate','controls','seeds','worker_init']:
        assert ast.dump(named_function(ast.parse(original),function),include_attributes=False)==ast.dump(named_function(ast.parse(helper),function),include_attributes=False)
    report={
        'status':'STATIC_ONLY_DRIVER_PREPARED_NOT_RUN','original_archive_sha256':sha(ARCHIVE.read_bytes()),
        'source_runner':str(SOURCE/'train_runner.py'),'source_runner_sha256':sha((SOURCE/'train_runner.py').read_bytes()),
        'outputs':outputs,'training_batch_loop_AST_identical':True,'native_load_and_restore_AST_identical':True,
        'data_prepare_evaluation_controls_seeds_worker_init_AST_identical':True,
        'changes':['Original BASE/FULL build and archived initial tensors, no ablation adapter','Complete RNG in evaluated/best/terminal and post-policy checkpoints','Distinct evaluated/next roles; same physical checkpoint only when no epoch-end rollback occurs','Every evaluated state included in closed backup manifest with metric binding','Fresh arm IDs, immutable247 parent inputs, strict bounded scope','GPU visibility supplied by per-arm launch protocol'],
        'limitations':['No GPU imports, preflight, inference, evaluation or training executed locally','Staging must bind actual path relocation, runtime and separate user authorization','Per-round full-state payload verification remains a local-backup-client responsibility; a manifest hash alone is not tensor validation'],
        'new_training_runs':0,'model_forwards':0,'evaluator_runs':0,
    }
    (HERE/'driver_static_semantic_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'outputs':outputs}))

if __name__=='__main__':main()
