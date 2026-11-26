"""Build the one-shot protocol from downloaded, verified preparation evidence."""
from pathlib import Path
from datetime import datetime, timezone
import copy, hashlib, json, re

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / 'remote_evidence'
PAPER = HERE.parent

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()

target = HERE / 'execution_protocol.json'
assert not target.exists(), 'An execution protocol already exists; do not overwrite it.'
proposal = read(PAPER / '公平对照方案_20261103' / 'protocol.json')
assert sha(PAPER / '公平对照方案_20261103' / 'protocol.json') == 'da959eaf28c6371f1f696f2b0ce0000baf990cebaa28494798731aa138d710d5'
assert read(EVIDENCE / 'preparation_queue_v2.json')['status'] == 'PREFLIGHTS_COMPLETED_FORMAL_TRAINING_NOT_STARTED'
cache = read(EVIDENCE / 'cache_verification.json')
assert cache['status'] == 'ALL_INPUT_BYTES_VERIFIED' and cache['files'] == 7021
assert cache['archive_sha256'] == 'fdde141711867fa1bc2de3a1e6342cca257eb056e2a1700795eb04068aa90266'
assert cache['manifest_sha256'] == sha(HERE / 'dataset_bundle_manifest.json')
server_data = read(EVIDENCE / 'input_verification.json')
assert server_data['status'] == 'INPUTS_VERIFIED'
for split in ['train','val']:
    assert not server_data['splits'][split]['mismatches'] and server_data['splits'][split]['annotation_matches']
    assert server_data['splits'][split]['annotation_actual_sha256'] == proposal['data'][split]['annotation_sha256']
source = read(EVIDENCE / 'source_manifest.json')
assert len(source) == 229
assert source == read(PAPER / '本地CPU核验_20261103' / 'source_manifest.json')
for row in source:
    assert sha(PAPER / '本地CPU核验_20261103' / row['snapshot']) == row['sha256']

preflights = {}
configs = {}
for arm in ['BASE', 'FULL']:
    report = read(EVIDENCE / 'preflight' / f'{arm}.json')
    assert report['status'] == 'PREFLIGHT_PASSED_NO_OPTIMIZER_UPDATE'
    assert report['model_optimizer_steps'] == 0 and report['loss_and_gradients_finite']
    assert report['runner_sha256'] == sha(HERE / 'fair_runner.py')
    assert report['config_sha256'] == sha(EVIDENCE / 'execution_configs' / f'{arm}.yml')
    assert report['environment_sha256'] == sha(EVIDENCE / 'environment.current.json')
    assert report['initial_file_sha256'] == sha(EVIDENCE / 'preflight' / f'{arm}.initial.pt')
    assert report['first100_batches_sha256'] == sha(EVIDENCE / 'preflight' / f'{arm}.first100.json')
    assert len(read(EVIDENCE / 'preflight' / f'{arm}.first100.json')) == 100
    preflights[arm] = report
    cfg = read(EVIDENCE / 'execution_configs' / f'{arm}.yml')
    expected = read(PAPER / '公平对照方案_20261103' / ('baseline.proposed.yml' if arm == 'BASE' else 'full.proposed.yml'))
    for split in ['train', 'val']:
        expected[f'{split}_dataloader']['dataset']['ann_file'] = cache['cache_root'] + f'/{split}/annotations/{split}.json'
        expected[f'{split}_dataloader']['dataset']['img_folder'] = cache['cache_root'] + f'/{split}/images'
    expected['output_dir'] = '/25085400120/mpsma_fair_20261103_v1/runs/' + arm
    assert cfg == expected, f'Unexpected configuration difference: {arm}'
    configs[arm] = cfg
assert preflights['BASE']['first100_batches_sha256'] == preflights['FULL']['first100_batches_sha256']
assert preflights['BASE']['initial_executed_lrs'] == preflights['FULL']['initial_executed_lrs']
common = read(EVIDENCE / 'preflight/common_initialization.json')
assert common['shared_tensors'] == len(common['keys']) and common['shared_tensors'] > 0
for key in common['keys']:
    assert key.startswith(('encoder.', 'decoder.')) or re.fullmatch(r'backbone\.stages\.\d+\.(downsample\..+|blocks\.\d+\.layers\..+)', key)

env = read(EVIDENCE / 'environment.current.json')
assert env['torch'] == '2.5.1+cu124' and env['cuda'] == '12.4'
assert 'GPU-85eca9af-10f3-4279-d72a-05ff1ecdcd3a' in env['gpu']
assert sha(EVIDENCE / 'source_loader.py') == sha(PAPER / '本地CPU核验_20261103' / 'evaluate_cpu.py')

frozen = [{'path': r['snapshot'], 'sha256': r['sha256']} for r in source]
for rel in ['source_manifest.json', 'source_loader.py', 'cache_verification.json', 'input_verification.json', 'environment.current.json',
            'execution_configs/BASE.yml', 'execution_configs/FULL.yml', 'preflight/common_initialization.json',
            'preflight/BASE.json', 'preflight/FULL.json', 'preflight/BASE.initial.pt', 'preflight/FULL.initial.pt',
            'preflight/BASE.first100.json', 'preflight/FULL.first100.json']:
    frozen.append({'path': rel, 'sha256': sha(EVIDENCE / rel)})
for rel in ['fair_runner.py', 'supervise.py', 'dataset_bundle_manifest.json', 'authorization.json']:
    frozen.append({'path': rel, 'sha256': sha(HERE / rel)})

protocol = {
    'status': 'FROZEN_AUTHORIZED_READY_FOR_ONE_SHOT_LAUNCH',
    'created_local_utc': datetime.now(timezone.utc).isoformat(),
    'authorized': True, 'arms': ['BASE', 'FULL'], 'epochs': 320,
    'authorization': read(HERE / 'authorization.json'),
    'question': proposal['question'], 'scope_limit': proposal['scope_limit'],
    'budget': {**proposal['budget'], 'hard_timeout_seconds_per_arm': 604800,
               'timeout_basis':'Pre-execution failsafe of seven days per arm; 320 epochs and 517440 attempted updates remain the actual fixed budget. The earlier unexecuted 48-hour placeholder was not a measured runtime estimate.'},
    'evaluation': proposal['evaluation'], 'acceptance': proposal['acceptance'],
    'proposed_protocol_sha256': sha(PAPER / '公平对照方案_20261103' / 'protocol.json'),
    'runtime': env, 'cache': cache, 'original_server_dataset_verification':server_data,
    'randomness': {
        'master_seed': 0, 'python_hash_seed': 0,
        'training_sampler_and_worker_generator_seed_each_epoch': '1000003 * actual_loop_epoch',
        'validation_worker_generator_initial_seed': 1000000007,
        'worker_python_numpy_seed': 'torch.initial_seed() modulo 2**32',
        'reset_after_model_construction': True,
        'common_initialization': common,
        'first100_preflight_batches_identical': True,
        'first100_batches_sha256': preflights['BASE']['first100_batches_sha256'],
        'actual_training_first100_must_match_preflight': True,
        'tf32_matmul_and_cudnn': False,
        'cudnn_benchmark': False, 'cudnn_deterministic': True,
        'torch_deterministic_algorithms': 'enabled with warn_only=True; no bitwise reproducibility claim',
        'torch_intraop_threads': 4, 'torch_interop_threads': 1,
        'training_and_validation_workers': 4,
    },
    'preserved_native_behavior': [
        'Frozen network, criterion, augmentation, EMA and COCO evaluator source; no neural-module edits.',
        'Original FP32 forward/loss/zero_grad/backward/clip(0.1)/optimizer/EMA/post-update-LR ordering.',
        'Initial LinearWarmup constructor scaling (base LR divided by 500) is preserved. Flatcosine is applied after each update.',
        'lr_gamma=1.0 makes the post-warmup LR constant; the scheduler name does not imply an actual cosine decrease.',
        'At actual epoch 296 restore stage-one best model/optimizer/EMA/warmup/last_epoch; preserve later non-best-AP rollback to that state.',
        'Native EMA decay_fn closes the constructor decay; later self.decay assignments do not modify that closure.',
        'Evaluate and bind all 12 metrics before any rollback; actual loop epoch and attempted updates are recorded separately from restored last_epoch.',
        'Shared parameters receive identical initial tensors; arm-specific stem/aggregation/gates are not copied.',
    ],
    'driver_differences_for_equal_control_and_evidence': [
        'Explicit sampler/worker generators and input fingerprints; both arms use the same controlled data schedule.',
        'Unrelated eager imports, FLOP profiling, plotting and TensorBoard output are omitted from both arms.',
        'Fail on nonfinite loss/gradient and preserve failure; no automatic retries or replacement runs.',
        'Keep actual terminal and selected-best checkpoints, matching predictions and COCO evaluation arrays.',
        'No periodic restart checkpoints are used; evaluation outputs and executed group LRs are recorded.',
    ],
    'preflights': preflights,
    'preparation_history': {'attempt1':'Mosaic registration failure before any optimizer update; preserved in preparation_attempt1.',
                            'attempt2':'Register unchanged Mosaic module; preflight receipts include code/config/environment hashes.'},
    'frozen_files': frozen,
    'time_note': 'Remote UTC labels and local receipt timestamps are kept separately; durations and hard timeouts use the remote monotonic clock. The server clock is not adjusted.',
    'no_submission_readiness_claim': 'Await both complete fixed-budget runs and evidence audit before revising comparative claims.'
}
with target.open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(protocol, stream, indent=2, ensure_ascii=False, allow_nan=False)
    stream.write('\n')
print(json.dumps({'file': str(target), 'sha256': sha(target), 'frozen_files': len(frozen),
                  'shared_tensors': common['shared_tensors'], 'status': protocol['status']}))
