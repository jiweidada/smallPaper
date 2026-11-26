"""Compare frozen removal inputs with FULL; no model construction or execution."""
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE = HERE.parent
PARENT_SHA = '8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'
NON_EXECUTABLE_OMISSIONS = {'source_snapshot/AGENTS.md', 'source_snapshot/requirements.txt'}
IDENTICAL_FUNCTIONS = ['seeds', 'worker_init', 'controls', 'prepare_epoch',
                       'batch_hash', 'tensor_hash', 'evaluate']
IDENTICAL_MODULE_HELPERS = ['train_runner.py', 'fair_helpers.py', 'ablation_model.py',
                            'source_loader.py', 'backup_protocol.py']
EXPECTED_PARAMETERS = {'NO_MPS': 7535617, 'NO_MSCA': 3699662}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def functions(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    return {node.name: ast.dump(node, include_attributes=False)
            for node in tree.body if isinstance(node, ast.FunctionDef)}


def normalized_config(config):
    result = deepcopy(config)
    result.pop('output_dir')
    result.pop('ablation', None)
    for split in ['train', 'val']:
        dataset = result[f'{split}_dataloader']['dataset']
        dataset['img_folder'] = f'FROZEN_DATA/{split}/images'
        dataset['ann_file'] = f'FROZEN_DATA/{split}/annotations/{split}.json'
    return result


def audit():
    import torch
    torch.set_num_threads(1)
    parent_path = REMOTE / 'execution_protocol.json'
    assert digest(parent_path) == PARENT_SHA
    parent = read(parent_path)
    frozen_parent = {item['path']: item['sha256'] for item in parent['frozen_files']}
    parent_sources = {key: value for key, value in frozen_parent.items() if key.startswith('source_snapshot/')}
    full_config_path = REMOTE / 'remote_evidence/execution_configs/FULL.yml'
    assert digest(full_config_path) == frozen_parent['execution_configs/FULL.yml']
    full_config = read(full_config_path)
    parent_functions = functions(REMOTE / 'fair_runner.py')
    assert digest(REMOTE / 'fair_runner.py') == frozen_parent['fair_runner.py']
    reference_first = read(REMOTE / 'remote_evidence/preflight/FULL.first100.json')
    assert len(reference_first) == 100
    assert rows(REMOTE / 'parallel_evidence/FULL_actual_first100.jsonl') == reference_first
    assert rows(REMOTE / 'parallel_evidence/BASE_actual_first100.jsonl') == reference_first
    result = {'status': 'FROZEN_REMOVAL_CONDITIONS_AND_INITIAL_TENSORS_VERIFIED',
              'observed_local_utc': datetime.now(timezone.utc).isoformat(),
              'parent_protocol_sha256': PARENT_SHA,
              'full_config_sha256': digest(full_config_path), 'arms': {},
              'model_forwards': 0, 'evaluator_runs': 0, 'training_updates': 0,
              'scope': 'Actual frozen inputs, helper semantics, archived first-100 batches and initial tensor equality. This is not a completed outcome or submission-readiness certificate.',
              'limitations': ['FULL has documented continuations, while the removals started from round zero.',
                             'Shared recipe and initial tensors do not imply uninterrupted or identical full stochastic trajectories.',
                             'Conditional effects include each component\'s capacity and apply with other FULL settings present.',
                             'BASE is not a joint-removal arm; no factorial interaction or statistical significance is established.']}
    helper_digests = {}
    for arm in ['NO_MPS', 'NO_MSCA']:
        root = HERE / arm
        plan_path = root / 'execution_protocol.json'
        plan = read(plan_path)
        frozen = {item['path']: item['sha256'] for item in plan['frozen_files']}
        assert plan['authorized'] is True and plan['arm'] == arm
        assert plan['parent_protocol_sha256'] == PARENT_SHA
        assert (plan['epochs'], plan['seed'], plan['batch_size'], plan['precision']) == (320, 0, 4, 'FP32')
        assert plan['optimizer_update_limit_per_arm'] == 517440 and plan['evaluation_limit_per_arm'] == 320
        assert plan['authorized_additional_arms'] == ['NO_MPS', 'NO_MSCA']
        config_path = root / 'execution_configs' / f'{arm}.yml'
        assert digest(config_path) == plan['configuration_sha256'] == frozen[f'execution_configs/{arm}.yml']
        config = read(config_path)
        assert normalized_config(config) == normalized_config(full_config)
        assert config['ablation'] == plan['architecture']
        assert plan['architecture']['reference_config_sha256'] == digest(full_config_path)
        assert digest(root / 'dataset_bundle_manifest.json') == parent['cache']['manifest_sha256']
        source_files = {key: value for key, value in frozen.items() if key.startswith('source_snapshot/')}
        assert parent_sources.keys() - source_files.keys() == NON_EXECUTABLE_OMISSIONS
        assert not source_files.keys() - parent_sources.keys()
        assert len(source_files) == 227
        for name, expected in source_files.items():
            assert expected == parent_sources[name] and digest(root / name) == expected, name
        for name in IDENTICAL_MODULE_HELPERS:
            value = digest(root / name)
            assert value == frozen[name]
            if name in helper_digests:
                assert value == helper_digests[name]
            helper_digests[name] = value
        helper_functions = functions(root / 'fair_helpers.py')
        assert all(helper_functions[name] == parent_functions[name] for name in IDENTICAL_FUNCTIONS)
        environment_path = root / 'environment.current.json'
        assert digest(environment_path) == frozen['environment.current.json']
        environment = read(environment_path)
        for key in ['python', 'torch', 'cuda', 'cudnn', 'torch_build', 'packages']:
            assert environment[key] == parent['runtime'][key], key
        preflight_path = root / 'inputs/preflight.json'
        assert digest(preflight_path) == frozen['inputs/preflight.json']
        preflight = read(preflight_path)
        assert preflight['status'] == 'PREFLIGHT_PASSED_NO_OPTIMIZER_UPDATE'
        assert preflight['optimizer_updates'] == 0 and preflight['finite_loss_and_gradients']
        assert preflight['initialization'] == plan['initialization']
        assert preflight['trainable_parameters'] == EXPECTED_PARAMETERS[arm]
        assert read(root / 'inputs/first100.json') == reference_first
        state_path = root / 'inputs/initial.pt'
        assert digest(state_path) == frozen['inputs/initial.pt'] == plan['probe_files']['initial.pt']
        state = torch.load(state_path, map_location='cpu', weights_only=True)
        reference_states = {}
        for name in ['FULL', 'BASE']:
            path = root / 'reference' / f'{name}.initial.pt'
            expected = frozen_parent[f'preflight/{name}.initial.pt']
            assert digest(path) == expected == plan['initialization']['reference_initial_sha256'][name]
            reference_states[name] = torch.load(path, map_location='cpu', weights_only=True)
        full, base = reference_states['FULL'], reference_states['BASE']
        assert len(state) == plan['initialization']['retained_tensor_count']
        if arm == 'NO_MPS':
            stem_keys = {key for key in base if key.startswith('backbone.stem.')}
            assert stem_keys == set(plan['initialization']['stem_keys_from_base'])
            assert set(state) == {key for key in full if not key.startswith('backbone.stem.')} | stem_keys
            for key, value in state.items():
                expected = base[key] if key in stem_keys else full[key]
                assert value.dtype == expected.dtype and torch.equal(value, expected), key
            phenotype = {'stem': 'Original StemBlock(3,16,16,use_lab=True), with its ReLU modules replaced by SiLU',
                         'base_stem_tensors': len(stem_keys), 'remaining_FULL_tensors': len(state) - len(stem_keys)}
        else:
            gates = ['backbone.stages.2.blocks.0.aggregation.1',
                     'backbone.stages.2.blocks.1.aggregation.1',
                     'backbone.stages.3.blocks.0.aggregation.1']
            assert plan['initialization']['identity_gate_paths'] == gates
            expected_keys = {key for key in full if not any(key.startswith(gate + '.') for gate in gates)}
            assert set(state) == expected_keys
            assert all(value.dtype == full[key].dtype and torch.equal(value, full[key]) for key, value in state.items())
            phenotype = {'identity_gate_paths': gates, 'FULL_tensors_retained_exactly': len(state),
                         'note': 'State equality alone cannot prove replacement of parameter-free Lite gates; their three declared paths are separately bound to the frozen apply adapter and genuine preflight.'}
        state_record = read(HERE / 'current_execution_state.json')[arm]
        first_folder = Path(state_record['backup_root']) / 'epochs/epoch0000'
        spec = importlib.util.spec_from_file_location(f'{arm}_shared_proof_protocol', root / 'backup_protocol.py')
        backup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backup)
        payload = (first_folder / 'manifest.json').read_bytes()
        manifest = backup.parse_manifest(payload, Path(config['output_dir']).parent.name, 0)
        assert (first_folder / 'ack.json').read_bytes() == backup.ack_bytes(payload, manifest)
        assert (first_folder / 'accepted.json').read_bytes() == backup.accepted_bytes(payload, manifest)
        actual_path = first_folder / 'actual_first100.jsonl'
        actual_record = next(item for item in manifest['files'] if item['name'] == actual_path.name)
        assert digest(actual_path) == actual_record['sha256'] and rows(actual_path) == reference_first
        result['arms'][arm] = {
            'protocol_sha256': digest(plan_path), 'configuration_sha256': digest(config_path),
            'configuration_comparison': 'Exactly FULL after normalizing only data/output paths and removing the explicit ablation declaration.',
            'matching_source_files': len(source_files), 'non_executable_parent_files_omitted': sorted(NON_EXECUTABLE_OMISSIONS),
            'identical_parent_function_ASTs': IDENTICAL_FUNCTIONS,
            'matching_runtime_fields': ['python', 'torch', 'cuda', 'cudnn', 'torch_build', 'packages'],
            'matching_dependency_versions': len(environment['packages']),
            'dataset_manifest_sha256': parent['cache']['manifest_sha256'],
            'initial_file_sha256': digest(state_path), 'initial_tensors_directly_compared': len(state),
            'trainable_parameters': preflight['trainable_parameters'], 'phenotype': phenotype,
            'actual_first100_sha256': digest(actual_path),
            'actual_first100_manifest_sha256': sha256(payload).hexdigest(),
            'primary_endpoint': plan['primary_endpoint'], 'secondary_endpoint': plan['secondary_endpoint']}
    result['identical_module_runtime_helpers'] = helper_digests
    result['analysis_script_sha256'] = digest(Path(__file__))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'],
                      'arms': {name: {'source_files': arm['matching_source_files'],
                                      'initial_tensors': arm['initial_tensors_directly_compared'],
                                      'runtime_dependencies': arm['matching_dependency_versions']}
                               for name, arm in report['arms'].items()},
                      'model_forwards': 0, 'evaluator_runs': 0}))
