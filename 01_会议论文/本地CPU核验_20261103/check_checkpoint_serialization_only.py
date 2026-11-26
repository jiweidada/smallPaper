"""Round-trip a retained checkpoint and fresh test RNG metadata on CPU only.

This is a storage/serialization check, not model inference, evaluation,
training, checkpoint recovery, or a new experiment result. The fake CUDA RNG
is a CPU generator state and must never be interpreted as the historical state.
"""
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import ast
import copy
import csv
import gzip
import json
import os
import random
import shutil
import sys
import tempfile

import numpy as np
import torch

CPU_ROOT = Path(__file__).resolve().parent
PAPER = CPU_ROOT.parent
REMOTE = PAPER / '远程公平对照_20261103'
PREP = REMOTE / '公平对照重做准备_20261105'
sys.path.insert(0, str(PREP))
import epoch_backup_guard as guard
import epoch_evidence_stager as stager

SOURCE = REMOTE / 'remote_evidence/interrupted_BASE_20261105/runs/BASE'
AUDIT = REMOTE / 'interrupted_base_saved_file_audit_20261105.json'
CANDIDATE = PREP / 'guarded_fair_runner_candidate_v2.py.txt'
REPORT = PREP / 'cpu_checkpoint_serialization_smoketest_v2.json'
SCRATCH_PARENT = PREP / 'cpu_serialization_temporary'
RUN = 'synthetic_cpu_serialization_only'
CANDIDATE_SHA = '8a4b588bfe79e1b323d5749c136d65aa595cc79b68e15a25ee983787e7da74cf'


def digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_hash(state):
    h = sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        h.update(json.dumps([key, str(tensor.dtype), list(tensor.shape)]).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def equal_tree(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, Mapping):
        return isinstance(b, Mapping) and set(a) == set(b) and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


class FrozenState:
    """Expose an already-loaded state dictionary without constructing a model."""
    def __init__(self, payload):
        self.payload = payload

    def state_dict(self):
        return self.payload


@contextmanager
def checked_scratch(parent):
    parent = Path(parent).resolve(strict=True)
    raw = Path(tempfile.mkdtemp(prefix='checked_', dir=str(parent)))
    verified = raw.resolve(strict=True)
    assert verified.is_relative_to(parent) and verified != parent and verified.name.startswith('checked_')
    try:
        yield verified
    finally:
        target = raw.resolve(strict=True)
        assert target == verified and target.is_relative_to(parent) and target != parent
        assert not raw.is_symlink() and not getattr(raw, 'is_junction', lambda: False)()
        shutil.rmtree(raw)


def candidate_functions():
    assert digest(CANDIDATE) == CANDIDATE_SHA
    tree = ast.parse(CANDIDATE.read_text(encoding='utf-8'))
    train = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'train_run')
    assert isinstance(train.body[0], ast.Expr) and ast.unparse(train.body[0]) == '_evidence_require_authorization()'
    trial = next(x for x in train.body if isinstance(x, ast.Try))
    names = {'_evidence_capture_rng', 'state'}
    functions = [copy.deepcopy(x) for x in trial.body if isinstance(x, ast.FunctionDef) and x.name in names]
    assert {x.name for x in functions} == names
    # Only these two inert state-capture functions are compiled; the candidate's
    # imports, training loop, preflight and entry point are never executed.
    for item in functions:
        assert not any(isinstance(x, (ast.Import, ast.ImportFrom, ast.For, ast.While, ast.With, ast.Try))
                       for x in ast.walk(item))
    return ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))


def main():
    assert not REPORT.exists() and not SCRATCH_PARENT.exists()
    assert torch.__version__ == '2.6.0+cpu' and not torch.cuda.is_available()
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    assert audit['status'] == 'BEST_CHECKPOINT_BOUND_LAST_ARRAYS_ONLY'
    checkpoint_path = SOURCE / 'best.pt'
    checkpoint_sha = digest(checkpoint_path)
    assert checkpoint_sha == audit['best_checkpoint_sha256']
    source_predictions = SOURCE / 'best.predictions.json'
    source_arrays = SOURCE / 'best.evaluation_arrays.npz'
    best = audit['saved_file_results']['best']
    assert digest(source_predictions) == best['prediction_sha256']
    assert digest(source_arrays) == best['evaluation_arrays_sha256']
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    epoch = checkpoint['actual_loop_epoch']
    assert epoch == 213 and checkpoint['arm'] == 'BASE'
    assert checkpoint['optimizer_step_attempts'] == (epoch + 1) * guard.BATCHES
    assert tensor_hash(checkpoint['ema']['module']) == audit['best_ema_tensor_sha256']
    learning_rates = []
    saw_next_epoch = False
    with gzip.open(SOURCE / 'executed_learning_rates.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            actual_epoch = int(row['actual_epoch'])
            if actual_epoch > epoch:
                saw_next_epoch = True
                break  # The interrupted gzip lacks a final trailer; do not read to EOF.
            if actual_epoch == epoch:
                rates = [float(row[key]) for key in row if key.startswith('group')]
                learning_rates.append((int(row['attempt']), epoch, int(row['batch']), rates))
    assert saw_next_epoch and len(learning_rates) == guard.BATCHES
    # These are fresh generators in this test process. No historical RNG state
    # was retained in the original checkpoint, and no CUDA generator is queried.
    train = SimpleNamespace(generator=torch.Generator(device='cpu').manual_seed(11),
                            sampler=SimpleNamespace(generator=torch.Generator(device='cpu').manual_seed(12)))
    val = SimpleNamespace(generator=torch.Generator(device='cpu').manual_seed(13))
    fake_cuda_state = torch.Generator(device='cpu').manual_seed(14).get_state()
    fake_protocol = '0' * 64
    receipt = {'protocol_sha256': fake_protocol, 'config_sha256': fake_protocol,
               'environment_sha256': fake_protocol}
    scope = {'random': random, 'np': np, 'torch': torch, 'train': train, 'val': val,
             'model': FrozenState(checkpoint['model']), 'ema': FrozenState(checkpoint['ema']),
             'optimizer': FrozenState(checkpoint['optimizer']), 'criterion': FrozenState(checkpoint['criterion']),
             'post': FrozenState(checkpoint['postprocessor']), 'warmup': FrozenState(checkpoint['lr_warmup_scheduler']),
             'last_epoch': checkpoint['last_epoch'], 'attempts': checkpoint['optimizer_step_attempts'],
             'receipt': receipt, 'arm': 'BASE'}
    exec(compile(candidate_functions(), '<isolated_state_capture_only>', 'exec'), scope)
    metrics = best['metrics']
    identities = {key: fake_protocol for key in stager.IDENTITY_FIELDS}
    assert set(metrics) == stager.METRICS
    SCRATCH_PARENT.mkdir(exist_ok=False)
    try:
        with checked_scratch(SCRATCH_PARENT) as temp:
            parent = SCRATCH_PARENT.resolve(strict=True)
            assert temp.is_relative_to(parent) and temp != parent and temp.name.startswith('checked_')
            out = temp / 'runs/BASE'
            out.mkdir(parents=True)
            shutil.copyfile(source_predictions, out / 'latest.predictions.json')
            shutil.copyfile(source_arrays, out / 'latest.evaluation_arrays.npz')
            control_before = {'last_epoch': int(checkpoint['last_epoch']), 'ema_updates': int(checkpoint['ema']['updates']),
                              'ema_decay': 0.9999, 'best_epoch': 212, 'best_ap': 0.17}
            control_after = {**control_before, 'best_epoch': epoch, 'best_ap': metrics['AP']}
            with patch.object(torch.cuda, 'get_rng_state_all', return_value=[fake_cuda_state]):
                before = scope['_evidence_capture_rng']()
                assert stager.validate_rng(before) == before
                expected_state = scope['state'](epoch)
                assert equal_tree(expected_state['rng_state'], before)
                handle = stager.begin(temp, RUN, 'BASE', fake_protocol, epoch, identities, metrics,
                                      best['prediction_sha256'], best['evaluation_arrays_sha256'],
                                      audit['best_ema_tensor_sha256'], before, control_before, learning_rates,
                                      lambda stream: torch.save(scope['state'](epoch), stream))
                manifest_data = stager.finish(handle, False, audit['best_ema_tensor_sha256'],
                                              scope['_evidence_capture_rng'](), control_after,
                                              scope['_evidence_capture_rng'], nonce='1' * 32)
                assert scope['_evidence_capture_rng']() == before
            manifest = guard.parse_manifest(manifest_data, RUN, 'BASE', fake_protocol)
            guard.check_files(handle['stage'], manifest, include_ack=False)
            restored = torch.load(handle['stage'] / 'evaluated.pt', map_location='cpu', weights_only=True)
            assert equal_tree(restored, expected_state)
            for key in ('model', 'ema', 'optimizer', 'criterion', 'postprocessor', 'lr_warmup_scheduler'):
                assert equal_tree(checkpoint[key], restored[key]), key
            assert tensor_hash(restored['ema']['module']) == audit['best_ema_tensor_sha256']
            assert equal_tree(restored['rng_state'], before)
            phase = json.loads((handle['stage'] / 'phase_record.json').read_text(encoding='utf-8'))
            assert phase['metrics'] == metrics and phase['evaluated_tensor_sha256'] == audit['best_ema_tensor_sha256']
            assert phase['rng_before'] == phase['rng_after'] == json.loads(json.dumps(before))
            assert phase['run_id'] == RUN and phase['protocol_sha256'] == fake_protocol
            scratch_checked = True
        assert not temp.exists()
    finally:
        # Only the newly created empty parent is removed; original evidence is untouched.
        SCRATCH_PARENT.rmdir()
    assert scratch_checked and digest(checkpoint_path) == checkpoint_sha
    assert digest(source_predictions) == best['prediction_sha256'] and digest(source_arrays) == best['evaluation_arrays_sha256']
    result = {'status': 'CPU_CHECKPOINT_AND_FRESH_RNG_SERIALIZATION_ONLY_VERIFIED',
              'checked_utc': datetime.now(timezone.utc).isoformat(), 'torch': torch.__version__, 'numpy': np.__version__,
              'source_checkpoint_sha256': checkpoint_sha, 'source_ema_tensor_sha256': audit['best_ema_tensor_sha256'],
              'source_predictions_sha256': best['prediction_sha256'], 'source_arrays_sha256': best['evaluation_arrays_sha256'],
              'candidate_sha256': digest(CANDIDATE), 'stager_sha256': digest(PREP / 'epoch_evidence_stager.py'),
              'learning_rate_rows_copied': len(learning_rates),
              'checks': {'complete_synthetic_state_round_trip_equal': True, 'ema_tensor_hash_equal': True,
                         'optimizer_and_auxiliary_states_equal': True, 'fresh_rng_round_trip_equal': True,
                         'fresh_rng_unchanged_during_staging': True, 'source_files_unchanged': True,
                         'temporary_files_removed': True},
              'limits': ['RNG values are freshly created in this test and are not the historical training state.',
                         'The CUDA RNG field is a CPU-generator placeholder; no real CUDA RNG was captured.',
                         'Synthetic run identity and metadata were used; no training trajectory or resumed checkpoint was produced.',
                         'No model, forward pass, evaluation, optimizer update or remote connection was performed.']}
    with REPORT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'source_epoch_zero_based': epoch,
                      'learning_rate_rows': len(learning_rates), 'historical_rng_recovered': False,
                      'report_sha256': digest(REPORT)}))


if __name__ == '__main__':
    main()
