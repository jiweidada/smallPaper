"""Integrity and import tests; no detector, evaluator numerical call, or optimizer."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch
import evaluator_binding as binding

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]/'data'/'00_实验计划与证据索引'/\
    '单一候选评价依赖补齐与构建核验_20261008_v1'/'isolated_runtime'/'pycocoeval'
checks = []


def ok(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


def rejects(name, call):
    try:
        call()
    except (ValueError, RuntimeError):
        checks.append(name)
        return
    raise AssertionError('Expected rejection: ' + name)


original = binding.fingerprint(PACKAGE)
ok('static inspection does not import torch or evaluator',
   'torch' not in sys.modules and 'pycocoeval' not in sys.modules)
ok('complete package contains six sources and one backend', len(original['files']) == 7)
rejects('missing binding rejected', lambda: binding.validate({}))
with tempfile.TemporaryDirectory(prefix='fine_residual_evaluator_') as temp:
    copy_root = Path(temp)/'pycocoeval'
    shutil.copytree(PACKAGE, copy_root)
    ok('identical copy has same portable binding', binding.fingerprint(copy_root) == original)
    pyfile = copy_root/'cocoeval.py'
    raw = pyfile.read_bytes()
    pyfile.write_bytes(raw + b'\n# altered test copy\n')
    rejects('changed source rejected', lambda: binding.fingerprint(copy_root))
    pyfile.write_bytes(raw)
    binary = copy_root/original['backend']
    original_binary = binary.read_bytes()
    binary.write_bytes(bytes([original_binary[0] ^ 1]) + original_binary[1:])
    rejects('changed backend rejected against prepared binding',
            lambda: binding.verify_files(copy_root, original))
    binary.write_bytes(original_binary)
    renamed = copy_root/(original['backend'] + '.held')
    binary.rename(renamed)
    rejects('missing compiled extension rejected', lambda: binding.fingerprint(copy_root))
    renamed.rename(binary)
    second = copy_root/'_mask.other.pyd'
    second.write_bytes(b'not a real extension')
    rejects('ambiguous compiled extensions rejected', lambda: binding.fingerprint(copy_root))
    second.unlink()
    extra = copy_root/'unexpected.py'
    extra.write_text('value = 1\n', encoding='utf-8')
    rejects('extra executable package file rejected', lambda: binding.fingerprint(copy_root))

source = {'module': 'pycocoeval.cocoeval', 'class': 'COCOeval',
          'sha256': original['files']['cocoeval.py'], 'binding': original}
endpoint = {'identity': {'evaluator_binding': original}, 'evaluator_source': source}
binding.same_result_evaluator(endpoint, deepcopy(endpoint))
checks.append('identical complete endpoint evaluators accepted')
old_endpoint = deepcopy(endpoint)
old_endpoint['evaluator_source'].pop('binding')
rejects('old Python-only endpoint rejected', lambda: binding.same_result_evaluator(endpoint, old_endpoint))
different = deepcopy(endpoint)
different['identity']['evaluator_binding']['files'][original['backend']] = 'a'*64
# deepcopy preserves source/identity sharing, as a coherent different runtime would.
ok('comparison fixture differs only in backend hash',
   different['evaluator_source']['sha256'] == source['sha256'] and
   different['evaluator_source']['binding'] == different['identity']['evaluator_binding'])
rejects('same cocoeval.py with different compiled backend rejected',
        lambda: binding.same_result_evaluator(endpoint, different))
mismatch = deepcopy(endpoint)
mismatch['evaluator_source'] = deepcopy(source)
mismatch['evaluator_source']['binding']['files'][original['backend']] = 'b'*64
rejects('endpoint package must match its prepared identity',
        lambda: binding.validate_result_source(mismatch['evaluator_source'], mismatch['identity']))

runtime = binding.activate(PACKAGE, original)
ok('activation loads all seven bound modules', len(runtime['loaded_modules']) == 7)
rejects('preloaded evaluator rejected on activation', lambda: binding.activate(PACKAGE, original))
rejects('foreign package location rejected', lambda: binding.loaded_fingerprint(original, HERE))
mask = sys.modules['pycocoeval.mask']
with patch.object(mask, 'iou', lambda *args: None):
    rejects('redirected compiled IoU call rejected', lambda: binding.loaded_fingerprint(original))
binding.loaded_fingerprint(original, PACKAGE)
checks.append('original IoU binding restored after test')
import torch
ok('CUDA remains uninitialized', not torch.cuda.is_initialized())
ok('original runtime files unchanged', binding.fingerprint(PACKAGE) == original)
out = HERE/'engineering_evaluator'
out.mkdir(exist_ok=False)
report = {'status': 'PASS_PACKAGE_IDENTITY_AND_IMPORT_ONLY', 'checks': checks,
          'check_count': len(checks), 'runtime': runtime, 'model_calls': 0,
          'evaluator_numerical_calls': 0, 'optimizer_calls': 0,
          'checkpoint_loads': 0, 'parameter_updates': 0, 'training_started': False,
          'CUDA_initialized': torch.cuda.is_initialized(),
          'test_script_sha256': binding.sha(__file__)}
(out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'status': report['status'], 'check_count': len(checks)}))
