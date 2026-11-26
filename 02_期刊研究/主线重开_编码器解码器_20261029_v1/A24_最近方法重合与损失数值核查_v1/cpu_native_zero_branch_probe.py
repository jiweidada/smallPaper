"""CPU synthetic operator check, not a detector replay or mechanism assay."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import torch

SOURCE = Path('/root/c10_region_restore_20261029_v1/architecture_probe_v1/dome_assets_v1/source/src/zoo/dome/dome_criterion.py')
assert not torch.cuda.is_initialized()
text = SOURCE.read_text(encoding='utf-8')
tree = ast.parse(text)
klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DomeCriterion')
method = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == 'loss_local')
branch = next(node for node in ast.walk(method) if isinstance(node, ast.If) and ast.unparse(node.test) == 'torch.equal(pred_corners, target_corners)')
statement = branch.body[0]
assert isinstance(statement, ast.Assign)
expression = ast.unparse(statement.value)
assert expression == 'pred_corners.sum() * 0'
code = compile(ast.Expression(statement.value), str(SOURCE), 'eval')
rows = []
for dtype, queries in [(torch.float16, 20), (torch.float16, 500), (torch.float16, 1804), (torch.float32, 1804)]:
    pred = torch.ones((2, queries, 4 * 33), dtype=dtype, device='cpu', requires_grad=True)
    target = pred.detach().clone()
    assert torch.equal(pred, target)
    value = eval(code, {'pred_corners': pred})
    total = pred.sum()
    rows.append({
        'dtype': str(dtype), 'shape': list(pred.shape), 'elements': pred.numel(),
        'all_inputs_finite': bool(torch.isfinite(pred).all()),
        'equal_teacher_branch_condition': True,
        'sum_dtype': str(total.dtype), 'sum_value_text': str(float(total.detach())),
        'native_zero_value_text': str(float(value.detach())),
        'native_zero_finite': bool(torch.isfinite(value)),
        'fixture': 'All-one synthetic corners; actual failed training tensors were not retained.'
    })
    if dtype == torch.float16 and queries == 1804:
        stable = pred.sum(dtype=torch.float32) * 0
        assert bool(torch.isfinite(stable)) and float(stable.detach()) == 0.0
        stable.backward()
        assert bool(torch.isfinite(pred.grad).all()) and not bool(pred.grad.any())
        rows[-1]['FP32_accumulator_zero_control'] = {
            'value': float(stable.detach()), 'dtype': str(stable.dtype),
            'gradient_finite': bool(torch.isfinite(pred.grad).all()),
            'gradient_all_zero': not bool(pred.grad.any()),
            'scope': 'Synthetic algebraically zero branch only; not a patch or full-criterion validation.'
        }
assert rows[0]['native_zero_finite']
assert not rows[1]['native_zero_finite'] and not rows[2]['native_zero_finite']
assert rows[3]['native_zero_finite']
assert not torch.cuda.is_initialized()
result = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'status': 'FINITE_FP16_INPUT_SYNTHETIC_NONFINITE_ZERO_BRANCH_REPRODUCED',
    'source': str(SOURCE), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'unmodified_expression': expression, 'source_line': statement.lineno,
    'torch_version': torch.__version__, 'device': 'cpu',
    'CUDA_initialized': torch.cuda.is_initialized(),
    'detector_forwards': 0, 'optimizer_updates': 0, 'real_image_inputs': 0,
    'cases': rows,
    'not_proven': [
        'The actual A22 failing loss key or upstream source.',
        'The lost step526 trained state or microbatch2105 predictions.',
        'A bug in the VSSD shared-state mechanism.',
        'Any task accuracy improvement or new research algorithm.'
    ],
    'no_training_relaunch': True,
}
print(json.dumps(result, ensure_ascii=False, allow_nan=False))
