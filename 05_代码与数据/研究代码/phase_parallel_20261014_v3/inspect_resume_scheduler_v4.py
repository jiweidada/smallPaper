import json
import pathlib
root = pathlib.Path('/25085400120/DEIM/experiments/phase_mechanism_20261012/artifacts/round2_20261014_v1/execution_v3/code')
result = {}
for path in (root/'engine/optim').rglob('*.py'):
    text = path.read_text()
    if 'class FlatCosineLRScheduler' in text:
        result[str(path)] = text
path = root/'engine/solver/det_solver.py'
lines = path.read_text().splitlines()
result['fit_945_1038'] = [{'line': i+1, 'text': line} for i, line in enumerate(lines) if 945 <= i+1 <= 1038]
result['bootstrap_595_658'] = [{'line': i+1, 'text': line} for i, line in enumerate(lines) if 595 <= i+1 <= 658]
print(json.dumps(result))
