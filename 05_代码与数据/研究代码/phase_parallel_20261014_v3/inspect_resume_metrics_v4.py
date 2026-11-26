"""Read-only locate resumed structured logs and their actual epoch labels."""
import datetime
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
result = {'remote_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'logs': [], 'source': []}
for path in root.rglob('log.txt'):
    if not path.is_file():
        continue
    raw = path.read_bytes()
    rows = []
    for line in raw.splitlines():
        try:
            row = json.loads(line)
            if 'test_coco_eval_bbox' in row:
                rows.append(row)
        except (ValueError, UnicodeError):
            pass
    result['logs'].append({'path': str(path), 'size': len(raw), 'mtime': path.stat().st_mtime,
                           'rows': len(rows), 'latest_epoch': rows[-1].get('epoch') if rows else None})
solver = root / 'artifacts/round2_20261014_v1/execution_v3/code/engine/solver/det_solver.py'
lines = solver.read_text().splitlines()
for index, line in enumerate(lines):
    if any(x in line for x in ['log_stats', 'log.txt', 'resume', 'output_dir']):
        result['source'].append({'line': index + 1, 'text': '\n'.join(lines[max(0,index-2):index+4])})
for name in ['engine/solver/_solver.py', 'train.py', 'train_round2.py']:
    path = root / 'artifacts/round2_20261014_v1/execution_v3/code' / name
    if path.is_file():
        content = path.read_text().splitlines()
        for index, line in enumerate(content):
            if any(x in line for x in ['output_dir', 'args.resume', 'load_resume_state']):
                result['source'].append({'file': name, 'line': index + 1,
                                         'text': '\n'.join(content[max(0,index-3):index+5])})
result['resume_directory_files'] = [{'name': p.name, 'bytes': p.stat().st_size, 'mtime': p.stat().st_mtime}
    for p in (root / 'artifacts/round2_20261014_v1/execution_v4/resume_inputs').iterdir() if p.is_file()]
print(json.dumps(result, ensure_ascii=False))
