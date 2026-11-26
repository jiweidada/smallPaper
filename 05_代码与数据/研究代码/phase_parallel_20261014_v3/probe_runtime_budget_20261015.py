"""Read remote timings, checkpoint inventory and resume code; no mutations."""
import datetime
import hashlib
import json
import pathlib
import re
import statistics

account = 'A' if pathlib.Path('/25085400120/DEIM').is_dir() else 'B'
root = pathlib.Path('/25085400120/DEIM/experiments/phase_mechanism_20261012' if account == 'A'
                    else '/24085403014/experiments/phase_mechanism_20261012')
stage = root / 'artifacts/round2_20261014_v1/execution_v3'
queue = json.loads((stage / 'queue_state.json').read_text())
result = {'account': account, 'remote_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'queue': queue, 'jobs': {}, 'code_evidence': {}}
names = set(queue['active_jobs']) | set(queue['training_completed_jobs'])
for name in sorted(names):
    log = root / 'logs' / (name + '.log')
    raw = log.read_bytes()
    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', raw.decode('utf-8', errors='replace'))
    starts = {}
    for line in text.splitlines():
        m = re.search(r'Epoch:\s*(\d+)/96\s+\[\s*0/1617\]', line)
        if m:
            try:
                starts.setdefault(int(m[1]), datetime.datetime.strptime(line[:19], '%Y-%m-%d %H:%M:%S'))
            except ValueError:
                pass
    durations = [{'epoch_zero_based': i, 'seconds': (starts[i+1]-value).total_seconds()}
                 for i, value in starts.items() if i+1 in starts]
    recent = durations[-12:]
    phases = {}
    for label, lo, hi in [('early', 4, 45), ('middle', 46, 83), ('noaug', 84, 94)]:
        values = [r['seconds'] for r in durations if lo <= r['epoch_zero_based'] <= hi]
        phases[label] = {'n': len(values), 'median_seconds': statistics.median(values) if values else None,
                         'mean_seconds': statistics.mean(values) if values else None}
    files = []
    for path in (root/'outputs'/name).glob('*.pth'):
        stat = path.stat()
        files.append({'name': path.name, 'path': str(path), 'bytes': stat.st_size, 'mtime': stat.st_mtime})
    status_path = root/'jobs'/(name+'.json')
    result['jobs'][name] = {'status': json.loads(status_path.read_text()), 'log_sha256': hashlib.sha256(raw).hexdigest(),
                            'recent_epoch_durations': recent, 'phase_timings': phases, 'checkpoints': files}
for relative in ['engine/solver/det_solver.py', 'engine/solver/_solver.py', 'train.py']:
    path = stage/'code'/relative
    if not path.is_file():
        result['code_evidence'][relative] = {'missing': True}
        continue
    raw = path.read_bytes()
    lines = raw.decode('utf-8', errors='replace').splitlines()
    patterns = r'checkpoint\.pth|last\.pth|last_epoch|save_checkpoint|state_dict|load_resume|resume|rng|random\.getstate|torch\.get_rng_state'
    indices = [i for i, line in enumerate(lines) if re.search(patterns, line, re.I)]
    included = sorted({j for i in indices for j in range(max(0, i-2), min(len(lines), i+4))})
    result['code_evidence'][relative] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                                        'excerpts': [{'line': i+1, 'text': lines[i]} for i in included]}
print(json.dumps(result, ensure_ascii=False))
