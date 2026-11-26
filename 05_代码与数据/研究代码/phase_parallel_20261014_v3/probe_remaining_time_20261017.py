"""Read current timing evidence and queued workload without running inference."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

account = sys.argv[1]
root = Path('/25085400120/DEIM/experiments/phase_mechanism_20261012' if account == 'A'
            else '/24085403014/experiments/phase_mechanism_20261012')
stage = root / 'artifacts/round2_20261014_v1'
pointer = json.loads((stage / 'current_execution.json').read_text())
execution = stage / pointer['execution_directory']
queue = json.loads((execution / 'queue_state.json').read_text())
plan = json.loads((execution / 'queue_plan.json').read_text())
result = {'account': account, 'remote_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'queue_state': queue['state'], 'jobs': [], 'datasets': {},
          'models_to_postprocess': len(plan['existing_models']) + len(plan['jobs']),
          'postprocess_outputs_exist': (root / 'round2_results').exists()}
for key in ['val_annotations', 'testdev_annotations']:
    data = json.loads(Path(plan[key]).read_text())
    result['datasets'][key] = {'images': len(data['images']), 'annotations': len(data['annotations'])}
for spec in plan['jobs']:
    job = spec.replace(':', '_seed')
    status = json.loads((root / 'jobs' / (job + '.json')).read_text())
    if status['state'] != 'running' and not job.startswith('soep_lite'):
        continue
    log_path = root / 'outputs' / job / 'log.txt'
    raw = log_path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    rows = [row for row in rows if 'test_coco_eval_bbox' in row]
    recent = [{'epoch': row['epoch'], 'train_seconds': row.get('experiment_train_seconds'),
               'epoch_seconds': row.get('experiment_epoch_seconds')} for row in rows[-8:]]
    phases = {}
    for name, low, high in [('early', 4, 45), ('middle', 46, 83), ('noaug', 84, 95)]:
        values = [row['experiment_epoch_seconds'] for row in rows
                  if low <= row['epoch'] <= high and row.get('experiment_epoch_seconds')]
        phases[name] = {'n': len(values), 'median_epoch_seconds': statistics.median(values) if values else None}
    console = root / 'logs' / (job + '.log')
    with console.open('rb') as handle:
        handle.seek(max(0, console.stat().st_size - 12 * 1024 * 1024))
        text = handle.read().decode(errors='replace')
    marker = text.rfind('RESUME_V4')
    if marker >= 0:
        text = text[marker:]
    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)
    starts = {}
    for line in text.splitlines():
        match = re.search(r'Epoch:\s*(\d+)/96\s+\[\s*0/1617\]', line)
        if match:
            try:
                starts[int(match[1])] = datetime.datetime.strptime(line[:19], '%Y-%m-%d %H:%M:%S')
            except ValueError:
                pass
    intervals = [{'epoch': epoch, 'seconds': (starts[epoch + 1] - stamp).total_seconds()}
                 for epoch, stamp in sorted(starts.items()) if epoch + 1 in starts]
    batches = list(re.finditer(r'Epoch:\s*(\d+)/96\s+\[\s*(\d+)/(\d+)\]', text))
    latest = {'epoch_zero_based': int(batches[-1][1]), 'batch_zero_based': int(batches[-1][2]),
              'batches_per_epoch': int(batches[-1][3])} if batches else None
    result['jobs'].append({'job': job, 'state': status['state'], 'evaluated_epochs': len(rows),
                           'latest_batch': latest, 'recent_timings': recent, 'phase_timings': phases,
                           'recent_start_to_start_seconds': intervals[-8:],
                           'raw_log_sha256': hashlib.sha256(raw).hexdigest()})
tools = Path(queue['code_directory']).parent / 'tools'
result['tool_evidence'] = {}
for name in ['run_round2_queue.py', 'run_translation_suite.py', 'analyze_preds.py', 'dump_preds.py']:
    path = tools / name
    if not path.is_file():
        result['tool_evidence'][name] = {'missing': True, 'path': str(path)}
        continue
    raw = path.read_bytes()
    relevant = []
    for number, line in enumerate(raw.decode(errors='replace').splitlines(), 1):
        if any(token in line for token in ['offsets =', 'offsets +=', 'coco_ap(', 'cli_test',
                                           'project-eval', "('val',", 'export_cli', 'elapsed_seconds']):
            relevant.append({'line': number, 'text': line.strip()})
    result['tool_evidence'][name] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'relevant_lines': relevant}
print(json.dumps(result, ensure_ascii=False))
