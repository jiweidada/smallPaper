"""Describe saved historical AP trajectories; no model or evaluator execution."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json
import math

ROOT = Path(__file__).resolve().parent
REC = ROOT.parent / '修订记录_20261103'
MANIFEST = REC / 'source_manifest.json'
NEW_RUN = ROOT / 'remote_evidence/interrupted_BASE_20261105/runs/BASE/epochs.jsonl'
VERIFIED = ROOT / 'interrupted_base_download_verified_20261105.json'
REPORT = ROOT / 'historical_ap_trajectory_context_20261105.json'

def sha(path):
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def describe(rows, epoch_key, metric):
    epochs = [row[epoch_key] for row in rows]
    aps = [metric(row) for row in rows]
    assert all(isinstance(e, int) and math.isfinite(ap) for e, ap in zip(epochs, aps))
    windows = []
    for i in range(8, len(rows)):
        sequence = epochs[i-8:i+1]
        if sequence == list(range(sequence[0], sequence[0]+9)) and 40 <= sequence[0] and sequence[-1] <= 295:
            windows.append({'start_epoch': sequence[0], 'end_epoch': sequence[-1],
                            'start_AP': aps[i-8], 'end_AP': aps[i], 'AP_decrease': aps[i-8]-aps[i]})
    same = [w for w in windows if w['start_epoch'] == 216 and w['end_epoch'] == 224]
    assert len(same) <= 1
    return {'rows': len(rows), 'first_epoch': epochs[0], 'last_epoch': epochs[-1],
            'unique_epoch_count': len(set(epochs)), 'same_216_to_224_window': same[0] if same else None,
            'largest_8_epoch_drop_within_40_to_295': max(windows, key=lambda w: w['AP_decrease']) if windows else None}

assert not REPORT.exists()
source_manifest = load(MANIFEST)
known = {Path(item['snapshot']).resolve(): item['sha256'] for item in source_manifest}
verified = load(VERIFIED)
entry = next(f for f in verified['files'] if f['relative_path'] == 'runs/BASE/epochs.jsonl')
assert sha(NEW_RUN) == entry['sha256']
current = [json.loads(line) for line in NEW_RUN.read_text(encoding='utf-8').splitlines()]
assert len(current) == 225 and [r['epoch'] for r in current] == list(range(225))
summary = {'BASE_interrupted': describe(current, 'epoch', lambda r: r['metrics']['AP'])}
sources = {'BASE_interrupted': {'path': str(NEW_RUN), 'sha256': sha(NEW_RUN)}}
for label in ['N0', 'N1', 'N2', 'N3', 'N4', 'S1']:
    path = REC / 'source_snapshot' / label / 'log.txt'
    assert sha(path) == known[path.resolve()]
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    assert rows
    summary[label] = describe(rows, 'epoch', lambda r: r['test_coco_eval_bbox'][0])
    sources[label] = {'path': str(path), 'sha256': sha(path)}
report = {'status': 'HISTORICAL_LOG_WINDOWS_DESCRIBED_NOT_MATCHED_COMPARISONS',
          'checked_local_utc': datetime.now(timezone.utc).isoformat(),
          'historical_source_manifest_sha256': sha(MANIFEST), 'verified_new_run_receipt_sha256': sha(VERIFIED),
          'source_logs': sources, 'trajectories': summary,
          'definition': 'AP decreases are signed AP(start)-AP(end) over nine contiguous log rows with epoch indices eight apart; only windows from epoch 40 through 295 are considered.',
          'limits': ['Historical batch sizes, model configurations and unverified runtime/source versions differ; these descriptive windows are not controlled comparisons.',
                     'The interrupted BASE has one validation-selected trajectory, ends at epoch 224, and its late model checkpoint was not saved.',
                     'Presence or absence of a similar drop in these records cannot identify the cause or establish frequency across seeds.'],
          'model_executions': 0, 'evaluator_executions': 0, 'optimizer_updates': 0}
with REPORT.open('x', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'status': report['status'], 'same_window': {k: v['same_216_to_224_window'] for k, v in summary.items()},
                  'largest_drops': {k: v['largest_8_epoch_drop_within_40_to_295'] for k, v in summary.items()}}))
