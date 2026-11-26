"""Inspect accepted-prefix metrics and current storage without running models."""
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BACKUPS = Path('D:/MPSMA_DEIM_experiment_backups/BASE_FULL_FRESH_20261112_v1')
METRICS = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint', required=True)
a = p.parse_args()
assert a.checkpoint.replace('_', '').isalnum()
export_dir = ROOT / '实验数据汇总_20261108/fresh_pair_20261112'
export = json.loads((export_dir / '逐轮导出核对_FRESH.json').read_text(encoding='utf-8-sig'))
raw = (export_dir / '逐轮指标_FRESH.csv').read_bytes()
csv_sha = hashlib.sha256(raw).hexdigest()
assert csv_sha == export['csv']['sha256']
rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
assert len(rows) == export['csv']['rows']
observations, advisories = [], []
for arm, cap_gib in [('BASE_FRESH', 45), ('FULL_FRESH', 75)]:
    group = sorted([r for r in rows if r['arm'] == arm], key=lambda r: int(r['round_1based']))
    assert len(group) == export['accepted_rounds'][arm]
    assert [int(r['round_1based']) for r in group] == list(range(1, len(group) + 1))
    running_best = 0.0
    for r in group:
        number = int(r['round_1based'])
        for metric in METRICS:
            if not math.isfinite(float(r[metric])):
                advisories.append({'arm': arm, 'round': number, 'type': 'nonfinite_metric', 'metric': metric})
        ap = float(r['AP'])
        if running_best >= .05 and ap < .1 * running_best:
            advisories.append({'arm': arm, 'round': number, 'type': 'extreme_AP_drop', 'AP': ap, 'prior_best_AP': running_best})
        if math.isfinite(ap):
            running_best = max(running_best, ap)
    latest, best = group[-1], max(group, key=lambda r: float(r['AP']))
    store = BACKUPS / arm / 'backups'
    cap_bytes = cap_gib * 1024**3
    events = (store / 'monitor_events.jsonl').read_text(encoding='utf-8').splitlines()
    for line in reversed(events):
        event = json.loads(line)
        if event['status'] == 'MONITOR_READY':
            cap_bytes = event['storage_limit_bytes']
            break
    used = sum(f.stat().st_size for f in (store / 'objects').iterdir() if f.is_file())
    accepted = sum((f / 'accepted.json').is_file() for f in (store / 'epochs').iterdir())
    assert accepted >= len(group)
    if used >= cap_bytes:
        advisories.append({'arm': arm, 'type': 'CAS_storage_limit_reached'})
    observations.append({'arm': arm, 'exported_accepted_rounds': len(group), 'local_accepted_stage_directories': accepted,
                         'latest_AP': float(latest['AP']), 'latest_APs': float(latest['APs']),
                         'best_AP_so_far': float(best['AP']), 'best_AP_round': int(best['round_1based']),
                         'CAS_used_bytes': used, 'declared_CAS_limit_bytes': cap_bytes,
                         'CAS_remaining_bytes': cap_bytes - used})
free = shutil.disk_usage(BACKUPS).free
if free < 32 * 1024**3:
    advisories.append({'type': 'D_free_space_below_declared_reserve'})
report = {'observed_local_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
          'scope': 'Descriptive inspection of the entire accepted prefix within each arm, and current storage only. No unequal-round comparison, final endpoint claim, future storage guarantee or budget change.',
          'extreme_drop_rule': 'Heuristic advisory only: AP below 10% of a prior within-arm best AP of at least 0.05; never changes training or endpoints.',
          'csv_sha256': csv_sha, 'accepted_rounds': export['accepted_rounds'], 'observations': observations,
          'advisories': advisories, 'D_free_bytes': free, 'minimum_D_reserve_bytes': 32 * 1024**3,
          'model_forwards': 0, 'evaluator_runs': 0, 'training_actions': 0}
out = HERE / 'monitor_goal_20261112' / (a.checkpoint + '_metric_storage_observation.json')
with out.open('x', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write('\n')
print(json.dumps({'report': str(out), 'accepted_rounds': report['accepted_rounds'], 'observations': observations,
                  'advisories': advisories, 'D_free_bytes': free}, ensure_ascii=False, allow_nan=False))
