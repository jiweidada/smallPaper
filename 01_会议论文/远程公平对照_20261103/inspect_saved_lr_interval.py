"""Describe saved per-step learning rates around the interrupted AP decline."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import csv
import gzip
import json
import math

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'remote_evidence/interrupted_BASE_20261105'
CONFIG = BASE / 'execution_configs/BASE.yml'
LR = BASE / 'runs/BASE/executed_learning_rates.csv.gz'
REPORT = ROOT / 'interrupted_base_lr_interval_20261105.json'

def sha(path):
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

assert not REPORT.exists()
verified_path = ROOT / 'interrupted_base_download_verified_20261105.json'
verified = json.loads(verified_path.read_text(encoding='utf-8'))
items = {i['relative_path']: i for i in verified['files']}
for name in ['execution_configs/BASE.yml', 'runs/BASE/executed_learning_rates.csv.gz']:
    assert sha(BASE / name) == items[name]['sha256']
cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
boundaries = cfg['train_dataloader']['dataset']['transforms']['policy']['epoch']
assert boundaries == [4, 156, 296]
values = [set() for _ in range(4)]
count, last_attempt, truncated = 0, -1, False
try:
    with gzip.open(LR, 'rt', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            attempt, epoch, batch = map(int, [row['attempt'], row['actual_epoch'], row['batch']])
            assert attempt == last_attempt + 1 and (epoch, batch) == divmod(attempt, 1617)
            last_attempt = attempt
            if 210 <= epoch <= 224:
                count += 1
                for i in range(4):
                    value = float(row['group' + str(i)])
                    assert math.isfinite(value)
                    values[i].add(value)
except EOFError:
    truncated = True
assert count == 15 * 1617 and last_attempt == 225 * 1617 - 1 and truncated
assert all(len(v) == 1 for v in values)
report = {'status': 'SAVED_LR_INTERVAL_DESCRIBED_NOT_CAUSAL',
          'checked_local_utc': datetime.now(timezone.utc).isoformat(),
          'lr_file_sha256': sha(LR), 'config_sha256': sha(CONFIG),
          'verified_download_sha256': sha(verified_path), 'epoch_indices': [210, 224],
          'logged_steps_in_interval': count, 'per_group_unique_learning_rates': [sorted(v) for v in values],
          'last_complete_lr_attempt': last_attempt, 'gzip_missing_end_marker': truncated,
          'configured_augmentation_boundaries': boundaries,
          'interpretation': 'Saved LR entries are constant over epochs 210-224 and no configured augmentation boundary lies there. This does not identify the cause of the AP decline or establish the LR that would prevent it.',
          'model_executions': 0, 'evaluator_executions': 0, 'optimizer_updates': 0}
with REPORT.open('x', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'status': report['status'], 'steps': count,
                  'per_group_unique_learning_rates': report['per_group_unique_learning_rates'],
                  'last_complete_lr_attempt': last_attempt}))
