"""Inspect the saved checkpoint structure for recovery planning, without model execution."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json
import torch

PAPER = Path(__file__).resolve().parent.parent
REMOTE = PAPER / '远程公平对照_20261103'
CHECKPOINT = REMOTE / 'remote_evidence/interrupted_BASE_20261105/runs/BASE/best.pt'
AUDIT = REMOTE / 'interrupted_base_saved_file_audit_20261105.json'
REPORT = REMOTE / 'interrupted_base_checkpoint_recovery_20261105.json'

def sha(path):
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

assert not REPORT.exists()
audit = json.loads(AUDIT.read_text(encoding='utf-8'))
assert audit['status'] == 'BEST_CHECKPOINT_BOUND_LAST_ARRAYS_ONLY'
assert sha(CHECKPOINT) == audit['best_checkpoint_sha256']
ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
keys = sorted(ckpt)
assert ckpt['arm'] == 'BASE' and ckpt['actual_loop_epoch'] == 213
assert ckpt['optimizer_step_attempts'] == 346038
assert ckpt['ema']['updates'] == 346038
assert not any('rng' in str(key).lower() or 'sampler' in str(key).lower() for key in keys)
assert set(['model', 'ema', 'optimizer', 'criterion', 'postprocessor', 'lr_warmup_scheduler']) <= set(keys)
report = {'status': 'SAVED_BEST_CHECKPOINT_STATE_INSPECTED',
          'checked_local_utc': datetime.now(timezone.utc).isoformat(),
          'saved_file_audit_sha256': sha(AUDIT), 'checkpoint_sha256': sha(CHECKPOINT),
          'checkpoint_top_level_keys': keys, 'actual_epoch_zero_based': ckpt['actual_loop_epoch'],
          'optimizer_step_attempts': ckpt['optimizer_step_attempts'], 'ema_updates': ckpt['ema']['updates'],
          'optimizer_param_groups': len(ckpt['optimizer']['param_groups']),
          'full_rng_and_sampler_state_saved_at_top_level': False,
          'recovery_limitation': 'This is the validation-best epoch 213, not the last completed epoch 224. No complete global RNG or sampler state is recorded as a top-level checkpoint field; exact continuation of the interrupted trajectory is not established.',
          'model_executions': 0, 'evaluator_executions': 0, 'optimizer_updates': 0}
with REPORT.open('x', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'status': report['status'], 'epoch': report['actual_epoch_zero_based'],
                  'checkpoint_keys': keys, 'exact_resume_established': False}))
