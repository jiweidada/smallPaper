"""Record the explicit user approval received for this one bounded diagnostic."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLAN = ROOT / '基线骤降最小诊断方案_待授权_20261105.txt'
STORAGE = ROOT / '诊断存储与中断保护补充_待授权_20261105.txt'
RECEIPT = HERE / 'diagnostic_user_authorization_20261105.json'
PLAN_SHA = 'fc8804407fcdbd9dd8210cd7f2f3c34ae503189c049659cbccf45a0023eff4fa'
STORAGE_SHA = '7dfb4c22fd467154713d9cf64451ecfe970358122b70b4197110b809b2ff4008'

assert sha256(PLAN.read_bytes()).hexdigest() == PLAN_SHA
assert sha256(STORAGE.read_bytes()).hexdigest() == STORAGE_SHA
assert not RECEIPT.exists(), 'Do not replace or duplicate an authorization receipt'
receipt = {'status': 'EXPLICITLY_AUTHORIZED_ONE_BOUNDED_DIAGNOSTIC',
           'recorded_local_utc': datetime.now(timezone.utc).isoformat(),
           'user_reply_question_id': 'call_94bd46e1c9f94c6093cfc273b21764b2',
           'user_reply': '授权该诊断',
           'plan_sha256': PLAN_SHA, 'storage_supplement_sha256': STORAGE_SHA,
           'run_id': 'mpsma_diag_base_apdrop_20261105_v1',
           'input_checkpoint_sha256': '33bb8012aa6b43a505b2f1a554189ed6abeeb4c39dc5cd6f20e83726fb922f65',
           'host_port': 30166, 'gpus': 1, 'model': 'BASE', 'fp32': True, 'batch_size': 4, 'seed': 0,
           'first_epoch_zero_based': 214, 'last_epoch_zero_based': 224,
           'max_training_epochs': 11, 'max_optimizer_updates': 17787,
           'max_full_validation_passes': 14, 'images_per_validation': 548,
           'initial_ema_max_absolute_ap_difference': 0.001, 'ema_ap_drop_stop': 0.05,
           'local_backup_required_after_each_complete_epoch': True, 'backup_timeout_seconds': 600,
           'automatic_retries': False, 'hyperparameter_tuning_authorized': False,
           'full_fair_pair_or_additional_seeds_authorized': False,
           'purpose': 'Diagnose whether collapse recurs and whether raw model and EMA diverge; not a fair baseline/full comparison.'}
with RECEIPT.open('x', encoding='utf-8') as f:
    json.dump(receipt, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'status': receipt['status'], 'run_id': receipt['run_id'],
                  'max_epochs': receipt['max_training_epochs'], 'max_validations': receipt['max_full_validation_passes']}))
