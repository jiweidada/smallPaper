from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

new_root = Path(__file__).resolve().parent
english_root = new_root.parent
old_root = english_root / '前期调研_码率受限航拍小目标_20261028_v1'
now = datetime.now(timezone.utc).isoformat()
paths = []

def amend(path, update):
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    before_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    update(data)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    paths.append({'path': str(path), 'before_sha256': before_sha,
                  'after_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})

def board_update(data):
    data['latest_user_steering'] = '重新开放新的encoder/decoder架构；停止围绕压缩分支继续寻找所有主线'
    data['latest_steering_recorded_utc'] = now
    data['current_priority'] = 'ARCHITECTURE_SEARCH_REOPENED'
    data['current_candidate_board'] = str(new_root / '01_候选判别表.json')
    data['mainline_validated'] = False
    data['C11']['status'] = 'NO_LONGER_CURRENT_PRIORITY_AFTER_USER_STEERING'
    data['C11']['next'] = 'No additional compression/restoration experiments; preserve completed results as historical evidence'

def ledger_update(data):
    data['status'] = 'ARCHITECTURE_SEARCH_REOPENED_NO_NEW_EXPERIMENT_THIS_TURN'
    data['latest_steering_recorded_utc'] = now
    data['mainline_validated'] = False
    data['current_priority'] = 'new encoder/decoder mechanism investigation'
    data['current_architecture_report'] = str(new_root / '00_方向纠偏与架构证据.txt')
    data['compression_branch_additional_experiments'] = 'STOPPED_AS_CURRENT_RESEARCH_PRIORITY'
    fit_path = old_root / '远端验证_20261029_v1' / 'calibration_control_v1' / 'fit_receipts.json'
    fits = json.loads(fit_path.read_text(encoding='utf-8-sig'))
    data['new_policy_training'] = len(fits)
    data['new_policy_training_definition'] = 'CPU utility-control fold fits in completed calibration stage; no new detector training'
    data['proxy_stage_decisions_frozen'] = True
    data['new_experiments_this_architecture_turn'] = {'detector_forwards': 0, 'detector_updates': 0, 'remote_jobs_started': 0}

amend(old_root / '主线目标_20261029_v1' / 'candidate_board.json', board_update)
amend(old_root / '远端验证_20261029_v1' / 'execution_ledger.json', ledger_update)

manifest = {'recorded_utc': now, 'amendments': paths, 'new_files': []}
for name in ('00_方向纠偏与架构证据.txt', '01_候选判别表.json', '02_检索记录.json'):
    path = new_root / name
    manifest['new_files'].append({'path': str(path), 'bytes': path.stat().st_size,
                                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if path.suffix == '.json':
        json.loads(path.read_text(encoding='utf-8'))
(new_root / '03_状态更新与文件校验.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status':'recorded', 'updated_existing':len(paths), 'new_artifacts':len(manifest['new_files']), 'cpu_utility_control_fold_fits':40, 'new_architecture_experiments':0}, ensure_ascii=False))
