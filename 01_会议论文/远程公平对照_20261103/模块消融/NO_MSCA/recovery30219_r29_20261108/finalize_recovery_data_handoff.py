"""Verify final data deliverables and retain a precise live-versus-audited cutoff."""
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

here = Path(__file__).resolve().parent
bundle = here.parents[1]
paper = here.parents[3]
data = paper / '实验数据汇总_20261108'
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return sha256(path.read_bytes()).hexdigest()
manifest_path = data / '交付清单.json'
manifest = read(manifest_path)
assert manifest['status'] == 'COMPLETE' and manifest['NO_MSCA_accepted_rounds'] == 30
for item in manifest['files']:
    path = (data / item['file']).resolve(strict=True)
    assert path.is_relative_to(data) and path.stat().st_size == item['bytes']
    assert sha(path) == item['sha256'], item['file']
xlsx = data / 'outputs/实验数据汇总_20261108/实验数据汇总.xlsx'
raw = data / '逐轮指标.csv'
assert sha(xlsx) == '1a36120f9d179fdac7a34da648b50ae197ed72025c9ce16d3dc78680980aa144'
assert sha(raw) == '747b56cf3b38175407e13d6b1a8e07c4a436cb3eb089bdea5106864cc67c59a8'
with raw.open(encoding='utf-8-sig', newline='') as stream:
    assert sum(1 for _ in csv.DictReader(stream)) == 990
update = read(data / '第30轮更新核查.json')
assert update['status'] == 'ROUND30_UPDATE_VERIFIED' and update['raw_metric_values_rechecked'] == 11880
assert update['NO_MSCA_320_round_endpoint_cells_blank'] == 24
state_path = bundle / 'current_execution_state.json'
state = read(state_path)
arm = state['NO_MSCA']
assert arm['status'] == 'RECOVERY_RUNNING' and arm['independent_lineage_audit_round'] == 30
snapshot_path = here / 'live_current_for_handoff.json'
live = json.loads(read(snapshot_path)['stdout'])
receipt = live['records']['run/receipt.json']
assert receipt['pid'] == 1779501 and receipt['status'] == 'RUNNING'
assert receipt['optimizer_step_attempts'] >= 48511
report = {'status': 'RUNNING_RECOVERY_AND_AUDITED_DATA_HANDOFF_VERIFIED',
    'created_local_utc': datetime.now(timezone.utc).isoformat(), 'local_date': '2026-11-08',
    'live_training_round_at_snapshot': receipt['actual_loop_epoch'] + 1,
    'live_optimizer_step_attempts_at_snapshot': receipt['optimizer_step_attempts'],
    'latest_transport_verified_round': arm['last_accepted_round'],
    'independently_verified_round': 30, 'data_snapshot_NO_MSCA_round': 30,
    'xlsx': {'path': str(xlsx), 'sha256': sha(xlsx)},
    'per_epoch_csv': {'path': str(raw), 'sha256': sha(raw), 'rows': 990},
    'data_manifest': {'path': str(manifest_path), 'sha256': sha(manifest_path), 'files_verified': len(manifest['files'])},
    'live_evidence': {'path': str(snapshot_path), 'sha256': sha(snapshot_path)},
    'backup_client': {'pid': arm['monitor_pid'], 'create_time': arm['monitor_create_time'],
                      'session_id': arm['monitor_session_id'], 'remain_running': True},
    'backup_recovery_references': read(here / 'recovery_references_D_archive_verified.json')['destination'],
    'remaining_work': ['NO_MSCA full320 trajectory and genuine exit/backup verification',
                       'conditional MSCA endpoint analysis and manuscript revision'],
    'submission_ready': False, 'paper_revision': 35}
with (here / 'running_recovery_and_data_handoff_verified.json').open('x', encoding='utf-8') as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
arm['data_handoff'] = report['data_manifest']
arm['data_snapshot_independently_verified_round'] = 30
arm['recovery_reference_D_archive'] = report['backup_recovery_references']
arm['current_running_round_at_snapshot'] = report['live_training_round_at_snapshot']
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
note = (f'【本次实验与数据交付快照：2026-11-08】\n'
        f'30219上的NO_MSCA正在第{report["live_training_round_at_snapshot"]}轮；原生逐轮本地备份已接受至第{report["latest_transport_verified_round"]}轮，独立状态/轨迹核验至第30轮。'
        'Excel/JSON/CSV数据快照截至已独立核验的第30轮，逐轮明细合计990行。已完成的BASE/FULL/NO_MPS结果与第29轮旧数据快照保留；NO_MSCA完整终点待完成。\n'
        '文件入口：01_会议论文/实验数据汇总_20261108。训练与备份客户端继续运行，当前稿件仍第三十五轮，尚不能投稿。以下为此前记录。\n\n')
entrance = paper / '当前修订入口.txt'
entrance.write_text(note + entrance.read_text(encoding='utf-8'), encoding='utf-8')
print(json.dumps({k: report[k] for k in ['status', 'live_training_round_at_snapshot',
    'latest_transport_verified_round', 'independently_verified_round', 'data_snapshot_NO_MSCA_round']}, ensure_ascii=False))
