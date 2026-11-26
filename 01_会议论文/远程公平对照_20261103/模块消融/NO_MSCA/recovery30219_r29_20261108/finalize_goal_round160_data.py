"""Independent local data readback and monitoring handoff. No model or remote actions."""
import collections
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parents[1]
PAPER = HERE.parents[3]
DATA = PAPER / '实验数据汇总_20261108'
SNAPSHOT = DATA / 'verified_snapshots/NO_MSCA_round160_verified_20261109'
PREVIOUS = DATA / 'verified_snapshots/NO_MSCA_round140_verified_20261109'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, value, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


manifest = read(DATA / '交付清单.json')
snapshot = read(SNAPSHOT / 'snapshot_manifest.json')
assert manifest['status'] == 'COMPLETE' and manifest['NO_MSCA_accepted_rounds'] == 160
assert digest(DATA / '交付清单.json') == '69c1fc03b99c2e52734036cc55de09acbe0fb2042198fb5c5da7b0eb9fc9bf1e'
assert digest(SNAPSHOT / 'snapshot_manifest.json') == '0f6df84c5e6af4bb40cd93c17e11c066e6a3a832acc5975954122ca05bd7484c'
assert len(manifest['files']) == 18 and len(snapshot['files']) == 19
assert {v['file']: v for v in snapshot['files'] if v['file'] != '交付清单.json'} == {
    v['file']: v for v in manifest['files']}
for item in snapshot['files']:
    for folder in [DATA, SNAPSHOT]:
        path = (folder / item['file']).resolve(strict=True)
        assert path.is_relative_to(folder.resolve())
        assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256']
old = read(PREVIOUS / '交付清单.json')
assert digest(PREVIOUS / '交付清单.json') == '02f6c67207a9d8ff298ba7b45ca5a01363d14bb0e1a167fa18a7712d2b53b268'
for item in old['files']:
    path = PREVIOUS / item['file']
    assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256']
assert (DATA / '逐轮指标.csv').read_bytes().startswith((PREVIOUS / '逐轮指标.csv').read_bytes())
with (DATA / '逐轮指标.csv').open(encoding='utf-8-sig', newline='') as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 1120
assert collections.Counter(v['arm'] for v in rows) == {
    'BASE': 320, 'FULL': 320, 'NO_MPS': 320, 'NO_MSCA': 160}
assert [int(v['round_1based']) for v in rows if v['arm'] == 'NO_MSCA'] == list(range(1, 161))
audit_path = HERE / 'lineage_recovered_through_round160.json'
audit = read(audit_path)
assert audit['completed_rounds'] == 160 and audit['learning_rates']['recorded_attempts'] == 258720
assert audit['learning_rates']['maximum_absolute_difference'] == audit['maximum_latest_metric_error'] == 0
update = read(DATA / '第160轮更新核查.json')
assert update['new_evidence']['NO_MSCA_RECOVERED_R160'] == digest(audit_path)

observer = HERE / 'goal_monitor_20261109_002'
status = read(observer / 'observer_status.json')
assert status['status'] == 'REMOTE_WORKER_CONFIRMED_LIVE'
assert status['observer_pid'] == 6732 and status['native_accepted_round'] >= 160
assert status['backup_client_alive'] and status['training_control_actions'] == 0
live_copy = HERE / 'goal_round160_data_delivery_live_20261109.json'
with live_copy.open('xb') as handle:
    handle.write((observer / 'latest_live.json').read_bytes())
subprocess.run([sys.executable, '-X', 'utf8', '-B',
    str(HERE / 'record_recovery_observation.py'), live_copy.name], check=True)

state_path = BUNDLE / 'current_execution_state.json'
state = read(state_path)
arm = state['NO_MSCA']
assert arm['independent_lineage_audit_round'] == 160
assert arm['monitor_pid'] == 18924 and arm['monitor_session_id'] == 8984
assert state['goal_status'] == 'active'
stamp = datetime.now(timezone.utc).isoformat()
report = {
    'status': 'ROUND160_DATA_HANDOFF_INDEPENDENTLY_READ_BACK_AND_PINNED',
    'observed_local_utc': stamp, 'user_facing_local_date': '2026-11-09',
    'goal_objective': '监测', 'goal_status': 'active',
    'independently_verified_round': 160, 'target_round': 320,
    'native_accepted_round_at_record': arm['last_accepted_round'],
    'running_round_at_snapshot': arm['current_running_round_at_snapshot'],
    'live_snapshot': {'file': str(live_copy), 'sha256': digest(live_copy)},
    'lineage_audit': {'file': str(audit_path), 'sha256': digest(audit_path)},
    'data_rows': 1120, 'data_metric_cutoff_round': 160,
    'current_delivery_files_rechecked': 18, 'immutable_snapshot_files_rechecked': 19,
    'previous140_snapshot_file_hashes_rechecked': len(old['files']),
    'previous1100_rows_exact_byte_prefix': True,
    'snapshot_delivery_manifest': {'file': str(SNAPSHOT / '交付清单.json'),
        'sha256': digest(SNAPSHOT / '交付清单.json')},
    'snapshot_manifest': {'file': str(SNAPSHOT / 'snapshot_manifest.json'),
        'sha256': digest(SNAPSHOT / 'snapshot_manifest.json')},
    'workbook': {'file': str(DATA / 'outputs/实验数据汇总_20261108/实验数据汇总.xlsx'),
        'sha256': digest(DATA / 'outputs/实验数据汇总_20261108/实验数据汇总.xlsx')},
    'next_independent_audit_round': 200,
    'active_observer_exec_session': 84874, 'backup_client_exec_session': 8984,
    'model_forwards': 0, 'evaluator_runs': 0, 'remote_connections_from_this_script': 0,
    'training_control_actions': 0, 'paper_revision': 36,
    'paper_scientific_results_unchanged': True, 'submission_ready': False}
report_path = HERE / 'goal_round160_data_handoff_verified_20261109.json'
write(report_path, report, exclusive=True)
arm['data_handoff'] = {'path': str(SNAPSHOT / '交付清单.json'),
    'sha256': digest(SNAPSHOT / '交付清单.json'), 'data_rows': 1120,
    'NO_MSCA_verified_round': 160, 'scope': 'Immutable validated snapshot'}
arm['current_data_delivery_manifest'] = {'path': str(DATA / '交付清单.json'),
    'sha256': digest(DATA / '交付清单.json'),
    'scope': 'Continuous current delivery; historic hashes resolve via immutable snapshots'}
arm['data_snapshot_independently_verified_round'] = 160
arm['latest_data_snapshot_manifest'] = report['snapshot_manifest']
arm['latest_status_evidence'] = {'file': str(report_path), 'sha256': digest(report_path)}
state['observed_local_utc'] = stamp
write(state_path, state)
contract_path = HERE / 'active_monitor_goal_20261109.json'
contract = read(contract_path)
contract['latest_verified_checkpoint'] = arm['latest_status_evidence']
contract['latest_data_handoff'] = arm['data_handoff']
contract['next_independent_audit_round'] = 200
contract['latest_progress_classification'] = 'Independent160 audit,1120-row export and live monitoring toward200'
write(contract_path, contract)

note = (
    '【第160轮核验及数据更新：2026-11-09】\n'
    f'BASE、FULL、NO_MPS均已完成320轮。30219原NO_MSCA续训进程及逐轮备份客户端仍在；本次现场已接受{arm["last_accepted_round"]}轮，正在第{arm["current_running_round_at_snapshot"]}轮。'
    '独立状态/轨迹核验截止160轮，258720条学习率记录误差0，548图及保存预测/数组/指标绑定通过；本地审计未重新推理或执行评价器。\n'
    'Excel/CSV/JSON已更新至独立核验160轮，共1120条逐轮记录；前1100行字节前缀及已完成三组结果保持。160轮交付另存verified_snapshots/NO_MSCA_round160_verified_20261109，19项文件经主控逐字节回查；原140轮快照不变。完整NO_MSCA终点继续留空。\n'
    '监测目标active，下一独立核验节点200轮；只读观察器不控制训练。完成320轮、真实成功退出及最终备份后才做完整终点核验。稿件仍第三十六轮，尚不能投稿。以下为此前历史记录。\n\n')
for filename in ['当前修订入口.txt', '目标自主执行入口_20261109.txt']:
    path = PAPER / filename
    original = path.read_text(encoding='utf-8')
    assert not original.startswith('【第160轮核验及数据更新：')
    path.write_text(note + original, encoding='utf-8')
with (BUNDLE / '方案与执行记录.md').open('a', encoding='utf-8') as handle:
    handle.write('\n\n## 2026-11-09：监测与第160轮数据核验\n\n' + note)
print(json.dumps(report, ensure_ascii=False))
