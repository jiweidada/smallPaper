"""Preserve last verified training state when both observation links are lost."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'parallel_evidence'
LEDGER = ROOT / 'parallel_execution_ledger.json'
EVENT = ROOT / 'connection_observation_20261104.json'
LAST = EVIDENCE / 'last_confirmed_before_connection_loss_20261104.json'
SNAPSHOT_NAMES = {'BASE': ('primary_parallel_live.json', 'BASE_before_connection_loss_20261104.json'),
                  'FULL': ('secondary_parallel_live.json', 'FULL_before_connection_loss_20261104.json')}

def sha_bytes(data):
    return sha256(data).hexdigest()

def exclusive(path, data):
    with path.open('xb') as f:
        f.write(data)

assert not EVENT.exists() and not LAST.exists()
assert all(not (EVIDENCE / saved).exists() for _, saved in SNAPSHOT_NAMES.values())
old_bytes = LEDGER.read_bytes()
old = json.loads(old_bytes)
assert old['status'] == 'SPLIT_HOST_TRAINING_RUNNING'
assert all(item['status'] == 'RUNNING' for item in old['arms'].values())
for arm, (live, _) in SNAPSHOT_NAMES.items():
    assert sha_bytes((ROOT / live).read_bytes()) == old['snapshots'][arm]['sha256']

probe = subprocess.run([sys.executable, '-I', '-X', 'utf8', str(ROOT / 'inspect_ssh_banner.py')],
                       check=True, capture_output=True, text=True, timeout=20)
ports = json.loads(probe.stdout)
assert {p['port'] for p in ports} == {30166, 30989}
if any(p.get('banner', '').startswith('SSH-') for p in ports):
    print(json.dumps({'status': 'SSH_BANNER_AVAILABLE_RETRY_EXISTING_HELPER', 'ports': ports}))
    raise SystemExit(2)

exclusive(LAST, old_bytes)
retained = {}
for arm, (live, saved) in SNAPSHOT_NAMES.items():
    data = (ROOT / live).read_bytes()
    exclusive(EVIDENCE / saved, data)
    retained[arm] = {'path': str(EVIDENCE / saved), 'sha256': sha_bytes(data)}
now = datetime.now(timezone.utc).isoformat()
event = {'status': 'OBSERVATION_CONNECTION_LOST_TRAINING_UNVERIFIED', 'observed_local_utc': now,
         'last_verified_local_utc': old['updated_local_utc'],
         'last_verified_ledger': {'path': str(LAST), 'sha256': sha_bytes(old_bytes)},
         'last_verified_snapshots': retained, 'last_verified_arms': old['arms'],
         'helper_reconnect_result': 'Both existing SSH helpers reported SSHException while reading the SSH protocol banner; no new training process was launched.',
         'direct_banner_probe': ports,
         'interpretation': 'Loss of the observation channel does not establish whether either remote training process is still running.',
         'training_restarted': False, 'clock_or_network_configuration_changed': False,
         'model_executions': 0, 'evaluator_executions': 0}
exclusive(EVENT, (json.dumps(event, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
unknown = {'status': event['status'], 'updated_local_utc': now,
           'last_verified_local_utc': old['updated_local_utc'],
           'last_verified_ledger': event['last_verified_ledger'],
           'connection_observation': {'path': str(EVENT), 'sha256': sha_bytes(EVENT.read_bytes())},
           'arms': {arm: {'status': 'UNKNOWN_OBSERVATION_UNAVAILABLE', 'last_confirmed': item}
                    for arm, item in old['arms'].items()},
           'submission_ready': False,
           'authority_note': 'Unknown is an observation state, not a claim that training stopped. Reconnect to the original supervisors and PID/start-tick identities; do not launch or resume training.'}
tmp = LEDGER.with_name('parallel_execution_ledger.connection.tmp')
assert not tmp.exists()
exclusive(tmp, (json.dumps(unknown, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
os.replace(tmp, LEDGER)

entry = ROOT.parent / '当前修订入口.txt'
text = entry.read_text(encoding='utf-8-sig')
old_text = 'BASE/FULL仍在两台RTX3090训练，实时状态见远程公平对照_20261103/parallel_execution_ledger.json。'
assert text.count(old_text) == 1
entry.write_text(text.replace(old_text, 'BASE/FULL的最后确认记录及当前连接状态见远程公平对照_20261103/parallel_execution_ledger.json；观察连接中断不代表训练停止。', 1), encoding='utf-8')
note = ('\n2026-11-04 观察连接中断：两台原SSH会话均失效，重连时均未完成SSH标识交换；'
        '端口只读探测见connection_observation_20261104.json。最后一次经PID/启动ticks核验的本机接收时间为'
        + old['updated_local_utc'] + '，BASE/FULL分别完成' + str(old['arms']['BASE']['completed_epochs']) + '/'
        + str(old['arms']['FULL']['completed_epochs']) + '轮。当前远程训练状态未知，不能据此判断已停止或已完成。'
        '原始状态和两份快照已按哈希留存，当前ledger标记为观察不可用；重连后须核验原进程，不自动重启或恢复训练。\n')
for p in [ROOT / '操作恢复说明.txt', ROOT / '第二服务器并行执行_20261104.txt',
          ROOT.parent / '修订记录_20261103/目标验收与阻塞记录.txt']:
    with p.open('a', encoding='utf-8') as f:
        f.write(note)
print(json.dumps({'status': unknown['status'], 'last_verified_local_utc': old['updated_local_utc'],
                  'last_verified_epochs': {a: v['completed_epochs'] for a, v in old['arms'].items()},
                  'ports': ports}, ensure_ascii=False))
