"""Record a genuine recovery observation; never infer progress from an old receipt."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import psutil

here = Path(__file__).resolve().parent
bundle = here.parents[1]
paper = here.parents[3]
parser = argparse.ArgumentParser()
parser.add_argument('snapshot')
parser.add_argument('--milestone', action='store_true')
args = parser.parse_args()
path = (here / args.snapshot).resolve(strict=True)
wrapper = json.loads(path.read_text(encoding='utf-8'))
assert wrapper['exit_code'] == 0
value = json.loads(wrapper['stdout'])
supervisor = value['records']['supervisor.json']
receipt = value['records']['run/receipt.json']
assert value['boot_id'] == '2e5d4a32-c510-4f86-8b5a-eb66f03b864a'
assert supervisor['supervisor_pid'] == 1779500 and supervisor['supervisor_start_ticks'] == 18097864
assert supervisor['pid'] == 1779501 and supervisor['start_ticks'] == 18098010
for key, expected_ticks in [('supervisor_pid', 18097864), ('pid', 18098010)]:
    assert value['processes'][key]['exists'] and value['processes'][key]['start_ticks'] == expected_ticks
process = psutil.Process(18924)
assert abs(process.create_time() - 1794207481.6664748) < .01
monitor_path = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA/backups/monitor_status.json')
monitor = json.loads(monitor_path.read_text(encoding='utf-8'))
assert monitor['status'] not in ['MONITOR_FAILED_NO_FURTHER_ACKS', 'MONITOR_STOPPED_BY_OPERATOR']
actual_receipt = receipt.get('pid') == supervisor['pid']
running_update = actual_receipt and receipt.get('status') == 'RUNNING' and receipt.get('optimizer_step_attempts', 0) > 46893
state_path = bundle / 'current_execution_state.json'
state = json.loads(state_path.read_text(encoding='utf-8'))
arm = state['NO_MSCA']
if arm.get('candidate_ssh_port') != 30219:
    arm['previous_interrupted_segment_identity'] = {k: arm.get(k) for k in [
        'candidate_ssh_port', 'candidate_gpu_uuid', 'remote_supervisor_pid', 'remote_trainer_pid',
        'remote_trainer_start_ticks', 'remote_boot_id', 'monitor_pid', 'monitor_create_time', 'monitor_session_id']}
    arm['previous_remote_live_observation'] = arm.get('last_remote_live_observation')
arm.update(status='RECOVERY_RUNNING' if running_update else 'RECOVERY_INITIALIZING_VALIDATING_FROZEN_INPUTS',
    candidate_ssh_port=30219, candidate_gpu_uuid='GPU-06ccf9c8-5f92-5aad-3458-e162506dd1dc',
    remote_supervisor_pid=1779500, remote_supervisor_start_ticks=18097864,
    remote_trainer_pid=1779501, remote_trainer_start_ticks=18098010,
    remote_boot_id=value['boot_id'], monitor_pid=18924, monitor_create_time=process.create_time(),
    monitor_session_id=8984, backup_monitor_terminated=False,
    remote_process_state='ORIGINAL_PROTOCOL_CONTINUED_ON_MATCHED_NEW_INSTANCE',
    pending_information=None, recovery_directory=str(here),
    recovery_addendum_sha256='e2db4f390756e33e742684e8f1eff526047403ba2db6bad17d5c9e21132e9d54',
    resumed_from_round=29, first_new_round=30)
arm['last_remote_live_observation'] = {'file': str(path), 'wrapper_sha256': sha256(path.read_bytes()).hexdigest(),
    'server_observed_utc': value['observed_server_utc'], 'boot_id': value['boot_id'],
    'supervisor_status': supervisor['status'], 'trainer_pid': 1779501, 'trainer_start_ticks': 18098010,
    'receipt_belongs_to_current_trainer': actual_receipt, 'receipt': receipt,
    'gpu': value['gpu'], 'gpu_processes': value['gpu_compute_processes']}
arm['current_running_round_at_snapshot'] = receipt['actual_loop_epoch'] + 1 if running_update else None
arm['last_local_backup_observation'] = {'observed_local_utc': datetime.now(timezone.utc).isoformat(),
    'monitor_pid': process.pid, 'monitor_create_time': process.create_time(), 'monitor_identity_alive': True,
    'monitor_status': monitor['status'], 'native_status': monitor}
if monitor['status'] == 'EPOCH_BACKUP_VERIFIED':
    arm['last_accepted_round'] = monitor['epoch'] + 1
    arm['last_acceptance_utc'] = monitor['observed_utc']
state['observed_local_utc'] = datetime.now(timezone.utc).isoformat()
state['operational_work_status'] = 'USER_AUTHORIZED_RECOVERY_IN_PROGRESS'
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
if args.milestone:
    assert running_update
    note = '''【NO_MSCA已在30219恢复续训：2026-11-08本地日期】
用户明确要求“跑实验，并且把数据整理一下”。已在30219从完整第29轮断点恢复，原始/EMA/优化器/预热及Python、NumPy、CPU/CUDA与数据加载器随机状态均恢复。新段从第30轮开始，计划补齐实际第320轮；配置、7021个输入文件字节、101项依赖和Torch/CUDA构建与原实验一致。
恢复附录单独绑定新GPU/boot、断点、恢复代码及剩余291次评价/470547次更新；原协议与冻结源码不改。整轮训练、评价、保存和回滚的AST与原版完全一致。失联后未保留的物理计算量不可确认，不声称完整训练未中断或物理计算相等。逐轮本地备份客户端已恢复，继续使用原nonce/manifest/ACK逐轮核验规则。
已完成实验另整理至01_会议论文/实验数据汇总_20261108，含Excel、CSV、JSON及来源哈希；NO_MSCA完整终点仍空缺。当前运行事实以模块消融/current_execution_state.json和D盘原生备份回执为准。
稿件仍为第三十五轮，尚不能投稿；目标工具上次blocked是旧记录，本次用户指令下的实际恢复工作正在进行。以下为历史记录。
'''
    entrance = paper / '当前修订入口.txt'
    entrance.write_text(note + '\n' + entrance.read_text(encoding='utf-8'), encoding='utf-8')
    with (bundle / '方案与执行记录.md').open('a', encoding='utf-8') as handle:
        handle.write('\n\n## 2026-11-08本地日期：第29轮断点恢复至30219\n\n' + note + '\n')
print(json.dumps({'status': arm['status'], 'first_new_round': 30, 'current_receipt_matches_trainer': actual_receipt,
                  'optimizer_step_attempts': receipt.get('optimizer_step_attempts'),
                  'last_accepted_round': arm['last_accepted_round'], 'backup_client_alive': True}))
