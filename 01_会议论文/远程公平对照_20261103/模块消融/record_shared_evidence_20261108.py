"""Record the completed cross-condition audit without altering the manuscript."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

from check_backup_health import observe

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent
CURRENT_SHA = 'b3aae17cf12667f34797179f399dcfd608edfab724633f023921c9703b0d4cee'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


report_path = HERE / 'module_shared_conditions_verified_20261108.json'
report = read(report_path)
assert report['status'] == 'FROZEN_REMOVAL_CONDITIONS_AND_INITIAL_TENSORS_VERIFIED'
master = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
assert digest(master) == CURRENT_SHA
archive_proof = HERE / 'NO_MPS/frozen_inputs_verified_20261108.json'
assert read(archive_proof)['status'] == 'ALL_FROZEN_INPUTS_LOCALLY_ARCHIVED_AND_VERIFIED'
assert len(read(archive_proof)['added_missing_files']) == 231
previous_index = PAPER / '修订记录_20261103/主要结论证据索引/主要结论证据索引_第三十三轮.json'
note = (
    '模块条件与初始化补核 2026-11-08\n\n'
    'NO_MPS 的完整冻结输入包已从正在运行的原服务器归档并下载；248 个文件逐项核对 SHA-256，'
    '补齐 231 个本地缺失文件，已有 17 个文件完全一致且没有覆盖。源包为 66,424,696 字节，'
    'SHA-256 为 1d5d66f34c1ceceb0ef6433d417b58eb6009e4ebaaef48b59a16cf1b77526e85。'
    '训练监督器与训练进程 PID/启动时标在补档前后保持一致。\n\n'
    'NO_MPS 和 NO_MSCA 的 227 份共享源码及相关文件均与父 FULL 冻结协议同路径、同哈希。'
    '母包另有 AGENTS.md 和 requirements.txt，未纳入两臂输入；两组安装环境中的全部 101 项依赖版本、'
    'Python、Torch、CUDA、cuDNN 与 Torch build 则逐项完全一致。\n\n'
    '两臂展开配置仅改变数据/输出的物理路径并增加明确的 ablation 声明。去除这些已声明字段后与 FULL 配置相等。'
    '随机控制、worker 初始化、逐轮数据种子、输入哈希和评价等七个函数的 AST 与父运行代码相同。'
    '已绑定的真实首 100 批训练输入也与 BASE/FULL 及预检记录一致。\n\n'
    '直接读取冻结初始权重并逐张量比较，不构造或执行模型：NO_MPS 共 684 个张量，其中 40 个替换 stem 张量来自 BASE，'
    '其余来自 FULL；NO_MSCA 保留的 673 个张量均与 FULL 相同。Identity 移除路径通过冻结适配器和真实预检绑定；'
    '无参数 Lite 门的移除不能仅由 state_dict 键集推断。\n\n'
    '上述证据支持预先声明的条件移除范围，不等于完整结果已到齐。FULL 有已披露的恢复历史，两臂从第 0 轮开始，'
    '不能声称四组全程随机状态或总计算开销完全相同；也不支持交互效应、统计显著性或模块机制与容量效应的分离。\n\n'
    '英文稿仍为第三十三轮，SHA-256 ' + CURRENT_SHA + '，本次没有修改正文。'
    '两个模块运行仍待完整第 320 轮终点与全程轨迹核验，当前尚不能投稿。\n')
note_path = HERE / '模块共同条件与初始化补核_20261108.txt'
with note_path.open('x', encoding='utf-8') as stream:
    stream.write(note)
paths = [report_path, archive_proof, HERE / 'NO_MPS/frozen_archive_remote_20261108.json',
         HERE / 'NO_MPS/status_after_frozen_archive_20261108.json', note_path]
supplement = {
    'status': 'MODULE_INPUT_CONDITIONS_VERIFIED_OUTCOMES_PENDING',
    'observed_local_utc': datetime.now(timezone.utc).isoformat(),
    'manuscript_content_revision': 33, 'manuscript_sha256': CURRENT_SHA,
    'main_evidence_index': str(previous_index.relative_to(PAPER)),
    'main_evidence_index_sha256': digest(previous_index),
    'new_evidence_files': {str(path.relative_to(PAPER)): digest(path) for path in paths},
    'model_execution': False, 'submission_ready': False}
supplement_path = HERE / 'module_conditions_evidence_supplement_20261108.json'
with supplement_path.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(supplement, ensure_ascii=False, indent=2) + '\n')
state_path = HERE / 'current_execution_state.json'
state = read(state_path)
for arm in ['NO_MPS', 'NO_MSCA']:
    root = Path(state[arm]['backup_root'])
    health = observe(arm, state[arm], root)
    assert health['monitor_alive'] and health['status'] == 'EPOCH_BACKUP_VERIFIED'
    events = [json.loads(line) for line in (root / 'monitor_events.jsonl').read_text(encoding='utf-8').splitlines()]
    accepted = next(event for event in reversed(events) if event['status'] == 'EPOCH_BACKUP_VERIFIED')
    state[arm].update(last_accepted_round=accepted['epoch'] + 1,
                      last_acceptance_utc=accepted['observed_utc'],
                      cross_checked_initial_tensors=report['arms'][arm]['initial_tensors_directly_compared'])
state['NO_MPS']['all_frozen_inputs_locally_archived_and_verified'] = True
state['module_shared_conditions_evidence'] = str(report_path)
state['module_shared_conditions_evidence_sha256'] = digest(report_path)
state['module_conditions_evidence_supplement'] = str(supplement_path)
state['observed_local_utc'] = supplement['observed_local_utc']
live_path = HERE / 'NO_MPS/status_after_frozen_archive_20261108.json'
live = read(live_path)
assert live['exit_code'] == 0 and live['stderr'] == ''
remote = json.loads(live['stdout'])
arm_info = remote['arms']['no_mps_20261107_v1']
supervisor = arm_info['supervisor.json']
assert supervisor['status'] == arm_info['receipt']['status'] == 'RUNNING'
assert arm_info[str(supervisor['pid'])]['start_ticks'] == supervisor['start_ticks']
state['NO_MPS']['last_remote_live_observation'] = {
    'file': str(live_path), 'server_observed_utc': remote['observed_utc'],
    'supervisor_status': supervisor['status'], 'trainer_pid': supervisor['pid'],
    'boot_id': remote['boot_id'], 'receipt': arm_info['receipt'], 'gpu': remote['gpu']}
temporary = state_path.with_suffix('.conditions.tmp')
with temporary.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
os.replace(temporary, state_path)
addition = ('\n补充核验 2026-11-08：两个条件移除组的共享源码、完整依赖环境、展开配方及保留参数初值已交叉核实。'
            'NO_MPS 完整 248 文件输入包已补档并核对哈希。证据补充索引为“远程公平对照_20261103/模块消融/'
            'module_conditions_evidence_supplement_20261108.json”；正文仍为第三十三轮，模块终点仍待完成。\n')
for path in [PAPER / '当前修订入口.txt', PAPER / '修订记录_20261103/修订说明与未解决问题.txt']:
    old = path.read_text(encoding='utf-8-sig')
    first, remainder = old.split('\n', 1)
    path.write_text(first + addition + remainder, encoding='utf-8')
with (HERE / '方案与执行记录.md').open('a', encoding='utf-8') as stream:
    stream.write('\n\n## 2026-11-08 共同条件、初始张量及完整输入补档\n\n' + note + '\n')
assert digest(master) == CURRENT_SHA
print(json.dumps({'status': supplement['status'], 'manuscript_revision_unchanged': 33,
                  'NO_MPS_accepted_round': state['NO_MPS']['last_accepted_round'],
                  'NO_MSCA_accepted_round': state['NO_MSCA']['last_accepted_round']}))
