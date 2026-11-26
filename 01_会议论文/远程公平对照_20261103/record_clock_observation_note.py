"""Append the bounded wall-clock observation to existing recovery records."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json

ROOT = Path(__file__).resolve().parent
RECORDS = ROOT.parent / '修订记录_20261103'
REPORT = ROOT / 'clock_observation_20261104.json'
MARKER = '【2026-11-04 时钟偏差观测】'

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

report = load(REPORT)
assert report['status'] == 'REMOTE_CLOCKS_BEHIND_LOCAL_AT_OBSERVATION'
assert report['model_executions'] == report['evaluator_executions'] == report['clock_changes'] == 0
assert report['training_settings_changed'] is False
for item in report['sources'].values():
    assert sha(Path(item['snapshot_path'])) == item['snapshot_sha256']
for key, filename in [('BASE_supervisor_sha256', 'supervise.py'),
                      ('FULL_supervisor_sha256', 'supervise_secondary_full.py')]:
    assert report['static_timeout_evidence'][key] == sha(ROOT / filename)
assert report['snapshot_freshness_evidence']['script_sha256'] == sha(ROOT / 'record_parallel_snapshot.py')
assert report['final_pair_audit_evidence']['script_sha256'] == sha(ROOT / 'summarize_split_pair.py')
targets = [ROOT / '操作恢复说明.txt', ROOT / '第二服务器并行执行_20261104.txt',
           ROOT / '结果并入与验收清单.txt', RECORDS / '目标验收与阻塞记录.txt']
old = [p.read_text(encoding='utf-8-sig') for p in targets]
assert all(MARKER not in value for value in old)
assert not (ROOT / 'clock_observation_note_receipt.json').exists()
note = ('\n' + MARKER + '\n'
        '本机时间2026-11-04 23:49附近的只读观测，用本机请求前后11.883秒包络两台远端快照。'
        'BASE远端时钟减本机时钟的界限为[-959.882, -947.999]秒；FULL为[-959.385, -947.502]秒，'
        '即此次两端均比本机约慢15分47秒至16分00秒。通信与调度延迟已包含在区间中。'
        '这只说明当次相对本机的时钟读数，不能当作NTP校准，也不能把该偏差直接套用于过去或将来的时间戳。\n'
        '源码与实际进程核对表明两组监督器以各自time.monotonic()计算604800秒硬超时；'
        '现行状态新鲜度判断使用本机接收文件的修改时间，远端UTC另行记录。'
        '最终配对核验以轮次、更新次数、保存权重及指标绑定为依据；这些检查不能替代最终结果审计，也不证明其他未检查过程不受时钟变化影响。\n'
        '观测和范围见clock_observation_20261104.json，SHA256：' + sha(REPORT) + '；两份原始快照及其哈希见该文件。'
        '最终配对审计还引用summarize_completed_pair.py，当前SHA256：' + sha(ROOT / 'summarize_completed_pair.py') + '。'
        '首次捕获前的本地脚本因次服务器监督器哈希常量多写一位，在创建包络或发送远端请求前被断言阻止；'
        '核对本地文件与远端快照一致后纠正常量，才完成本次观测。没有校时、改训练设置或执行新模型、评价器。\n')
for path, text in zip(targets, old):
    path.write_text(text + note, encoding='utf-8')
receipt = {'status': 'CLOCK_OBSERVATION_APPENDED', 'recorded_local_utc': datetime.now(timezone.utc).isoformat(),
           'observation_sha256': sha(REPORT),
           'target_paths_and_resulting_sha256': [{'path': str(p), 'sha256': sha(p)} for p in targets]}
with (ROOT / 'clock_observation_note_receipt.json').open('x', encoding='utf-8') as f:
    json.dump(receipt, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'status': receipt['status'], 'report_sha256': receipt['observation_sha256'],
                  'records_updated': len(targets)}))
