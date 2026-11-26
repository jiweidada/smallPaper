"""Record the completed simple control and verified startup of the candidate."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


if __name__ == '__main__':
    note = '16_M12终点与Q启动验收.txt'
    receipt = '17_M终点与继续验证回执.json'
    assert not (ROOT / note).exists() and not (ROOT / receipt).exists()
    raw = read('observer_23.json')
    assert raw['status'] == 'EXECUTED' and raw['exit'] == 0
    observed = json.loads(raw['stdout'])
    b, m, q = [observed['arms'][a] for a in ['B', 'M', 'Q']]
    verified = read('M_completed_local_verification.json')
    ready = read('M_bundle_ready.json')
    assert verified['status'] == 'PASS' and verified['arm'] == 'M'
    assert verified['endpoint_metrics'] == ready['endpoint'] == m['result']['endpoint_metrics']
    assert verified['optimizer_step_histogram'] == {'9708': 482}
    assert not b['identity']['present'] and not m['identity']['present']
    assert q['identity']['present'] and q['identity']['same_start_ticks'] and q['identity']['owned_command']
    assert q['progress']['status'] == 'TRAINING'
    initial = read('Q_startup/initialization.json')
    assert initial == read('completed_B_v1/arm_B/initialization.json') == read('completed_M_v1/arm_M/initialization.json')
    assert initial['all_initial_tensors_identical'] and initial['model_parameters'] == 9803112
    checks = observed['common_epoch_checks']['M_vs_B']
    assert len(checks) == 12 and all(row['all_pass'] for row in checks)
    delta = {k: m['result']['endpoint_metrics'][k] - b['result']['endpoint_metrics'][k]
             for k in ['AP', 'APs', 'APs75']}
    ast.parse((ROOT / 'judge_fixed_endpoint_A48.py').read_text(encoding='utf-8'))
    lines = [
        'A48：普通框像素加权对照M已完成，实例等量统计候选Q已启动。研究目标active，主线尚未确认。',
        '',
        '数据/宿主：VisDrone2019-DET train6471/val548，10类，ECDet-S。',
        '比较口径：固定第12轮EMA；COCO-style maxDets300、原图小目标面积、class0 union-IoA过滤；不是官方VisDrone分数。',
        '',
        '第12轮EMA（数值单位为百分比，差值为百分点）：',
        '组别\tAP\tAPs\tAPs75']
    for label, row in [('B原版', b), ('M普通框像素加权', m)]:
        metrics = row['result']['endpoint_metrics']
        lines.append(f"{label}\t{metrics['AP']:.8f}\t{metrics['APs']:.8f}\t{metrics['APs75']:.8f}")
    lines += [
        'M-B\t' + '\t'.join(f'{delta[k]:+.8f}' for k in ['AP', 'APs', 'APs75']),
        '',
        '原始观察：M在固定12轮未提高AP或APs，APs75有很小的正差值。',
        '推断：本次普通框像素加权对照未显示小目标AP收益；不能据此替代Q的结果。单seed短程差值也不支持统计显著性或普遍无效结论。',
        '',
        'M已正常退出0；完整结果包字节数/SHA：' + str(ready['bytes']) + ' / ' + ready['sha256'],
        '38个payload文件逐项SHA/字节数及CRC核验通过，包括第4/8/12轮权重、预测、GT、精度召回数组和日志；远端原件保留。',
        'CPU检查raw/EMA键、shape、dtype及非结构状态有限性通过；4个BN计数9708，482份AdamW参数状态step均9708。',
        'M前12轮与B的逐轮顺序、首8批实际输入、首8批前向前RNG、全部目标流SHA均一致。不声称全批输入张量或全批RNG均已逐项保存。',
        '以上核验不新增模型前向/优化，不是独立重跑AP。',
        '',
        'Q进程身份：' + json.dumps(q['identity'], ensure_ascii=False),
        'Q最近快照：' + json.dumps(q['progress'], ensure_ascii=False),
        'Q初始化与B/M完全一致（公共初始化SHA、seed、9803112参数及所有初始tensor检查通过）。Q未完成轮次，没有精度结果，不做收益判断。',
        'Q检验的是每个GT实例以单位空间总质量参与训练BN统计，相对于普通像素统计是否改善小目标检测；泛化的加权BN和等组贡献原则已有文献先例，不主张新BN公式。',
        '',
        '预先固定门槛继续保持：Q-B APs>=+0.5pp，Q-M APs>=+0.3pp，Q-B AP>=-0.2pp，三项均须满足。只用于决定资源投入，不代表显著性/新颖性/因果机制成立。',
        '三个完成包均核验后才运行 judge_fixed_endpoint_A48.py。该脚本当前仅做语法解析，尚未进行三组终点裁决；不会选最佳轮次。',
        '如失败，停止本配方，不搜索lambda/位置/seed救援；寻找研究主线的总目标仍未完成。',
        '本次没有启动新GPU任务；Q由此前已授权的B/M/Q固定监督队列启动。',
        '最新状态入口CURRENT_A48.json为可变快照；下一步操作前须重新查询。']
    with (ROOT / note).open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    names = [note, 'record_M_terminal_A48.py', 'judge_fixed_endpoint_A48.py',
             '15_M8同轮比较_中间结果.txt', 'M_completed_local_verification.json',
             'M_bundle_ready.json', 'M_bundle_ready_response.json', 'M_completed_bundle_download.json',
             'Q_startup/initialization.json', 'Q_startup_evidence_download.json',
             'observer_22.json', 'observer_23.json']
    output = {'status': 'PASS', 'goal_status': 'active', 'mainline_confirmed': False,
              'classification': 'progress_and_verified_wait', 'M_completed_exit0': True,
              'B_M_fixed12_deltas_pp': delta, 'M_completed_12_epoch_common_flow_checks': checks,
              'Q_live_identity': q['identity'], 'Q_observed_progress': q['progress'],
              'Q_initialization_identical_to_B_M': True,
              'new_separate_gpu_jobs': 0, 'model_forwards_added_by_verification': 0,
              'optimizer_updates_added_by_verification': 0,
              'observed_stage_counters_lower_bound': observed['observed_counters_lower_bounds'],
              'protocol_sha256': sha('protocol_FROZEN.json'),
              'files_sha256': {name: sha(name) for name in names}}
    with (ROOT / receipt).open('x', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({'status': 'PASS', 'goal_status': 'active', 'M12_deltas_pp': delta,
                      'Q_initialization_identical': True, 'Q_identity': q['identity'],
                      'note': str(ROOT / note)}, ensure_ascii=False))
