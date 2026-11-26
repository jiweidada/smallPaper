"""Archive a matched-epoch observation without making a final-gate decision."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, choices=[4, 8], required=True)
    parser.add_argument('--observer', required=True)
    args = parser.parse_args()
    epoch = args.epoch
    observer_path = (ROOT / args.observer).resolve()
    assert observer_path.parent == ROOT
    raw = read(observer_path)
    assert raw['status'] == 'EXECUTED' and raw['exit'] == 0
    observed = json.loads(raw['stdout'])
    q = observed['arms']['Q']
    assert q['identity']['present'] and q['identity']['same_start_ticks'] and q['identity']['owned_command']
    assert q['completed_epochs'] >= epoch
    metrics = {}
    paths = [observer_path, ROOT / 'record_Q_intermediate_A48.py']
    for arm in ['B', 'M', 'Q']:
        folder = ROOT / 'Q_intermediate' if arm == 'Q' else ROOT / ('completed_' + arm + '_v1') / ('arm_' + arm)
        path = folder / ('epoch%02d_EMA_metrics.json' % epoch)
        metrics[arm] = read(path)
        assert metrics[arm] == observed['arms'][arm]['evaluations'][str(epoch)]
        paths.append(path)
    keys = ['order_sha256', 'first_eight_actual_input_sha256',
            'first_eight_pre_forward_rng_sha256', 'all_batches_targets_sha256']
    first_two = []
    for e in [1, 2]:
        bp = ROOT / 'completed_B_v1' / 'arm_B' / ('epoch%02d_input_flow.json' % e)
        qp = ROOT / 'Q_startup' / ('epoch%02d_input_flow.json' % e)
        base, candidate = read(bp), read(qp)
        assert all(base[k] == candidate[k] for k in keys)
        calls = candidate['normalization_calls']
        assert len(calls) == 4 and all(v['weighted_training_calls'] == e * 809 - 1 for v in calls.values())
        first_two.append({'epoch': e, 'all_flow_checks_pass': True, 'weighted_calls_each_site': e * 809 - 1})
        paths += [bp, qp]
    epoch_checks = observed['common_epoch_checks']['Q_vs_B']
    assert all(next(r for r in epoch_checks if r['epoch'] == e)['all_pass'] for e in range(1, epoch + 1))
    delta = {a + '_minus_' + b: {k: metrics[a][k] - metrics[b][k] for k in ['AP', 'APs', 'APs75']}
             for a, b in [('M', 'B'), ('Q', 'B'), ('Q', 'M')]}
    number = 18 if epoch == 4 else 20
    note = ROOT / (str(number) + '_Q' + str(epoch) + '三组同轮比较_中间结果.txt')
    receipt = ROOT / ('Q_epoch%02d_intermediate_review.json' % epoch)
    assert not note.exists() and not receipt.exists()
    lines = [f'A48：第{epoch}轮三组EMA中间比较；主线尚未确认，目标active。', '',
             '数据/算法：VisDrone2019-DET，ECDet-S；同初始化/seed/数据流和评估口径。',
             '口径：COCO-style maxDets300、原图小目标面积及class0过滤；不是官方VisDrone分数。',
             'B：原版BN；M：普通框像素加权；Q：每个GT实例单位总质量的加权训练统计。', '',
             '同轮指标（百分比）：', '组别\tAP\tAPs\tAPs75']
    for arm, row in metrics.items():
        lines.append(arm + '\t' + '\t'.join(f'{row[k]:.8f}' for k in ['AP', 'APs', 'APs75']))
    lines += ['', '同轮差值（百分点）：']
    for pair, row in delta.items():
        lines.append(pair + '\t' + '\t'.join(f'{row[k]:+.8f}' for k in ['AP', 'APs', 'APs75']))
    lines += ['', '观察范围：以上只是中间结果，不是第12轮固定终点。不用早期门槛裁决，不挑最佳轮次，不和其他轮次混比。',
              'Q前两轮四个加权BN位置实际调用808/1617次，且与B的顺序、首8批实际输入、首8批前向前RNG和全批目标流哈希一致。',
              f'只读观察中Q前{epoch}轮共同数据流核查通过。不外推未完成轮次；不声称所有批次输入/RNG均有完整张量哈希。',
              '继续原定第12轮EMA裁决：Q-B APs>=+0.5pp，Q-M APs>=+0.3pp，Q-B AP>=-0.2pp三项均须满足。',
              '这些是资源分配门槛，不代表显著性、新颖性或因果机制证明。没有改lambda、位置、seed或训练长度，没有新开GPU任务。',
              '已有通用加权归一化和等组贡献先例，本候选只检验有区别的任务适配；不能因中间正差值宣称新BN原理。',
              '最新Q进度快照：' + json.dumps(q['progress'], ensure_ascii=False),
              '最新实时入口CURRENT_A48.json可变；下一步操作前须重新查询。']
    with note.open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    paths.append(note)
    output = {'status': 'INTERMEDIATE_EVIDENCE_RECORDED', 'epoch': epoch,
              'goal_status': 'active', 'mainline_confirmed': False, 'classification': 'progress_and_verified_wait',
              'metrics_percent': metrics, 'deltas_pp': delta,
              'Q_first_two_flow_and_weighted_call_checks': first_two,
              'Q_common_epoch_checks': [r for r in epoch_checks if r['epoch'] <= epoch],
              'Q_live_identity': q['identity'], 'Q_observed_progress': q['progress'],
              'final_gate_decision_made': False, 'new_gpu_jobs': 0, 'verification_model_forwards_added': 0,
              'files_sha256': {str(p.relative_to(ROOT)): sha(p) for p in paths}}
    with receipt.open('x', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({'status': output['status'], 'epoch': epoch, 'deltas_pp': delta, 'note': str(note)}, ensure_ascii=False))
