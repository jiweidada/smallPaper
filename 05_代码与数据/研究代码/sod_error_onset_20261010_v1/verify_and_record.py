"""Independent box-oracle identities and a new-goal progress receipt."""
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/sod_error_onset_20261010_v1'
OLD = ROOT / 'outputs/small_object_encoder_decoder12_analysis_20261010_v1'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    report = read(OUT / 'report.json')
    for name, digest in read(OUT / 'analysis_manifest.json')['files'].items():
        assert sha(OUT / name) == digest, name
    errors = []
    for seed in (1, 2):
        with np.load(OUT / f'seed{seed}_box_component_oracles.npz', allow_pickle=False) as z:
            o = {k: z[k].copy() for k in z.files}
        wr, hr = o['width_ratio'], o['height_ratio']
        # Coincident centers: intersection is the product of the smaller sides.
        intersection = np.minimum(1., wr) * np.minimum(1., hr)
        center_iou = intersection / (1. + wr * hr - intersection)
        # Equal sizes: overlap shrinks by center displacement on each axis.
        intersection = np.maximum(0., 1.-np.abs(o['dx_over_GT_width'])) * np.maximum(0., 1.-np.abs(o['dy_over_GT_height']))
        size_iou = intersection / (2. - intersection)
        valid = o['valid_best_query']
        diffs = [float(np.max(np.abs(x[valid] - o[name][valid]))) for x, name in
                 ((center_iou, 'center_replaced_IoU'), (size_iou, 'size_replaced_IoU'))]
        assert max(diffs) < 1e-12, diffs
        for t in (.5, .75):
            assert np.array_equal(center_iou[valid] >= t, o['center_replaced_IoU'][valid] >= t)
            assert np.array_equal(size_iou[valid] >= t, o['size_replaced_IoU'][valid] >= t)
        errors.append({'seed': seed, 'GT': len(valid), 'max_abs_error_center_size': diffs})
    validation = {
        'material_passport': {'skill': 'academic-research-suite', 'workflow': 'experiment-agent',
                             'mode': 'run+validate', 'verification_status': 'ANALYZED',
                             'engineering_status': 'VERIFIED_SOURCE_HASHES_AND_INDEPENDENT_GEOMETRY'},
        'status': 'INDEPENDENT_ORACLE_GEOMETRY_IDENTITIES_PASS',
        'formula_checks': errors, 'raw_baseline_stage_files_verified': 1096,
        'same_query_and_maximum_IoU_reproduction': 'Exact over38759GT per seed, total77518GT.',
        'scientific_scope': 'Exploratory descriptive analysis. No causal intervention, new native AP evaluation or training efficacy claim.',
        'fallacy_scan': {'coverage': '11/11', 'items': {
            '1_Simpson': 'Both seeds, input-size groups and model arms are separate. No pooling rescues a failed gate.',
            '2_Ecological': 'GT counts are not COCO AP or image-level improvement.',
            '3_Berkson': 'All548 images retained. Failed-box component-oracle selection is explicitly conditional.',
            '4_Collider': 'Conditioning on missing geometry describes errors, not the causal effect of size.',
            '5_Base_rates': 'Denominators5096 under4 and26586 COCO-small are stated; original-area and input-pixel definitions differ.',
            '6_Regression_to_mean': 'Fixed12 endpoints reused, no best epoch or worst-image intervention selected.',
            '7_Survivorship': 'Fine and degraded negatives retained. No arm or failed engineering run hidden.',
            '8_Look_elsewhere': 'Exploratory follow-up after prior outcomes. All arms/groups/thresholds disclosed, no significance tests.',
            '9_Forking_paths': 'Protocol frozen before new computation, but existing results were already observed; not preregistered efficacy evidence.',
            '10_Correlation_causation': 'Absent early support does not identify a backbone/root cause. GT replacements are undeployable oracles.',
            '11_Reverse_causality': 'Stage order tracks observed output boundaries, not the optimization history or a unique causal mechanism.'}}
    }
    write(OUT / 'validation_report.json', validation)
    lines = [
        '新目标：定位可干预的问题，再验证小目标检测改进',
        '状态：ACTIVE；本次新增训练0轮、模型前向0次、optimizer更新0次。',
        '旧目标及P2 fine/degraded、ROI等负结果保持原始收档。本阶段使用新目录，不清零旧预算。',
        '',
        '本次实际完成：两个种子、548张完整验证图、每组38759GT，六套BASE12/fine12/degraded12保存输出的阶段追踪；1096份原始基线阶段数组SHA核验；77518个GT的最终最佳query和IoU精确重现。',
        '阶段顺序：编码器选出的初始框、第一层后pre框、最终框。三个输出边界没有覆盖骨干内部，不能确定具体卷积层根因。',
        '',
        '主观察：实际640输入短边<4像素，共5096GT，IoU>=0.75。以下是保存框的几何支持/原生all-area匹配按GT过滤计数，不是APs或COCO-small专门召回。',
        'seed | 三阶段均无合格框 | 早期有框而最终无框 | 最终有框但原生命中缺失 | 原生命中',
    ]
    for s in report['seeds']:
        r = s['arms']['baseline']['input_under4']['0.75']
        lines.append(f"{s['seed']} | {r['never_supported']} | {r['supported_early_but_lost_final']} | {r['final_geometry_without_native_TP']} | {r['native_all_area_TP_filtered_GT']}")
    lines += ['', '上述四项各自合计5096。早期没有合格几何占主要缺口，不能把这一群体的问题概括为decoder把好框普遍破坏或分类压分。',
              '同一最终几何最佳query的GT辅助分解：仅替换中心或仅替换尺寸，两种单独替换都不能达到IoU>=.75的失败目标为seed1 4435/5012、seed2 4503/5035。此oracle包含GT信息，不能部署，不能当救回AP或证明某种注意力有效。',
              '', '下一步候选问题：训练增强的缩小是否妨碍早期几何形成。先对训练数据做保持标签和源实例身份的受控增强比较，核对目标尺寸、完整目标曝光、裁切损失与密集监督的取舍，再决定正式配置。',
              '已有半幅source crop试验确实放大部分保留目标，却降低绝对保留目标数；不能只凭<4px比例下降就晋级。暂不直接训练该旧crop原型。',
              '若设计条件性尺度保护，必须有同等平均增强强度的随机控制；简单关闭Mosaic只算消融，不包装为创新。通用尺度感知增强已有历史先例，需核查差异。',
              '后续训练：复用两个同seed BASE12与共同初态；每个新增配置所有训练阶段累计<=12轮，固定第12轮EMA。两seed均APs严格提高且AP不降才过原生筛选；机制归因还需事前冻结的匹配对照。未通过不延长、不换最好轮。',
              '本次未启动新GPU训练；新目标继续进行。',
              '', '交付：report.json完整计数；stage_onset_counts.csv六套模型分层；seed1/2_box_component_oracles.npz逐GT数组；validation_report.json独立几何恒等式和11/11统计误区检查；protocol.json计算前冻结口径。']
    with (OUT / '新目标与首项诊断结果.txt').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    write(OUT / 'execution_state.json', {
        'goal_status': 'active', 'phase': 'EXPLORATORY_ERROR_ONSET_COMPLETED_NEXT_INTERVENTION_DESIGN',
        'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'new_training_epochs': 0, 'new_training_updates': 0, 'new_model_forwards': 0,
        'baseline_training_reused': True, 'accepted_new_mechanism': False,
        'prior_goal_unchanged': True,
        'next': 'Design and data-only validate scale-protection intervention versus matched stochastic augmentation control; freeze training configurations before authorized remote tests.',
        'implementation_files': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
        'max_epochs_per_new_configuration': 12,
        'endpoint': 'fixed12_EMA', 'seeds': [1, 2],
        'gate': 'Both independently APs>BASE12_APs and AP>=BASE12_AP; no best epoch or extension.'})
    write(OUT / 'delivery_manifest.json', {'files': {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}})
    print(json.dumps({'status': validation['status'], 'goal_status': 'active', 'new_training_epochs': 0,
                      'next_intervention_not_yet_validated': True}))


if __name__ == '__main__':
    main()
