"""Recompute the human-requested Stem comparison from existing verified endpoints.

This reads local evidence only: no training, model forward, or checkpoint change.
"""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs' / 'stem_random_primary_comparison12_20261011_v1'
EXPECTED = {
    (1, 'native_stem_scratch'): '7b10e9d9f84cb4bdded70e121c8c55c578a1f56f0637e5cbae65e73b732d65e3',
    (1, 'original_stem'): '40a0c8f1326aa434f38cb29c47d77c6c830cd0071368188b05df056baf96304c',
    (2, 'native_stem_scratch'): 'd39f5cb2e25b306781e0dcd618ee874ef40262c342c9b9849010dcccdd34f023',
    (2, 'original_stem'): '84243698e78c7cdeb73be38d32254abb87321bfbfa545c663bc4edcee5f848ab',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    assert not OUT.exists(), 'Versioned report already exists; do not overwrite evidence.'
    rows = []
    sources = []
    for seed in (1, 2):
        arms = {}
        for arm in ('native_stem_scratch', 'original_stem'):
            base = ROOT / 'outputs' / f'stem_fixed12_seed{seed}_formal_collection12_20261011_v1' / f'seed{seed}_{arm}'
            verification = base / 'verification.json'
            assert sha(verification) == EXPECTED[seed, arm]
            d = read(verification)
            assert d['status'].startswith('PASS_')
            assert d['fixed_endpoint'] == {'epochs': 12, 'zero_based_epoch': 11, 'weights': 'EMA'}
            assert d['all548_stage_arrays_reloaded_exact'] is True
            runtime = base / 'extracted' / 'run' / 'runtime_start.json'
            parity = base / 'extracted' / 'run' / 'first100_parity.json'
            completion = base / 'extracted' / 'run' / 'training_completion.json'
            start = read(runtime)
            assert start['initialization']['exact_shared_tensors'] == 754
            assert start['initialization']['only_missing_prefix'] == 'backbone.stem.'
            assert read(parity)['all100_rows_exact'] is True
            assert read(completion)['completed_epochs'] == 12
            assert start['native_augmentation_and_supervision_weights_unchanged'] is True
            arms[arm] = {'metrics_percent': d['candidate_overall_percent'], 'initialization': start['initialization'], 'identity': d['identity']}
            sources.extend({'path': str(p), 'sha256': sha(p)} for p in (verification, runtime, parity, completion))
        baseline = arms['native_stem_scratch']
        candidate = arms['original_stem']
        for key in ('common_init_sha256', 'train_annotation_sha256', 'val_annotation_sha256', 'image_manifest_sha256', 'evaluator_binding'):
            assert baseline['identity'][key] == candidate['identity'][key], key
        assert baseline['initialization']['new_stem_parameters'] == 6474
        assert candidate['initialization']['new_stem_parameters'] == 11799
        delta = {k: candidate['metrics_percent'][k] - v for k, v in baseline['metrics_percent'].items()}
        rows.append({'seed': seed, 'baseline': baseline, 'enhanced_stem': candidate, 'delta_percentage_points': delta, 'APs_improves': delta['APs'] > 0, 'AP_non_decrease': delta['AP'] >= 0, 'passes_joint_metric_rule': delta['APs'] > 0 and delta['AP'] >= 0})

    contract = {
        'schema': 'human_stem_main_comparator_v1',
        'human_instruction': '你要和 没加预训练的权重 stem 的基线比较啊',
        'scope': 'Stem architecture experiments and their component ablations',
        'main_baseline_arm': 'native_stem_scratch',
        'main_baseline_definition': '原始 HGNetv2 Stem 随机初始化（包含其 BN 状态）；其余网络使用相同公共初始化。不是整网从零训练。',
        'candidate_requirement': '替换 Stem 随机初始化；非 Stem 初始张量保持相同，匹配种子、数据、训练配方及固定第12轮 EMA 评估。',
        'pretrained_stem_baseline_role': '辅助参照；不能替代公平的随机 Stem 主对照来判定 Stem 结构收益。',
        'existing_pretrained_stem_P2_rank_experiments': '另列为保留预训练 Stem 的 P2 实验；不得将其相对随机 Stem 的数值差解释为 Stem 改进收益。',
        'gate': {'both_seeds_APs_strict_improvement': True, 'both_seeds_AP_non_decrease': True, 'endpoint_epochs': 12, 'weights': 'EMA', 'best_epoch_substitution': False},
        'reuse_completed_random_stem_controls': True,
        'new_baseline_training_or_forwards': False,
        'source_report': str(OUT / 'report.json'),
    }
    report = {
        'status': 'VERIFIED_EXISTING_FIXED12_STEM_ENDPOINTS_COMPARED_TO_RANDOM_NATIVE_STEM',
        'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'comparison_contract': contract,
        'arm_name_explanation': '历史目录 original_stem 对应用户原有 EnhancedStemBlock；native_stem_scratch 才是原始原生 Stem 的随机初始化对照。',
        'paired_results': rows,
        'both_seeds_APs_improve': all(x['APs_improves'] for x in rows),
        'both_seeds_AP_non_decrease': all(x['AP_non_decrease'] for x in rows),
        'two_seed_joint_gate_passed': all(x['passes_joint_metric_rule'] for x in rows),
        'conclusion': '增强 Stem 的 APs 在两个种子均提升；seed1 总体 AP 下降，尚未证明两种子稳定提升总体精度。不能用带预训练 Stem 的基线否定该公平比较中的小目标收益。',
        'causal_limit': '整体 Stem 结构和参数量均改变；本比较不能单独确定高频、分支、注意力或其他组件是唯一原因。',
        'evidence_sources': sources,
        'new_training_epochs': 0,
        'new_model_forwards': 0,
        'historical_evidence_changed': False,
    }
    OUT.mkdir()
    write(OUT / 'report.json', report)
    write(OUT / 'comparison_contract.json', contract)
    lines = [
        'Stem 主对照修正：原始 Stem 不加载预训练，非 Stem 使用共同初始化。',
        '以下为已有固定第12轮 EMA 结果；两种子四个端点均通过保存权重独立重载与548张阶段数组一致核验。',
        '种子 | 随机原始 Stem AP | 增强 Stem AP | ΔAP | 随机原始 Stem APs | 增强 Stem APs | ΔAPs',
    ]
    for x in rows:
        b, c, d = x['baseline']['metrics_percent'], x['enhanced_stem']['metrics_percent'], x['delta_percentage_points']
        lines.append(f"{x['seed']} | {b['AP']:.6f} | {c['AP']:.6f} | {d['AP']:+.6f} | {b['APs']:.6f} | {c['APs']:.6f} | {d['APs']:+.6f}")
    lines.extend(['', report['conclusion'], '本轮未训练、未重跑基线、未执行模型前向。', report['causal_limit']])
    (OUT / 'Stem随机基线对比.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write(OUT / 'manifest.json', {'files': {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}})
    write(ROOT / 'outputs' / 'active_stem_comparison_contract.json', contract)
    print(json.dumps({'report': str(OUT / 'report.json'), 'both_seeds_APs_improve': report['both_seeds_APs_improve'], 'two_seed_joint_gate_passed': report['two_seed_joint_gate_passed'], 'deltas': [x['delta_percentage_points'] for x in rows]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
