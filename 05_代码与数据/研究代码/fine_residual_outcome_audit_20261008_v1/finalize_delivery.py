"""Finalize local engineering evidence, not a method-effectiveness result."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INDEX = ROOT.parent.parent/'data'/'00_实验计划与证据索引'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, text):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


binding = json.loads((ROOT/'stage_binding.json').read_text(encoding='utf-8'))
for name, expected in binding['files_sha256'].items():
    assert sha(ROOT.parent/binding['directory']/name) == expected
report = json.loads((ROOT/'engineering_synthetic.json').read_text(encoding='utf-8'))
assert report['check_count'] == 26 and report['status'] == 'PASS_SYNTHETIC_OUTCOME_COUNTEREXAMPLES'
for name, expected in report['input_code_sha256'].items():
    assert sha(ROOT/name) == expected
files = [p for p in ROOT.iterdir() if p.is_file()]
receipt = {
    'created_at_UTC': datetime.now(timezone.utc).isoformat(),
    'status': 'OUTCOME_DAMAGE_AUDIT_IMPLEMENTED_SYNTHETIC_CHECKS_PASS',
    'only_main_candidate': 'fine_residual',
    'model_directory_unchanged': 'fine_residual_geometry_20261008_v5',
    'stage_directory_unchanged': binding['directory'],
    'substantive_change': 'Expose early geometry gain with native TP loss; fixed control-defined success/miss cohorts and original GT-area groups; continuous change counts prevent mean cancellation.',
    'synthetic_checks': 26, 'formal_six_run_execution_verified': False,
    'model_calls': 0, 'evaluator_calls': 0, 'optimizer_or_parameter_updates': 0,
    'real_dataset_arrays_read': 0, 'real_research_images_read': 0,
    'remote_accesses': 0, 'subagents': 0,
    'method_effectiveness_proven': False, 'goal_complete': False,
    'previous_goal_turn_classification': 'progress_authorized_remote_lookup_completed',
    'current_goal_turn_classification': 'progress_missing_outcome_damage_breakdown_implemented',
    'files_sha256': {p.name: sha(p) for p in files},
}
write_new(ROOT/'01_交付回执与SHA.json', json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
status = '''当前总审查补充：已实现早期几何与最终损伤的直接分解

本轮发现并补齐的是报告缺口，不是新增网络根因或实测收益。
既有逐GT数组可手工关联损伤，但原汇总未直接显示早期支持新增却丢失正式TP的交集，缺少按原GT面积和对照组成功/失败群体展开的阶段变化。
现已在独立补充工具中实现每个固定IoU门槛的4×4交叉表，明确损伤目标编号、连续IoU正负变化及分组结果；与原TP新增/损失计数自动核对。
主AP、评价规则、模型v5、三组双seed准备包和原阶段分析v1均不变。

自审边界：原GT面积分组只是筛选全面积原生TP，不能冒充COCO面积专属TP/APs/APm/APl；边界GT按原闭区间可重叠；群体按对照组固定，不按候选收益挑选。
所有伴随变化不等于因果中介。早期改善但正式TP丢失必须保留，平均IoU为零也可能同时包含改善与损伤。
26项纯合成工程检查通过，没有读取真实研究数组/图片，没有模型、评价器、优化器、远端或训练调用。
真实六组终点仍缺失，因此未执行正式端到端分析，未产生任何新的候选AP或机制有效性结果。

代码和说明：
'''+str(ROOT/'00_使用说明_早期改善与最终损伤配对.txt').replace('\\','/')+'\n'+str(ROOT/'01_交付回执与SHA.json').replace('\\','/')+'''

上一轮属于进展：已授权的两处原训练目录实际查找完成、指定范围未找到原逐图预测。
本轮属于进展：将原判定要求中未直接落实的损伤分解实现并通过合成反例检查。
完整研究目标未完成，不把工程进展替代成因识别或效果证据，不沿用已经解除的缓存评价/远端查找授权阻塞。
'''
write_new(INDEX/'单一候选当前总审查_20261008_v1'/'07_早期改善与最终损伤分析补充已实现.txt', status)
for name, expected in receipt['files_sha256'].items():
    assert sha(ROOT/name) == expected
print(json.dumps({'status': receipt['status'], 'synthetic_checks': 26, 'bound_files': len(files),
                  'delivery_receipt_sha256': sha(ROOT/'01_交付回执与SHA.json')}, ensure_ascii=False))
