"""Close out the fixed 317 recipe. Standard-library evidence reads only."""
import sys, json, hashlib, importlib
from pathlib import Path
from datetime import datetime, timezone

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
WORK = HERE.parent
SPEC = WORK.parents[1] / '持续寻找研究主线_20261020_v2'
sys.path.insert(0, str(WORK / '348_cpu_training_20261023_v1'))
C = importlib.import_module('348_common')
VAL = WORK / '349_cpu_val_20261023_v1'
review = SPEC / '351_主线val548结果_主控核对.txt'

def write(name, document):
    C.write(HERE / name, document)

def main():
    seals = [C.verify_seal(C.HERE / '348_交付封存SHA256.json'),
             C.verify_seal(VAL / '349_交付封存SHA256.json')]
    totals = C.read(VAL / '349_正式计数双阈值与G更新披露.json')['formal_counts_totals']
    decision = C.read(VAL / '349_十二项判定_result.json')
    training = C.read(C.HERE / '348_训练_result.json')
    expected = {1: [149, 234, 146, 125], 2: [353, 634, 471, 391]}
    sources = {str(p): C.sha(p) for p in [Path(__file__), review,
        VAL / '349_正式计数双阈值与G更新披露.json',
        VAL / '349_十二项判定_result.json', C.HERE / '348_训练_result.json']}
    rows = []
    for seed in (1, 2):
        for index, arm in enumerate('NSGR'):
            run_id = f'{arm}_seed{seed}'
            one = totals[str(seed)][arm]
            assert one['0.5']['new_candidate_instances_input'] == 21920
            assert one['0.5']['new_candidate_instances_top300_selected'] == expected[seed][index]
            for threshold in ('0.5', '0.75'):
                assert one[threshold]['GT_added_vs_global_baseline'] == 0
                assert one[threshold]['B'] == 0
            run = next(r for r in training['runs'] if r['run_id'] == run_id)
            log_path = Path(run['run_dir']) / '347_steps.json'
            logs = C.read(log_path)
            assert len(logs) == 1218 and run['full_budget_completed'] is True
            sources[str(log_path)] = C.sha(log_path)
            curve = {key: {'first100_mean': sum(r[key] for r in logs[:100])/100,
                           'last100_mean': sum(r[key] for r in logs[-100:])/100}
                     for key in ('loss_det', 'loss_mass', 'loss_assoc')}
            rows.append({'run_id': run_id, 'top300_selected': expected[seed][index],
                         'final_selected': one['0.5']['new_candidate_instances_final_selected'],
                         'added_unique_GT_IoU05': 0, 'added_unique_GT_IoU075': 0,
                         'lost_unique_GT_IoU05': one['0.5']['GT_lost_vs_global_baseline'],
                         'loss_first100_vs_last100': curve})
    directions = decision['directions12']
    sign_counts = {'positive': sum(r['delta'] > 0 for r in directions),
                   'zero': sum(r['delta'] == 0 for r in directions),
                   'negative': sum(r['delta'] < 0 for r in directions)}
    assert sign_counts == {'positive': 3, 'zero': 8, 'negative': 1}
    record = {'status': '本轮317固定配方不保留，停止',
        'user_decision_date': '2026-10-25', 'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'decision_authority': '本聊天用户明确裁定及351主控核对原文',
        'scope': 'development3235、6轮、1218步、固定末轮、N/S/G/R×两seed、本机CPU、val548',
        'direction_counts': sign_counts, 'run_evidence': rows,
        'G_delta': {s: {'delta': g['delta'], 'verdict': g['verdict']}
                    for s, g in decision['G_effect'].items()},
        'scientific_statement': '本固定配方未产生统一后处理后的新增unique GT，未观察到接口差异带来的补全收益，未达到预注册保留条件。',
        'inference_limits': [
            '不能由本结果推出单位解释接口普遍无效。',
            '损失曲线相近不能证明读取已有框对训练没有影响，也不是等效结论。',
            '微小AP差异不作补全收益证据；其是否纯浮点误差未核。',
            '加步数的效果未核；本轮不加轮、不改阈值或标签、不换臂救结果。',
            '本记录的新增GT指相对global baseline的新增unique GT，不扩大为对每个未筛选新框几何质量的诊断。'],
        'next_work_status': {'optional_score_free_IoU_diagnostic': '未执行，留待单独指令',
             'C_line_P2_to_P3': '用户建议，未启动；本轮未核其正结果与缺口',
             'GPU_or_remote_training': '未执行，未获本轮启动指令',
             'aspect_ratio_paired_arm': '用户建议仅附属于未来C线重训，本轮未启动'},
        'new_execution_counts': {'model_forwards': 0, 'optimizer_steps': 0,
                                 'heldout_reads': 0, 'new_evaluations': 0},
        'previous_seals_verified': seals, 'source_sha256': sources}
    write('352_阴性停止交接_result.json', record)
    text = ['# 352｜317固定配方阴性停止记录', '',
        '用户裁定日期：2026-10-25。依据351主控核对，本轮固定配方不保留，停止。', '',
        '八run均完成1218步；12项方向为3正、8零、1负。所有臂在IoU0.5和0.75的统一后处理新增unique GT均为0，主B均为0。',
        'G Δ为0.0055634058182583235和0.0024468467339710163，均未有效校准；R−G四项不得支持R。', '',
        '| run | 新候选进top300 | 最终保留 | 新增GT@0.5 | 新增GT@0.75 | 丢失GT@0.5 |',
        '|---|---:|---:|---:|---:|---:|']
    text += [f"| {r['run_id']} | {r['top300_selected']} | {r['final_selected']} | 0 | 0 | {r['lost_unique_GT_IoU05']} |" for r in rows]
    text += ['', '第一和最后100步损失均值已逐run从原日志重算，见result JSON。', '',
        '结论限于本固定预算和配方：未观察到接口差异带来的补全收益。不能推出接口普遍无效，也不能由相近曲线证明读取已有框对训练没有影响。微小AP差异不作补全收益证据，是否纯浮点误差未核。', '',
        '保留348训练包、349评估包和351主控原文。本轮不加轮、不改阈值/标签、不换臂。',
        '可选的无分数IoU诊断未执行。C线汇总、UAVDT、GPU及保持宽高比配对臂均留待单独指令；本记录不核定C线效果或自动启动后续实验。', '',
        '本次仅核验既有封存、读取JSON日志并记录裁定；新增模型前向0、优化器步骤0、heldout读取0。', '']
    C.text(HERE / '352_阴性停止交接.md', '\n'.join(text))
    print(json.dumps(C.seal_tree(HERE, '352_交付封存SHA256.json'), ensure_ascii=False))

if __name__ == '__main__':
    main()
