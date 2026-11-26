"""Publish the verified exact-recipe decision while keeping the research goal active."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

P = Path(__file__).resolve().parent
B = P.parent
ROOT = B.parent.parent
ID = 'A56_LEVEL_MATCHED_CONTRAST_READ'


def load(p):
    return json.loads(p.read_text('utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(p, obj, mode='x'):
    with p.open(mode, encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    judgment_path = P/'fixed_endpoint_judgment_A56.json'
    j = load(judgment_path)
    assert j['decision'] == 'RETIRED_EXACT_RECIPE_FIXED12' and not j['all_gates_passed']
    audit_path = P/'terminal_frozen_input_audit_A56.json'
    audit = load(audit_path)
    assert audit['status'] == 'PASS' and all(not r['owned_live'] for r in audit['owned_processes'])
    checks = {a: load(P/f'{a}_completed_local_verification.json') for a in ['M','Q']}
    assert all(c['status'] == 'PASS' and c['optimizer_step_histogram'] == {'9708': 494} for c in checks.values())
    secondary = load(P/'secondary_recovered_rows_A56_v1.json')
    assert secondary['A56_total_cached_COCO_evaluate_calls'] == 3
    assert secondary['new_COCO_evaluate_calls_by_recovery'] == 0
    endpoints = j['endpoints']
    results = {a: load(P/f'completed_{a}_v1/arm_{a}/result.json') for a in ['M','Q']}
    b_result = load(B/'A49_下一主线有界筛选_20261101_v1/completed_B_v1/arm_B/result.json')
    latencies = {'B': b_result['latency']['median_ms'], **{a:r['latency']['median_ms'] for a,r in results.items()}}
    relative_latency = {a:(latencies[a]/latencies['B']-1)*100 for a in ['M','Q']}
    now = datetime.now(timezone.utc).isoformat()
    report = P/'14_固定12轮终点与裁决.txt'
    lines = [
        'A56：尺度内对比读出，固定12轮终点与裁决',
        '记录日期：2026-11-02；状态：本次具体配方停止。寻找期刊主线的总目标仍未完成。',
        '',
        '一、直接结论',
        '原版B、普通差分残差M、尺度内质量匹配残差Q的固定12轮EMA对比，Q未达到两项小目标收益门槛。本次不扩大训练、不增加seed，不通过调增益、层、点数或部分平衡继续挽救。',
        '该结论针对当前模型、数据和12轮配方；不证明所有对比注意力、所有编码器/解码器方法无效。普通M的小目标AP微增也不被重新命名为创新成立。',
        '',
        '二、三组终点（AP单位0–100；差值为百分点）',
        '组别 | AP | APs | AP75 | APs75',
    ]
    for a in ['B','M','Q']:
        e = endpoints[a]
        lines.append(f'{a} | {e["AP"]:.6f} | {e["APs"]:.6f} | {e["AP75"]:.6f} | {e["APs75"]:.6f}')
    lines += ['', 'Q−B：' + '，'.join(f'{k} {v:+.6f}pp' for k,v in j['Q_minus_B_pp'].items() if k in ['AP','APs','AP75','APs75']),
        'Q−M：' + '，'.join(f'{k} {v:+.6f}pp' for k,v in j['Q_minus_M_pp'].items() if k in ['AP','APs','AP75','APs75']),
        '', '三、原始资源门槛（未事后改变）',
        f'Q−B APs ≥ +0.5pp：实际{j["Q_minus_B_pp"]["APs"]:+.6f}pp，未通过。',
        f'Q−M APs ≥ +0.3pp：实际{j["Q_minus_M_pp"]["APs"]:+.6f}pp，未通过。',
        f'Q−B AP ≥ −0.2pp：实际{j["Q_minus_B_pp"]["AP"]:+.6f}pp，通过。',
        '这些是投入下一阶段的项目门槛，不是统计显著性或期刊接收标准。单seed、短训、反复用于开发的val均限制推广。',
        '', '四、协议与实现核验〔运行事实〕',
        '数据：VisDrone2019-DET train6471、val548；640直接拉伸，resize+hflip，无Mosaic。评估保留38602个GT、过滤157个有效ignore GT；maxDets300，原图裁剪面积small≤1024。属于明确的COCO风格口径，不能与官方VisDrone榜单直接等同。',
        'B复用已验证的原版12轮；M、Q从同一公共初始化fresh训练，新增参数初始逐tensor一致、RNG状态保持。优化器、增强、输入次序、监督与评价一致。',
        'M、Q均完成12轮、9708次正式更新、10159次整模型前向；494个优化器状态的step均9708，moment有限；四个BN计数9708。raw/EMA权重结构与新增12个参数tensor检查通过。',
        '12轮记录的全部目标/ignore/顺序输入摘要与B一致；源像素摘要仅覆盖每轮前8批，不能据此声称每张图的像素字节均逐一核对。',
        '冻结8个本候选文件、92个外部文件逐项SHA通过。原监督器及两个worker均已退出0，按PID+start_ticks确认不存活；后续队列为空。',
        '训练中所有原生参数均可学习。保留原生读取路径指结构和初始函数，不表示该路径训练期间数值不变。',
        'Q新增残差对独立每尺度常量value偏移的不变性只是固定query下的算子性质，不是整个检测器不变性、纯背景识别或任务增益证明。',
        '', '五、成本〔运行事实，非部署基准〕',
        '原版9803112参数；M、Q各9877256，新增74144。',
        f'记录的batch1/BF16/CUDA eager模型前向中位数：B {latencies["B"]:.3f}ms，M {latencies["M"]:.3f}ms（相对B {relative_latency["M"]:+.2f}%），Q {latencies["Q"]:.3f}ms（{relative_latency["Q"]:+.2f}%）。',
        '每组10次warmup+30次计时，排除decode/postprocess；顺序运行的短计时不能作为干净的板端/TensorRT因果延迟比较。',
        f'M训练峰值allocated {results["M"]["peak_allocated_bytes"]/(1<<30):.3f}GiB，Q {results["Q"]["peak_allocated_bytes"]/(1<<30):.3f}GiB，24GB单卡可运行；可运行不等于值得继续。',
        '', '六、预指定次要分层〔描述事实及缺口〕',
        '固定原图small且640输入短边[4,8)的10200个GT：',
    ]
    for a in ['B','M','Q']:
        r=secondary['arms'][a]['groups']['small_4to8']
        lines.append(f'{a}：R50 {r["recall50_percent"]:.6f}%（{r["matched50"]}/10200），R75 {r["recall75_percent"]:.6f}%（{r["matched75"]}/10200）。')
    lines += [
        '推断：这一预指定目标组没有给出Q改善漏检的支持；不能用个别类别、图片或有利分层替换全局主门槛。TP条件下尺寸误差排除了漏检，仅作描述，不作为定位机制因果结论。',
        '已按冻结上限做3次CPU缓存COCO评价、0次新模型前向/训练。各组计算的8项指标通过与原记录1e−8复现断言后，最终写JSON时numpy int32序列化失败；原错误、未完整result和GT_rows均保留。',
        '从已保存GT_rows恢复固定分层、逐图和逐类召回、尺寸误差，新增评价0次。逐类AP/APs未能完整落盘，明确缺失，不假装次要方案全项成功，也不为补齐而超出原3次评价上限。主裁决来自独立已验证的终点文件，不依赖该保存故障。',
        '', '七、最近工作与历史边界〔文献范围与推断分开〕',
        'Differential Transformer已提出双softmax差分；A35已阅读§2.1/2.2。M是本任务的简单差分对照，RMSNorm/lambda等与作者方法不同，不宣称复现原论文。https://arxiv.org/abs/2410.05258v2',
        'PiDiViT的中心差分/多尺度、PFE-Det的噪声/局部多路径、CDANet的跨差分注意力均是相关先例。各自明确阅读范围及原证据见08矩阵；不把摘要或局部方法当全文。',
        'RT-DETR+（10.1109/ICCASIT66611.2025.11348852）存在元数据与差分注意力相关索引线索，但未取得全文，不能判断逐算子重合已排除。https://doi.org/10.1109/ICCASIT66611.2025.11348852',
        'ECDet代码：https://github.com/Intellindust-AI-Lab/EdgeCrafter；固定commit d8752b75c9beaecfcdd0e5cd207bae3b6946ae23；是源码核查与本配方运行，不称整篇论文复现。',
        'D008是替换/去均值的冻结探针；D025是分层归一化接口未训练；本次保留原路径、学习新增残差并做匹配对照，实质差异已记录。该差异现在未通过任务收益门槛。D015纯背景假设不被继承。',
        '推断：目前不能把“尺度内常量抑制”写成经证实的小目标核心瓶颈；算子性质虽成立，12轮控制实验未显示额外价值。',
        '', '八、下一步',
        '封存A56具体配方，保留数据、源码、日志、预测、优化器、raw/EMA和失败输出。不继续调层/点数/增益/seed/epochs，也不把普通M重包装为新主线。',
        '下一候选先在少数实质不同机制之间核对旧编号与裁决、最近原文、具体操作差异及强简单对照。只有问题证据和方法差异足够支持一次有限判别，才固定下一份协议。当前没有已确认的新主线。',
        '', '证据入口：', str(judgment_path), str(audit_path),
        str(P/'M_completed_local_verification.json'), str(P/'Q_completed_local_verification.json'),
        str(P/'secondary_recovered_rows_A56_v1.json'), str(P/'08_A56核心文献与方法对照.json'),
    ]
    with report.open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')
    backup=P/'before_terminal_publish'
    backup.mkdir(exist_ok=False)
    names=['01_候选判别表.json','15_当前状态与结果校验.json']
    for n in names:
        shutil.copy2(B/n, backup/n)
    shutil.copy2(P/'CURRENT_A56.json', backup/'CURRENT_A56.json')
    terminal={'phase':'A56', 'status':j['decision'], 'recorded_utc':now,
              'goal_status':'active','mainline_validated':False,'novelty_cleared':False,
              'candidate_selected_for_pilot':False,'current_priority_candidate':None,
              'terminal_fixed12_verified':True,'counts_terminal':True,
              'owned_jobs_live':False,'post_Q_queue':None,
              'primary_gates':j['gates'],'all_gates_passed':False,
              'endpoints':endpoints,'Q_minus_B_pp':j['Q_minus_B_pp'],'Q_minus_M_pp':j['Q_minus_M_pp'],
              'report':str(report),'judgment':str(judgment_path),'terminal_audit':str(audit_path),
              'A56_formal_forwards':20318,'A56_capacity_forwards':9,
              'A56_formal_updates':19416,'A56_capacity_updates':6,'A56_formal_epochs':24,
              'CPU_cached_COCO_evaluate_calls':3,'secondary_status':secondary['status'],
              'secondary_missing':['per-class AP/APs'], 'no_automatic_rescue':True,
              'standalone_synthetic_attention_calls':7,'standalone_synthetic_component_steps':4,
              'new_detector_forwards_during_terminal_verification':0,
              'new_optimizer_updates_during_terminal_verification':0}
    save(P/'14_固定12轮终点与裁决.json',terminal)
    for n in names:
        d=load(B/n)
        d.update({'status':j['decision'],'goal_status':'active','mainline_validated':False,
                  'current_priority_candidate':None,'candidate_selected_for_pilot':False,
                  'mainline_selected_for_pilot':False,'latest_report':str(report),
                  'latest_update_utc':now,'A56_terminal_result':terminal,
                  'A56_Q_terminal_verification':{'status':'PASS','file':str(P/'Q_completed_local_verification.json'),
                                               'sha256':sha(P/'Q_completed_local_verification.json')},
                  'mainline_selected_for_pilot_scope':'A56 exact recipe retired; research goal remains active; next candidate unselected'})
        d['A56_execution'].update({'status':j['decision'],'report':str(report),'candidate_selected_for_pilot':False,
                                   'terminal_fixed12_verified':True,'owned_jobs_live':False,'counts_terminal':True})
        if n.startswith('01_'):
            row=next(r for r in d['candidates'] if r['id']==ID)
            row.update({'status':j['decision'],'selected_for_pilot':False,'decision':j['decision'],
                        'mainline_validated':False,'terminal_report':str(report),
                        'stop_reason':'Q−B APs +0.189212pp<+0.5; Q−M APs −0.060198pp<+0.3; no recipe rescue'})
        else:
            assert d['model_forwards']==d['cap']==226266
            assert d['training_updates']==194747 and d['formal_training_epochs']==240
            assert d['formal_detector_training_updates']==194686 and d['capacity_only_optimizer_updates']==61
            d.update({'latest_authoritative_phase':'A56','current_phase':'A56_TERMINAL_VERIFIED',
                      'current_native_job_terminal':True,'counts_terminal':True,'training_counts_terminal':True,
                      'counters_observed_lower_bounds_while_job_live':False,'full_forward_cap_closed':True,
                      'current_user_task_status':'research mainline unresolved; A56 exact recipe retired',
                      'CPU_saved_prediction_COCO_evaluate_calls_A56':3,
                      'full_model_forwards_scope':'exact205939 through A55 plus A56 capacity9+formal20318=226266; terminal',
                      'formal_training_epochs_scope':'216 through A55 plus A56 M/Q 12+12=240; temporary/component steps excluded'})
        save(B/n,d,mode='w')
    current=load(P/'CURRENT_A56.json')
    current.update(terminal)
    current.update({'selected_for_formal_training':False,'locally_verified_fixed12_arms':['M','Q']})
    save(P/'CURRENT_A56.json',current,mode='w')
    supplement=ROOT/'文献总库_20261101_v1/03_索引与核对/后续补链_A56固定12轮裁决.json'
    catalogue=load(ROOT/'文献总库_20261101_v1/03_索引与核对/文献总库.json')
    assert len(catalogue['records'])==423
    save(supplement,{'phase':'A56','scope':'experimental disposition supplement, original423-record catalogue unchanged',
                     'new_fulltext_PDFs':0,'linked_ids':['L153','L380','L046','L372'],
                     'report':str(report),'literature_matrix':str(P/'08_A56核心文献与方法对照.json'),
                     'decision':j['decision'],'goal_status':'active','novelty_cleared':False,
                     'read_scope_note':'use source-bound matrix; no human read attestation or new full-paper claim'})
    files=[report,P/'14_固定12轮终点与裁决.json',judgment_path,audit_path,
           P/'protocol_FROZEN_v1.json',P/'secondary_analysis_scope_A56_FROZEN_v1.json',
           P/'secondary_cached_failure_A56_v1.json',P/'secondary_cached_analysis_A56_v1/result.json',
           P/'secondary_cached_analysis_A56_v1/GT_rows.jsonl',P/'secondary_recovered_rows_A56_v1.json',
           P/'recover_secondary_from_saved_rows_A56.py',Path(__file__),supplement,P/'CURRENT_A56.json']
    for arm in ['M','Q']:
        files += [P/f'{arm}_completed_bundle_v1.zip',P/f'{arm}_bundle_ready.json',
                  P/f'{arm}_completed_local_verification.json',P/f'{arm}_bundle_transfer_verification.json']
    files += [B/n for n in names]+list(backup.iterdir())
    receipt={'status':'TERMINAL_PRESERVED','decision':j['decision'],'goal_status':'active',
             'new_detector_forwards':0,'new_optimizer_updates':0,
             'files':{str(f):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in files}}
    save(P/'A56_terminal_preservation_receipt.json',receipt)
    assert all(sha(Path(p))==r['sha256'] for p,r in receipt['files'].items())
    print(json.dumps({'status':'TERMINAL_PRESERVED','decision':j['decision'],
                      'report':str(report),'files_hashed':len(files),'goal_status':'active',
                      'A56_cached_COCO_evaluate_calls':3,'new_detector_forwards':0},ensure_ascii=False))


if __name__=='__main__':
    main()
