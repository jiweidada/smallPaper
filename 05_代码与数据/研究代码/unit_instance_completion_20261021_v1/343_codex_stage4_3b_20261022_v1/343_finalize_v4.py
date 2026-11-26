"""343 handoff v2: stage4 done; stage3b fails one frozen nonpositive fullP row."""
import importlib,json,ast,time
C=importlib.import_module('343_context')
def read(name):return json.loads((C.HERE/name).read_text(encoding='utf-8'))
def link(path):return f'[{C.Path(path).name}]({C.Path(path).as_posix()})'
def main():
    start=time.perf_counter();d=read('343_result_3b已生成6469条独立验收.json');stats=read('343_result_3b完整目标数量统计补充.json')
    smoke=read('343_result_3b新标签四臂冒烟与续跑.json');bad=read('343_result_3b非正fullP全量排查.json')
    formal=read('343_result_阶段4正式计数_v2.json');positive=read('343_result_阶段4主B阳性构造_v2.json')
    boot=read('343_result_阶段4自然val配对bootstrap.json');history=read('343_result_阶段4历史AP精确对照.json')
    assert d['validated_entries']==6469 and len(bad['bad_entries'])==1 and boot['replicates']==1000
    assert formal['low_new4_formal_counts_equal_baseline_both_seeds']
    protected=read('343_result_旧缓存与源码不变核对.json')['entries']
    checked=[{'path':r['path'],'expected_sha256':r['expected_sha256'],'actual_sha256':C.sha(r['path'])} for r in protected]
    firstseal=read('343_交付封存SHA256.json')
    for p,r in firstseal['artifacts'].items():checked.append({'path':p,'expected_sha256':r['sha256'],'actual_sha256':C.sha(p)})
    assert all(r['expected_sha256']==r['actual_sha256'] for r in checked)
    py=list(C.HERE.rglob('*.py'))
    for p in py:ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    C.write('343_result_本轮不覆盖与语法核对_v2.json',{'status':'已验证','checked_files':len(checked),'entries':checked,
        '已执行':{'protected_file_SHA_comparisons':len(checked),'Python_AST_parses':len(py)},
        '未执行':['任何旧文件覆盖','真实训练','阶段5'],'未核':['主控独立核对'],'source_sha256':C.sources()})
    stage3_evidence=[C.HERE/n for n in ('343_追加放行与计数口径.json','343_result_3b来源预检.json',
        '343_result_3b构建_seed1_first16.json','343_result_3b构建_seed1_full_after_source_release.json',
        '343_result_3b构建_seed2_full_after_source_release.json','343_result_3b构建_seed2_collect_valid_after_failed1466.json',
        '343_result_3b已生成6469条独立验收.json','343_development6469_逐条SHA清单.json',
        '343_result_3b完整目标数量统计补充.json','343_result_3b新标签四臂冒烟与续跑.json',
        '343_result_3b非正fullP全量排查.json','343_result_3b非正框独立几何与PNG.json')]
    stage4_evidence=[C.HERE/n for n in ('343_result_阶段4映射与融合_v3.json','343_result_阶段4全量评估.json',
        '343_result_阶段4历史AP精确对照.json','343_result_阶段4Geo0资格.json','343_result_阶段4正式计数_v2.json',
        '343_result_阶段4主B阳性构造_v2.json','343_result_阶段4bootstrap.json','343_result_阶段4自然val配对bootstrap.json',
        '343_result_阶段4方向规则.json','343_result_阶段4合成PNG显示修正.json')]
    C.write('343_阶段3b_result_v2.json',{'status':'实现失败（6470条中1条冻结fullP非正框）；其余6469条已独立验证',
        'supersedes':'此前来源待主控已获接受；旧result保留，当前状态以v2为准。',
        'expected_entries':6470,'created_and_validated_entries':6469,'failed_missing_entries':bad['bad_entries'],
        'all_created_non_M0_fields_bit_equal_to_old340':True,'same_window_seeds':d['same_window_seeds'],
        'full_target_count_stats':stats['stats_full3235_each_seed'],'empty_G_W_completion_basis':stats['basis_for_missing_entry'],
        'old340_stats':d['old340_stats'],'M0_difference_scope':'seed1全3235；seed2仅3234条有新M0。#1466的旧跨类owner为0，完整旧跨类计数仍可相加。',
        'M0_differences':{s:{k:d['stats'][s][k] for k in ('owner_changed_images','U_W_changed_images','old_cross_class_owners','old_cross_class_owners_in_G_W')} for s in ('1','2')},
        'resume_cache_test':'seed1先建16后恢复验证跳过16；seed2失败后恢复验证跳过723；坏条目明确失败而非无声跳过。',
        'new_label_smoke_and_resume':smoke,'png_paths':d['png_paths'],
        'visual_QA':'6张development PNG已目视；seed1#12 U_W由1到0，seed2#13由2到1；缓存crop和独立crop位置相同；#5/#3空G_W无K_g红圈。',
        '待主控':['seed2 development#1466 query62负高框已进入fullP；当前317要求非正框报错，必须裁定该冻结输入的处理。未过滤、排序修复、取绝对值、修改score或重跑global。'],
        '已执行':{'successful_M0_builds':6469,'failed_build_entries':1,'raw_fullP_audit_entries':6470,
            'independent_M0_certificates':6469,'independent_cost_matrices':6469,'independent_crops':6469,
            'smoke_optimizer_steps':160,'resume_optimizer_steps':32,'checkpoint_saves':8,'checkpoint_loads':8,
            'head_forwards':194,'global_model_loads':0,'global_forwards':0},
        '未执行':['#1466的新M0/证书','完整6470条缓存通过','真实训练','GPU','阶段5'],
        '未核':['主控独立核对','真实训练输入绑定；本轮通过显式加载新缓存调用train342，未改其旧load_entry'],
        'evidence_sha256':{str(p):C.sha(p) for p in stage3_evidence},'source_sha256':C.sources()})
    C.write('343_阶段4_result_v2.json',{'status':'已验证（开训前评估链路工程）；待主控独立核对',
        'supersedes':'此前正式计数待主控已获自行决定授权；口径已写盘冻结；旧result保留。',
        'checklist':[{'item':i,'status':'已验证','evidence':str(stage4_evidence[j])} for i,j in (
            ('1 融合、双seed反变换、独立/往返/反向测试',0),('2 记录配置与Windows绑定源码评估器12项统计',1),
            ('3 B、全GT Geo0、.75、FP、TP新增丢失、候选入选挤出',4),('4 1000次548图共享索引配对bootstrap',7),
            ('5 十二项方向规则的五类构造',8),('6 双seed raw baseline与历史AP差异披露',2),('7 低分和随机new4端到端',1))],
        'policy_record':str(C.HERE/'343_追加放行与计数口径.json'),'policy_supplement':str(C.HERE/'343_计数口径补充冻结.json'),
        'baseline_AP_history':history['per_seed'],'natural_synthetic_counts':formal['summary'],
        'B_positive_constructed':[{'case':r['case'],'B':r['metrics']['B']} for r in positive['cases']],
        'bootstrap_source_ID_note':boot['ID_handling'],'bootstrap_native_accumulator_proofs':len(boot['native_accumulator_proof']),
        'bootstrap_fresh_clone_proofs':len(boot['fresh_full_clone_proof']),
        'evaluator_limit':'本机Windows.pyd与receipt Linux.so不同；using_tiny_metrice/using_vtiny_metrice均False、max_dets100与resolved一致。',
        'visual_QA':{'viewed_natural_formal_PNG':read('343_result_阶段4合成PNG显示修正.json')['png_paths'],
            'viewed_B_positive_PNG':positive['png_paths'],'display_fix':'只分离绘图面板，原始候选坐标/匹配/指标未裁剪。'},
        '已执行':{'natural_primary_matches':6576,'natural_independent_matches':6576,'natural_bootstrap_replicates':1000,
            'natural_vectorized_AP_APs_recomputations':6000,'natural_B_resampled_sums':6000,
            'natural_bootstrap_native_sampled_accumulate_proofs':18,'natural_bootstrap_fresh_full_clone_proofs':2,
            'positive_B_constructed_cases':6,'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
        '未执行':['任何训练产物评估','八run科学方向判定','真实训练','GPU','阶段5'],
        '未核':['主控独立核对','baseline历史AP差异因果','Windows与Linux二进制一致性'],
        'evidence_sha256':{str(p):C.sha(p) for p in stage4_evidence},'source_sha256':C.sources()})
    rows=[]
    for s in ('1','2'):
        old=d['old340_stats'][s];new=stats['stats_full3235_each_seed'][s]
        rows.append(f'| {s} | {old["U_W_positive"]["count"]} → {new["U_W_positive"]["count"]} | {old["U_W_gt4"]["count"]} → {new["U_W_gt4"]["count"]} | {old["overflow_GT"]} → {new["overflow_GT"]} | {new["total_U_W_GT"]} |')
    C.text('343_阶段3b_中文汇总_v2.md','''阶段3b：实现失败，单一冻结输入违反非正框合同；其余6469条已生成并独立验收。

来源链已由用户接受。复用340的final300，无新global前向。seed1完成3235条，seed2完成3234条。合法条目的非M0字段全部与旧340逐位相同；窗、GT、fullP、K_g、crop、M0合法边/最大基数/精确最优证书均做独立核对，另有FP64代价重算。6张PNG已看，每seed含U_W变化图。冻结1e-6容差保持。

| seed | U_W>0旧→新 | 溢出图旧→新 | 溢出GT旧→新 | 新U_W总数 |
| --- | --- | --- | --- | --- |
'''+ '\n'.join(rows)+'''

两seed同窗1443/3235不变。旧跨类owner608/600，其中G_W内74/76；合法条目U_W变化643/600图。Wilson及逐图差异见result。seed2缺失条目的G_W独立核为0，按定义任何M0下U_W必为0，因此上表目标数量可完整按3235计；该条M0仍未建成。

两seed各前32窗，四臂各20步loss/梯度有限；8组连续2步与1+保存恢复+1的模型、optimizer、RNG逐位相同。G十类κ_c均19/20步非零梯度。160步冒烟+32步续跑都是有界CPU工程测试，不作学习结论；阶段2旧标签仅seed1可并列。global模型本次未构造，旧state合同未冒称重测。

唯一阻塞：seed2 development#1466、query62、class3、score0.012048937，raw h=-0.00255471468，确已进入fullP且与339逐位同。317要求非正框报错；保留实现失败，未过滤/裁剪/取绝对值或重跑模型。该条有独立坐标对比、反向拒绝和6个目视倍率。待主控裁定后才可完成6470条缓存；阶段5与真实训练未执行。
''')
    ap=' / '.join(f'{r["cached_raw_AP"]:.12f}' for r in history['per_seed'])
    C.text('343_阶段4_中文汇总_v2.md',f'''阶段4评估链路工程已验证，等待主控独立核对。没有评估任何训练产物。

双seed共1096条val v2缓存通过共享反变换、独立实现、往返/反向测试；批准几何原文保留。全程只用Windows绑定源码评估器，设置False/False/max_dets100与resolved一致；.pyd与Linux.so差异保留披露。

正式计数口径已按用户授权决定并冻结：最终输出按同类score降序、并列flatten索引，对全图GT作一对一IoU≥.5/.75匹配；优先最大IoU，并列annotation_id。完成匹配后筛Geo0/tiny/完整窗组；新增/丢失相对同seed统一后处理global-only。tiny严格使用_all并对应annotation_id，完整窗1e-6容差不变。两种实现共6576次自然匹配一致；新增覆盖阳性、重复、错类、并列、先全GT后分组和挤出构造均通过。

baseline AP实测{ap}，未复现历史约.16603/.16409；未调配置。resolved启用EMA且当前本机solver优先EMA，但历史身份及偏差因果未核，未切换EMA。两个seed低分new4的12项COCO和正式TP/FP/B计数均与baseline相同；随机new4可计算。随机B为0只属合成输入结果，不作科学结论。

1000次548图配对bootstrap使用同一抽样矩阵，seed2026102103；重复图赋新ID，逐重采样数据集重算AP/APs和B。6000次自然冻结baseline/合成输出重算完成；18组与绑定native accumulate逐位相同，2组独立完整克隆重评逐位相同。另有1000次构造验证。GT正ID重编号只用于重采样，避免0号与未匹配哨兵混淆；原baseline统计未改。

十二项严格方向规则的五类构造已验证，真实八run判定未执行。PNG显示只限制面板绘制，数值不裁剪；6张自然合成和6张B构造均已看。结果入口为343_阶段4_result_v2.json。阶段3b仍因#1466失败；不进入阶段5，无真实训练/GPU/SSH/队列。
''')
    lines=['最新交接：阶段4工程已验证；阶段3b因单一非正fullP框实现失败。停下等主控核对，不进入阶段5。','',
        '本页替代旧343_交接索引.md中的待主控状态；旧文件与封存保持原样。用户已接受来源链并授权自行冻结计数口径。','',
        '本轮结果与一页汇总：','']
    for name in ('343_阶段3b_result_v2.json','343_阶段3b_中文汇总_v2.md','343_阶段4_result_v2.json','343_阶段4_中文汇总_v2.md'):
        lines.append('- '+link(C.HERE/name))
    lines+=['','活动实现与旧版本界限：'+link(C.HERE/'343_活动实现与失败产物索引.md')+'。','','唯一阻塞：seed2 development#1466 query62负高，已进入fullP。当前317要求非正框报错，未自行改变候选定义。G_W=0只允许补齐数量统计，不补造M0或证书。','',
        '主控优先核对：','']
    for p in (C.HERE/'343_result_3b非正fullP全量排查.json',C.HERE/'343_result_3b非正框独立几何与PNG.json',
        C.HERE/'343_result_3b已生成6469条独立验收.json',C.HERE/'343_result_3b完整目标数量统计补充.json',
        C.HERE/'343_result_3b新标签四臂冒烟与续跑.json',C.HERE/'343_result_阶段4正式计数_v2.json',C.HERE/'343_result_阶段4主B阳性构造_v2.json',C.HERE/'343_result_阶段4自然val配对bootstrap.json'):
        lines.append('- '+link(p))
    lines+=['','6张development PNG（两seed均含标签变化图）：','']
    lines.extend('- '+link(p) for p in d['png_paths'])
    lines+=['','6张非正框图为同一坏条目的6个显示倍率；不是6个坏条目。','']
    lines.extend('- '+link(p) for p in read('343_result_3b非正框独立几何与PNG.json')['png_paths'])
    lines+=['','阶段4数值证据：','']
    lines.extend('- '+link(p) for p in stage4_evidence)
    lines+=['','当前新缓存逐条SHA清单保留失败条目说明，禁止当作6470条完整通过：'+link(C.HERE/'343_development6469_逐条SHA清单.json')+'。',
        '',f'旧文件及先前交付{len(checked)}项SHA未变；源文件语法解析{len(py)}份通过。所有global前向为0，未改科学阈值、窗、标签定义。',
        '', '正式训练输入绑定尚未做；本轮冒烟显式加载新缓存再调用train342，未改变旧load_entry。',
        '', '最终封存：'+link(C.HERE/'343_交付封存SHA256_v2.json')+'。没有生成342_开训前审查包.md，没有训练命令。']
    C.text('343_交接索引_v2.md','\n'.join(lines)+'\n')
    C.text('343_本轮最新入口.txt',str(C.HERE/'343_交接索引_v2.md')+'\n阶段4工程已验证；阶段3b实现失败：seed2 development1466非正fullP。\n阶段5/真实训练未执行。\n')
    artifacts={str(p):{'sha256':C.sha(p),'bytes':p.stat().st_size} for p in C.HERE.rglob('*') if p.is_file()}
    C.write('343_交付封存SHA256_v2.json',{'status':'阶段4工程已验证；阶段3b单条实现失败；等主控','artifacts':artifacts,
        'failed_missing_cache':{'seed':2,'image_id':1466},'source_sha256':C.sources(),
        '已执行':{'artifact_hashes':len(artifacts),'protected_old_hashes':len(checked),'global_forwards':0,'real_training_runs':0},
        '未执行':['#1466新M0','阶段5','真实训练','GPU','SSH','队列'],'未核':['主控独立核对'],'wall_seconds_finalization':time.perf_counter()-start})
    print(json.dumps({'status':'handoff_v2_ready','artifacts':len(artifacts),'protected':len(checked),'index':str(C.HERE/'343_交接索引_v2.md')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
