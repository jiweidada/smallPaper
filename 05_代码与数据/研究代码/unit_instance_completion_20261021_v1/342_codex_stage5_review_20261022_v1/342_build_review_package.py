"""342 section VI review: source-bound statuses, frozen run plans, no launch code."""
import sys,importlib,json,time,ast,math,collections
sys.stdout.reconfigure(encoding='utf-8');C=importlib.import_module('342_stage5_context')

def link(p,label=None):
    p=C.Path(p);return f'[{p.name if label is None else label}]({p.as_posix()})'

def dump_ast(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    return {n.name:ast.dump(n) for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}

def main():
    start=time.perf_counter();target=C.WORK/'342_开训前审查包.md';assert not target.exists(),'exclusive-create only'
    release={'status':'阶段5已明确授权；真实训练未授权','authorization_source':'2026-10-22本轮用户直接消息',
        'authorization_quote':'344 我核过了，可以放行阶段 5，也就是写 `342_开训前审查包.md`。',
        'independent_user_verified':['6470清单与唯一非正框；随机200条文件SHA一致（用户报告，非本轮代跑）',
            '#1466非M0字段、PW=36、无合法M0边、owner=-1、最大匹配、GW/UW=0',
            'train344三处diff、geom342只增三个函数、mask只看raw宽高',
            'EMA文本AP接近日志；主控判断历史日志基本确定采用EMA'],
        'user_did_not_rerun':['四臂冒烟和续跑，仅读JSON','EMA前向，仅读COCO文本','bootstrap与12项方向脚本沿阶段4结论'],
        'paper_baseline':'raw AP=0.13924191755885088/0.1482370834937324；新头所有增益相对同seed raw。',
        '已执行':{'authorization_records':1,'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
        '未执行':['真实训练','GPU','SSH','队列'],'未核':['阶段5审查包主控审阅'],'source_sha256':C.sources()}
    C.write('342_阶段5放行记录.json',release)
    # Recheck the complete old-file seal, including all historical repaired artifacts.
    old=C.read(C.S344/'344_result_旧文件不覆盖与语法审计.json')
    expected={r['path']:r['expected_sha256'] for r in old['entries']}
    oldseal=C.read(C.S344/'344_交付封存SHA256.json')
    for p,v in oldseal['artifacts'].items():
        if p in expected:assert expected[p]==v['sha256']
        expected[p]=v['sha256']
    for p,h in oldseal.get('source_sha256',{}).items():
        if p in expected:assert expected[p]==h
        expected[p]=h
    expected[str(C.S344/'344_交付封存SHA256.json')]=C.sha(C.S344/'344_交付封存SHA256.json')
    assert not any('heldout' in p.lower() or 'holdout' in p.lower() for p in expected)
    unchanged=[]
    for n,(p,h) in enumerate(sorted(expected.items()),1):
        actual=C.sha(p);assert actual==h,p
        unchanged.append({'path':p,'expected_sha256':h,'actual_sha256':actual})
        if n%2048==0 or n==len(expected):print(f'phase5 old SHA {n}/{len(expected)}',flush=True)
    dev_manifest=C.S344/'344_development6470_逐条SHA清单.json';val_manifest=C.S3/'342_val548_逐条SHA清单.json'
    cache_rows=[];storage={};cache_checks=0
    for split,mp,N in [('development',dev_manifest,6470),('val',val_manifest,1096)]:
        doc=C.read(mp);assert len(doc['entries'])==N
        seen=set()
        for r in doc['entries']:
            key=(r['seed'],r['image_id']);assert key not in seen;seen.add(key)
            p=C.Path(r['path']);assert p.stat().st_size==r['bytes'] and C.sha(p)==r['file_sha256'];cache_checks+=1
            storage.setdefault(split,{}).setdefault(str(r['seed']),{'entries':0,'bytes':0})
            s=storage[split][str(r['seed'])];s['entries']+=1;s['bytes']+=p.stat().st_size
            cache_rows.append({'split':split,**r})
        assert all(storage[split][str(s)]['entries']==N//2 for s in (1,2))
    C.write('342_活动缓存7576_逐条SHA总清单.json',{'status':'已验证','entries':cache_rows,'source_manifests':[C.evidence(dev_manifest),C.evidence(val_manifest)],
        'storage':storage,'已执行':{'cache_file_SHA_checks':cache_checks,'model_forwards':0,'optimizer_steps':0},
        '未执行':['重新计算标签/M0','训练'],'未核':['远端副本身份'],'source_sha256':C.sources()})
    C.write('342_result_阶段5只读与输入身份核验.json',{'status':'已验证','old_file_entries':unchanged,
        '已执行':{'old_file_SHA_checks':len(unchanged),'active_cache_SHA_checks':cache_checks,'global_forwards':0,'optimizer_steps':0},
        '未执行':['真实训练','GPU','SSH','队列','heldout文件读取'],'未核':['远端运行环境身份'],'source_sha256':C.sources()})
    # Inherit historical 311 contracts only where the exact definition survives.
    v1=C.SPEC/'311_unit_instance_core/core.py';v2=C.SPEC/'311_unit_instance_core_v2/core.py'
    a,b=dump_ast(v1),dump_ast(v2);same={k:a[k]==b.get(k) for k in a}
    historic=C.read(C.SPEC/'311_unit_instance_core/synthetic_contracts.json')
    for p,h in historic['sources'].items():assert C.sha(p)==h
    inheritance={'unchanged_AST_definitions':same,'original_311_contracts':historic,
        'v2_current_sha256':C.sha(v2),'v1_current_sha256':C.sha(v1),
        'scope':'317允许继承311原始合同；v2数值稳定化由result_redo3v2_frozen311v2/result_redo3v2_tinyu补核；未把历史v1结果冒称当前适配器独立重跑。',
        '已执行':{'AST_definition_comparisons':len(same),'model_forwards':0,'optimizer_steps':0},
        '未执行':['历史311完整重跑','训练'],'未核':['当前适配器所有历史合成用例逐项重跑'],'source_sha256':C.sources()}
    C.write('342_result_311合同继承身份.json',inheritance)
    idx=C.read(C.SPEC/'327_训练实施输入索引.json');ids=idx['sample_plan']['development_ids'];assert len(ids)==3235
    cp_by_seed={r['seed']:r for r in idx['checkpoints']}
    for p in cp_by_seed.values():assert C.sha(p['path'])==p['expected_SHA256_from_288']
    phase2=C.read(C.S2/'342_阶段2_result.json');phase3=C.read(C.S3/'342_阶段3_result.json')
    dev=C.read(C.S344/'344_result_6470补全与非M0对账.json')
    smoke=C.read(C.S344/'344_result_train344四臂CPU冒烟与续跑.json')
    resources=C.read(C.HERE/'342_result_CPU单窗资源实测_v2.json');ema=C.read(C.S344/'344_result_EMA诊断_v2.json')
    res={(r['seed'],r['arm']):r for r in resources['rows']}
    # Plans describe the exact fixed budget. They are data, with no command/entry.
    def lr(step,base,lo):
        if step<100:return lo+(base-lo)*step/100
        t=(step-100)/(1218-100)
        return lo+.5*(base-lo)*(1+math.cos(math.pi*min(1.,t)))
    curves={'indexing':'zero-based optimizer steps 0..1217; end-point lo at hypothetical step1218, not an extra step',
        'non_kappa':[lr(i,1e-4,1e-5) for i in range(1218)],'G_kappa':[lr(i,1e-3,1e-4) for i in range(1218)]}
    C.write('342_预定学习率曲线1218点.json',curves)
    plans=[]
    descriptions={'N':'density-only；生成不读P；P只供共同aux','S':'同mass/RGB＋P-token cross-attention和10类log1p计数',
        'G':'固定d_out＋10类可训练null标量；独立可学习aux；class_null专用×10 LR',
        'R':'可学习逐candidate content/null的同一plan进入aux与残余生成；P无旁路'}
    for seed in (1,2):
        for arm in 'NSGR':
            row={'run_id':f'{arm}_seed{seed}','arm':arm,'global_seed':seed,'head_seed':2026102100+seed,
                'global_checkpoint':cp_by_seed[seed]['path'],'global_checkpoint_sha256':cp_by_seed[seed]['expected_SHA256_from_288'],
                'global_state_key':'model','global_mode':'eval / frozen / detached','head_implementation':str(C.S2/'342_head.py'),
                'CPU_review_entry':str(C.S344/'train344.py'),'CPU_review_entry_sha256':C.sha(C.S344/'train344.py'),
                'real_training_entry':'未执行；train344直接运行退出、run_steps仅允许CPU且<=20步；另行授权后实现并验收正式入口。',
                'development_manifest':str(dev_manifest),'development_manifest_sha256':C.sha(dev_manifest),
                'val_manifest':str(val_manifest),'val_manifest_sha256':C.sha(val_manifest),
                'head_parameters':res[seed,arm]['head_parameters'],'untrained_initial_state_sha256':res[seed,arm]['state_sha256_before'],
                'interface':descriptions[arm],'crop':[3,128,128],'f_grid':[16,16,128],'density':[10,256],'queries':4,
                'epochs':6,'fixed_last':True,'select_best':False,'batch':16,'drop_last':False,'windows_per_epoch':3235,
                'steps_per_epoch':203,'last_batch_size':3,'total_steps':1218,'total_crop_exposures':19410,
                'shuffle':'同seed四臂逐轮相同shuffle；head seed为2026102101/2；正式shuffle序列/游标尚未物化，不能当已执行。',
                'optimizer':'AdamW','betas':[.9,.999],'eps':1e-8,'weight_decay':1e-4,'bias_norm_decay':0.,'clip_grad_norm':1.,'dtype':'FP32',
                'lr_non_kappa':{'base':1e-4,'lo':1e-5,'warmup_steps':100,'post_warmup':'cosine','curve':str(C.HERE/'342_预定学习率曲线1218点.json')},
                'lr_class_null':None if arm!='G' else {'base':1e-3,'lo':1e-4,'warmup_steps':100,'weight_decay':0.,'gradient':'L_det only'},
                'loss_weights':{'L_det':1.,'L_mass':1.,'L_assoc':1.},'L_det_normalization':'max(1,sum batch |U_W|), including overflow；主控解读',
                'augmentation':'none','EMA':False,'ROI_warmstart':False,'extra_epochs':False,'NMS':False,
                'evaluation':'fixed last, native val548; same raw global baseline; sigmoid flatten top300, score>0.001, per-class100',
                'status':'已验证配置算术；真实run未执行'}
            assert row['epochs']*row['steps_per_epoch']==1218 and row['epochs']*row['windows_per_epoch']==19410
            plans.append(row)
            C.write(f'342_预定配置_{arm}_seed{seed}.json',{'config':row,'已执行':{'config_writes':1,'model_forwards':0,'optimizer_steps':0},
                '未执行':['真实run','正式训练入口','正式shuffle物化'],'未核':['GPU资源与远端身份'],'source_sha256':C.sources()})
    C.write('342_八run配置清单.json',{'runs':plans,'planned_totals':{'runs':8,'head_epochs':48,'optimizer_steps':9744,'crop_exposures':155280},
        '已执行':{'configurations':8,'budget_arithmetic_checks':8,'real_runs':0,'optimizer_steps':0},
        '未执行':['真实训练','正式shuffle物化','远端运行目录创建'],'未核':['真实训练资源'],'source_sha256':C.sources()})
    evidence_used=set();contracts=[]
    def add(item,status,scope,*paths):
        pp=[C.Path(p) for p in paths]
        for p in pp:assert p.is_file();evidence_used.add(p)
        contracts.append({'item':item,'status':status,'scope':scope,'evidence':[C.evidence(p) for p in pp]})
    add('M0类/最大基数/精确cost/索引lex及U_W闭合','已验证','val1096与development6469独立证书＋#1466补建独立证书；主控另抽1262条scipy最小代价；#448 FP32边界保留。',
        C.S3/'342_result_M0独立核查.json',C.S3/'342_result_val548独立验收.json',C.S343/'343_result_3b已生成6469条独立验收.json',C.S344/'344_result_1466独立验收与6PNG_v2.json')
    add('K_g单位质量、完整/partial、边界与crop','已验证','两seed val1096、development6469＋补1466独立实现；沿冻结1e-6语义。几何反向测试与PNG见各阶段。',
        C.S3/'342_result_val548独立验收.json',C.S343/'343_result_3b已生成6469条独立验收.json',C.S344/'344_result_1466独立验收与6PNG_v2.json',C.S344/'344_result_6PNG本轮目视.json')
    add('唯一geom342/候选与目标尺度','已验证','候选窗内归一化xyxy、目标窗内归一化cxcywh；不裁剪；独立d_out差1.787072456e-6。主控已核阶段1/2。',
        C.S2/'342_result_geom_unit.json',C.S2/'342_result_训练尺度诊断.json',C.S344/'344_geom342_vs343.diff')
    add('native检测损失parity与batch分母','已验证','两个16图batch的native总损失差0/2.384185791e-7；分母含空图但不为其额外加1，含overflow；修正额外batch平均。344保持公式。',
        C.S2/'342_阶段2_result.json',C.S344/'344_result_train344全量输入与入口验收.json')
    add('L_mass零目标项及L_assoc目标/空组','已验证','L_mass独立NumPy差<=9.068182579e-7，m*=0仍保留+m且GW=0进入均值；L_assoc与collate共用PW mask，1466有限loss。',
        C.S2/'342_阶段2_result.json',C.S344/'344_result_train344四臂CPU冒烟与续跑.json',C.S344/'344_train344_vs_train342.diff')
    add('optimizer路由、317v4学习率、联合clip','已验证','1218点路由核对；G额外10参数无decay、只接L_det、整曲线×10；N/S/R不变。',
        C.S2/'342_阶段2_result.json',C.S344/'344_result_train344全量输入与入口验收.json',C.HERE/'342_预定学习率曲线1218点.json')
    add('有界CPU冒烟与续跑','已验证','两seed×四臂各20步；8组continuous2 vs1+save/load+1的模型/优化器/RNG逐位一致；主控只读JSON未重跑。',
        C.S344/'344_result_train344四臂CPU冒烟与续跑.json')
    add('global冻结、零grad/state与detach','已验证','阶段2 seed1真实raw模型794项state不变且每步grad None；阶段3两seedraw缓存前后state不变；344 head优化无global参数。不是正式8run实测。',
        C.S2/'342_阶段2_result.json',C.S3/'342_阶段3_result.json',C.S344/'344_result_train344四臂CPU冒烟与续跑.json')
    add('κ_c端到端梯度与损失路由','已验证','新标签两seed每类20步中19步非零，逐步值/更新均保存；L_mass与L_assoc对class_null为None；不外推训练终值。',
        C.S344/'344_result_train344四臂CPU冒烟与续跑.json')
    add('G/R初态计划和预测同值、G的aux隔离','已验证','冻结v2合成记录：预测、aux、初态生成同值；改变aux不动G生成；不是本轮重跑。hidden等历史合同见下一项。',
        C.WORK/'result_redo3v2_frozen311v2.json')
    add('311守恒、empty/overflow、S排列/count、N无P与sg(m)','已验证','按317继承311历史合成合同；明确v1来源与v2 AST继承关系，v2极小u安全梯度另有记录；未冒称train344把全部合成用例重跑。',
        C.HERE/'342_result_311合同继承身份.json',C.WORK/'result_redo3v2_tinyu.json',C.SPEC/'320_geometry_calibrated_core/synthetic_contracts.json')
    add('G/R decoder hidden及所有继承合同在当前适配器上的完整重跑','未验证','历史320记录hidden/nonzero-readout同值；v2直接记录仅计划/预测。当前适配器全套合同未逐项重跑，不把输出头零初始化当hidden证明。',
        C.SPEC/'320_geometry_calibrated_core/synthetic_contracts.json',C.WORK/'result_redo3v2_frozen311v2.json',C.S2/'342_head.py')
    add('G有效性Δ度量实现与合成','已验证','g_effect_342必须由调用方供训练后mass，无uniform fallback；32窗合成及非均匀mass、空P、错误格式均有记录。',
        C.S2/'g_effect_342.py',C.S2/'342_result_g_effect_synthetic.json',C.S2/'342_result_闸门补充核对.json')
    add('训练后G有效校准','未验证','未执行。需训练后各seed548窗用真实mass计算Δ并报告逐类κ终值/位移；Δ<.01时R−G四项不能作为支持R的证据。',C.SPEC/'317v4_G臂kappa_c学习率修订.txt')
    add('局部框反变换与融合parity','已验证','1096窗独立/往返/反向；native映射与统一预算。低分new4的12项与baseline逐位相同；随机new4可计算。',
        C.S343/'343_result_阶段4映射与融合_v3.json',C.S343/'343_result_阶段4全量评估.json')
    add('绑定评估器与raw baseline','已验证','Windows绑定库False/False、max_dets100；独立pycocotools重评12项完全相同。baseline明确raw。',
        C.S343/'343_evaluator.py',C.S344/'344_result_独立pycocotools_baseline_v2.json',C.S344/'344_result_EMA诊断_v2.json')
    add('正式B/TP/FP及新增丢失计数','已验证','全图同类一对一score-greedy后筛组；相对同seed global-only；双实现全量6576对照和6个B阳性构造。口径由用户委托后冻结。',
        C.S343/'343_追加放行与计数口径.json',C.S343/'343_计数口径补充冻结.json',C.S343/'343_result_阶段4正式计数_v2.json',C.S343/'343_result_阶段4主B阳性构造_v2.json')
    add('1000次配对bootstrap工程链路','已验证','自然val baseline/合成new4，548共享索引、seed2026102103；6000次重算AP/APs/B，18 native累积与2全克隆逐位对照；非8run科学CI。',
        C.S343/'343_result_阶段4自然val配对bootstrap.json')
    add('12项严格方向规则脚本','已验证','五种构造含零/负/跨seed抵消/G未校准脚注；未执行正式判定。G有效性需另结合317v4，不能只看sign布尔。',
        C.S343/'343_result_阶段4方向规则.json',C.S343/'343_direction_rule.py')
    add('正式8run、正式科学判定和训练资源','未验证','全部未执行；不生成训练命令。当前train344仅有界CPU，正式运行入口和完整shuffle恢复尚需授权后实现/验收。',C.S344/'train344.py',C.HERE/'342_八run配置清单.json')
    C.write('342_合同逐条证据表.json',{'contracts':contracts,'已执行':{'evidence_contract_rows':len(contracts),'new_contract_model_calls':0,'optimizer_steps':0},
        '未执行':['科学训练与正式判定','本轮合成合同重跑'],'未核':['主控对审查包裁定'],'source_sha256':C.sources()})
    # Store timing receipts without pretending overlapping processes form a total wall clock.
    timings=[]
    for x in phase3['timing_receipts']:timings.append({'scope':'val','path':str(C.S3/x['receipt']),'seconds':x['seconds']})
    for p in sorted(C.S343.glob('343_result_3b构建_seed*.json')):
        d=C.read(p);timings.append({'scope':'development_rebuild','path':str(p),'seconds':d['elapsed_seconds'],'counts':d['counts']})
    timings.append({'scope':'344_full_scan_plus_1466_build_plus_manifest','path':str(C.S344/'344_result_6470补全与非M0对账.json'),
        'seconds':dev['wall_seconds'],'note':'该耗时包括全量scan/对账/清单；#1466单独构建耗时未核。'})
    C.write('342_缓存耗时与存储账本.json',{'storage_active':storage,'val_raw_storage_historical':phase3['storage'],'timing_receipts':timings,
        'timing_scope':'各进程实测耗时；有并行/断点/失败与扫描，不相加为总墙钟。development复用340/339 raw，历史global完整构建成本未核。',
        'GPU_memory':'NA','GPU_hours':'NA','GPU_wall_clock':'NA',
        '已执行':{'active_cache_sizes_counted':7576,'timing_receipts_read':len(timings),'global_forwards':0,'optimizer_steps':0},
        '未执行':['新global缓存构建','真实训练'],'未核':['完整累计缓存墙钟','development历史global前向成本'],'source_sha256':C.sources()})
    # Build the review in the exact six-section order required by 342 section VI.
    lines=['# 342 开训前审查包','',
        '日期：2026-10-22。状态：阶段5汇总完成，待主控审查；本文件不构成真实训练授权。', '',
        '主线：已有检测的可拒绝解释与漏检补全。冻结BASE24 raw全局模型，比较N/S/G/R四臂在同一局部RGB、同一辅助监督和固定预算下的生成信息路径。R学习逐candidate对应/null并将残余用于四槽补全；R−S检验受限残余接口相对自由P融合，R−G检验可学习content/逐candidate null相对固定几何＋类别null校准的联合增量。', '',
        '主依据为296、317v3；仅G的class_null学习率由317v4修订。后续342/343/344落实几何、M0、计数和#1466明示例外。本轮直接用户消息已放行阶段5，记录见 '+link(C.HERE/'342_阶段5放行记录.json')+'。', '',
        '状态词：“已验证”均限定于所列执行范围；“未验证”同时注明未执行/未核；“待主控”表示需要审阅或授权。主控独立核对与执行方自测分列，旧报告中的历史闸门不在旧文件里回写。','',
        '## 1. 全部合同','',
        '| 合同 | 状态 | 已有证据与范围 |','|---|---|---|']
    for r in contracts:
        refs='；'.join(link(e['path']) for e in r['evidence'])
        lines.append(f'| {r["item"]} | {r["status"]} | {r["scope"]}<br>{refs} |')
    lines+=['','主控核对范围：阶段1/2已独立批准；3b统计、M0抽查、浮点边界、阶段4映射及raw baseline已独立核对；344独立核了6470清单、随机200条SHA、#1466和代码diff。四臂冒烟/续跑仅看JSON，EMA前向仅看COCO文本，bootstrap/方向脚本沿阶段4结论未重跑。上述是用户报告，不冒充本轮代跑。','',
        '## 2. 缓存清单与SHA','',
        '状态：已验证。阶段5重新核验development 6470条＋val1096条，共7576条文件SHA与大小。所有活动条目均有源checkpoint、源图像及payload身份；训练加载只绑定新development清单。','',
        '| split/seed | 条目数 | 活动文件字节 | 清单 |','|---|---:|---:|---|']
    for split in ('development','val'):
        for s in ('1','2'):
            x=storage[split][s];mp=dev_manifest if split=='development' else val_manifest
            lines.append(f'| {split}/seed{s} | {x["entries"]} | {x["bytes"]} | {link(mp)} |')
    lines+=['','合并逐条证据（路径、文件/payload/source SHA）：'+link(C.HERE/'342_活动缓存7576_逐条SHA总清单.json')+'。',
        'development中6469条继续引用343只读缓存，seed2 #1466为344新文件；mask在旧343条目读取并核SHA后由raw宽高重算入内存，#1466显式存fullp_nonpositive_mask。val只采用342_cache_val_v2_seed1/2；保留的首版失败缓存不作输入。',
        '来源链：340有效子集SHA＋339 final300逐位对照＋本轮整文件封存，用户已接受；340历史子集SHA本来不覆盖final300，只有seed1前32条具有历史整文件SHA，不把新封存写成更早的历史证明。证据：'+link(C.S343/'343_result_3b来源预检.json')+'、'+link(C.S343/'343_追加放行与计数口径.json')+'。','',
        '| development指标 | seed1 | seed2 |','|---|---:|---:|']
    for title,key in [('G_W>0图','G_W_positive'),('U_W>0图','U_W_positive'),('U_W总GT','U_W_total'),('|U_W|>4图','U_W_gt4_images'),('溢出GT','overflow_GT'),('旧跨类owner','old_cross_class_owners'),('其中owner在G_W','old_cross_class_owners_in_G_W'),('U_W变化图','U_W_changed_images')]:
        lines.append(f'| {title.replace("|","\\|")} | {dev["stats"]["1"][key]} | {dev["stats"]["2"][key]} |')
    lines+=['','两seed同窗1443/3235='+f'{1443/3235*100:.4f}%'+ '。val的U_W>0为310/548、311/548；|U_W|>4为84/548、101/548，溢出412/507个GT，与development分别报告。Wilson区间沿原始验收，不以统计调整配方。证据：'+link(C.S3/'342_阶段3_result.json')+'、'+link(C.S344/'344_result_6470补全与非M0对账.json')+'。','',
        '## 3. 参数量、CPU单窗资源、缓存耗时与存储','',
        '参数量与所列CPU计量已验证；完整逐算子FLOPs未验证。使用当前train344真实头，B=1、RGB128²、CPU FP32、torch两线程、两seed各development#1；每例预热2次＋计时10次＋FLOP计量1次，无优化器或global前向。参数先按训练模型记录，计量临时模型关闭requires_grad，权重/state未变。','',
        '| arm | head参数 | seed1 P_W | seed1已计FLOPs | seed1时延中位ms | seed2 P_W | seed2已计FLOPs | seed2时延中位ms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in 'NSGR':
        a1,a2=res[1,arm],res[2,arm]
        lines.append(f'| {arm} | {a1["head_parameters"]} | {a1["P_W"]} | {a1["FLOPs_registered_ops"]} | {a1["latency_ms_median"]:.6f} | {a2["P_W"]} | {a2["FLOPs_registered_ops"]} | {a2["latency_ms_median"]:.6f} |')
    lines+=['','S−N参数为133888，G−R为10；global实际参数为10231997，冻结且不计入head可训练参数。','',
        'FLOPs口径：PyTorch FlopCounterMode所注册的卷积/矩阵乘等算子，乘加按2 FLOPs；逐算子清单与未注册调用见证据。CPU融合attention、softmax、归一化和elementwise等可能未计入，数字不冒称完整FLOPs或完整部署成本。时延包含aux分支，未剪枝部署aux；不含cache I/O、collate、损失、反向、优化器、global和评价。仅两个指定窗，不冒称数据集均值。',
        '实测与运行环境：'+link(C.HERE/'342_result_CPU单窗资源实测_v2.json')+'。首次计量因torch module_tracker的grad-hook断言实现失败，计量临时模型关闭requires_grad后另存v2通过；原记录与diff保留：'+link(C.HERE/'342_result_CPU计量首次实现失败.json')+'、'+link(C.HERE/'342_CPU计量接口修正.diff')+'。',
        '缓存活动存储量见第2节；含逐条大小与分阶段实测耗时的账本：'+link(C.HERE/'342_缓存耗时与存储账本.json')+'。各进程存在并行、断点和失败段，不相加冒称总墙钟；#1466单独构建时间及development历史global完整成本未核。GPU显存=NA、GPU小时=NA、GPU墙钟=NA，未外推。','',
        '## 4. 八个run配置清单','',
        '状态：已验证配置算术与源文件绑定；八个真实run均未执行。下表为固定计划，未创建真实运行目录。每run6轮固定last，3235图/轮，batch16、drop_last=False、每轮203步（末批3图），1218步、19410 crop曝光；全部8run为48 head epochs、9744步、155280曝光。', '',
        '| run | head seed | epochs | batch/drop_last | steps | crop曝光 | LR | 配置 |','|---|---:|---:|---|---:|---:|---|---|']
    for r in plans:
        x='常规1e-5→1e-4→1e-5；κ 1e-4→1e-3→1e-4' if r['arm']=='G' else '1e-5→1e-4→1e-5'
        lines.append(f'| {r["run_id"]} | {r["head_seed"]} | 6 | 16/False | 1218 | 19410 | {x} | {link(C.HERE/f"342_预定配置_{r[\"arm\"]}_seed{r[\"global_seed\"]}.json")} |')
    lines+=['','共同AdamW：betas(.9,.999)、eps1e-8、wd1e-4，bias/norm无decay；前100步线性warmup后cosine、联合clip1、FP32。G仅class_null整条曲线×10、无decay、只接L_det；其余与317一致。四臂同公共初态、同seed逐轮同shuffle，各自独立优化器/状态；正式shuffle序列及完整run恢复尚未执行。无增强/EMA/ROI warmstart/best选轮/额外轮。',
        '1218点曲线与配置总表：'+link(C.HERE/'342_预定学习率曲线1218点.json')+'、'+link(C.HERE/'342_八run配置清单.json')+'。配置含实际未训练初态state SHA、raw CP SHA和development/val清单SHA。',
        '正式run入口未执行：train344保留CPU、<=20步、<=32窗断言且__main__直接退出。当前不能把该文件视为1218步启动程序；真实训练获单独授权后需另存正式入口、绑定数据/初态/每轮shuffle/完整恢复并验收，不能通过删除守卫直接启动。','',
        '## 5. 已知限制与论文口径','',
        '### 5.1 raw baseline与EMA诊断','',
        '主控判断：结合EMA优先评估源码、配置与COCO AP对齐，历史日志基本确定评的是EMA。执行方实测的完整12项仍有残差，原因未核；不把三位小数一致写成逐位复现。论文的baseline必须写raw，所有新增头增益相对同seed raw计算。','',
        '| seed | raw AP（主线） | EMA AP（诊断） | 日志AP | EMA−日志AP | 12项最大绝对差 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in ema['per_seed']:
        lines.append(f'| {r["seed"]} | {r["raw_stats12"]["AP"]:.12f} | {r["EMA_stats12"]["AP"]:.12f} | {r["training_log_stats12"]["AP"]:.12f} | {r["EMA_minus_log"]["AP"]:.12f} | {r["max_abs_EMA_minus_log"]:.12f} |')
    lines+=['','EMA权重路径checkpoint[ema][module]，两seed各548，共1096次CPU前向；预测只在内存供评价，未进入缓存/M0/训练输入。独立pycocotools重评raw的12项完全一致。原始12项及全精度差值：'+link(C.S344/'344_result_EMA诊断_v2.json')+'、'+link(C.S344/'344_result_独立pycocotools_baseline_v2.json')+'。','',
        '### 5.2 其他固定限制','',
        '- 评估后端：本机Windows.pyd与历史receipt Linux.so不是同一二进制，跨平台逐位一致未核；using_tiny_metrice=False、using_vtiny_metrice=False、max_dets100固定。',
        '- #1466明示例外：唯一seed2/query62/fullP259 raw h<=0，原坐标保留参与top300/score/逐类100；native类内第24，最终仍保留。P_W排除，剩36候选；IoU=0、M0合法边False、owner=-1，旧owner也为-1。GW=UW=0，标签不变。无abs/clip/交换边/eps修框。',
        f'- development四槽overflow：seed1 474/3235={474/3235*100:.4f}%图、3047个溢出GT；seed2 468/3235={468/3235*100:.4f}%图、3199个溢出GT。旧16.5%与3435/3515不作为当前数字。',
        f'- 两seed同窗1443/3235={1443/3235*100:.4f}%，不是完全相同局部输入。#448 FP32 top300并列及#3222/#6383的448.00003完整边界按冻结语义保留。',
        '- L_det分母max(1,Σ batch |U_W|)为主控对296/317的解读，包含overflow完整目标；不是新增标签或每图分母平均。',
        '- G有效性必须在训练后用真实mass核验Δ：<0.01为“G未有效校准”，R−G四项不能作为支持R的证据；[0.01,0.03)为“校准能力受限”；>=0.03可作正式对照。20步κ更新、初值mass或位移不替代Δ。',
        '- 12项方向全正仅为有限同向筛查，零不算正，不平均、不跨seed抵消；总体AP任何负值仍全披露。mixed就停，不加轮、不改M0/窗/阈值/标签/子组。',
        '- val548已用于历史研究，不能称全新未见测试集；bootstrap按548图配对，两seed同图不当1096独立样本。当前bootstrap只有baseline/合成新框工程输出，尚无训练产物CI。',
        '- R−G是可学习content/实例null与梯度路径的整体比较，不是null的单因果试验；原图重读、单位质量、密度减法、补集监督、solver均不据此宣称首创。',
        '- 当前adapter完整hidden/继承合同重跑、完整FLOPs覆盖、累计缓存墙钟、远端训练环境身份均未核，不能用有限loss或参数量代替。','',
        '## 6. 开训前还差什么','',
        '| 事项 | 状态 | 处理边界 |','|---|---|---|',
        '| 主控审阅本包及未验证项 | 待主控 | 本轮仅交付审查包，未自行标“批准训练”。 |',
        '| 是否补核当前adapter的hidden/全部继承合成合同与FLOPs覆盖 | 待主控 | 本包已明确证据范围；未把缺项写成已验证。若要求补核，在另存CPU脚本中完成，不改配方。 |',
        '| 正式训练入口、shuffle物化与1218步完整恢复验收 | 未验证 | 未执行，当前train344为CPU有界入口；须另行授权后实现/验收并绑定SHA。 |',
        '| 真实训练执行与执行环境放行 | 待主控 | 未授权；GPU/SSH/队列/远端副本/显存与墙钟均未执行或NA；本包不给命令。 |',
        '| 训练后G Δ、八run指标/CI/12项方向 | 未验证 | 未执行，属于训练后的必需判读，不能要求预检预先证明。 |',
        '| EMA残差及跨平台二进制差异 | 未验证 | 原因未核，保持已知限制，主线raw与冻结评估设置不变。 |','',
        '本包完成“开训前审查材料”交付；是否补核、是否批准正式入口及真实训练，由主控与用户裁定。当前没有启动任何真实训练，没有读取heldout文件。','',
        '阶段5调用数、输入封存、失败修复与交付SHA见 '+link(C.HERE/'342_阶段5_result.json')+'；一页中文汇总见 '+link(C.HERE/'342_阶段5_中文汇总.md')+'。','']
    package='\n'.join(lines)
    assert '可行' not in package and '不可行' not in package
    C.text('342_开训前审查包.md',package,root=C.WORK)
    evidence_used.update([target,dev_manifest,val_manifest,C.HERE/'342_活动缓存7576_逐条SHA总清单.json',C.HERE/'342_result_CPU单窗资源实测_v2.json'])
    C.write('342_审查包证据文件SHA256.json',{'files':{str(p):C.sha(p) for p in sorted(evidence_used)},
        '已执行':{'evidence_files':len(evidence_used),'global_forwards':0,'optimizer_steps':0},
        '未执行':['真实训练'],'未核':['主控审阅'],'source_sha256':C.sources()})
    countstatus=collections.Counter(r['status'] for r in contracts)
    result={'status':'阶段5审查包已写盘，等待主控审阅；真实训练未授权','package_path':str(target),'package_sha256':C.sha(target),
        'checklist_342_section_VI':[
            {'item':1,'status':'已验证/未验证逐条列示','contract_status_counts':dict(countstatus),'evidence':str(C.HERE/'342_合同逐条证据表.json')},
            {'item':2,'status':'已验证','entries':7576,'evidence':str(C.HERE/'342_活动缓存7576_逐条SHA总清单.json')},
            {'item':3,'status':'已验证所列资源；完整FLOPs未验证','evidence':str(C.HERE/'342_result_CPU单窗资源实测_v2.json')},
            {'item':4,'status':'已验证计划算术；真实run未执行','evidence':str(C.HERE/'342_八run配置清单.json')},
            {'item':5,'status':'已逐项披露','limits_updated':['3b新溢出474/468与3047/3199','1466明示例外','EMA基本解释历史但残差未核','raw baseline论文口径']},
            {'item':6,'status':'待主控；列出真实入口与未核项，不启动训练'}],
        '已执行':{'old_file_SHA_checks':len(unchanged),'active_cache_SHA_checks':cache_checks,'review_contract_rows':len(contracts),
            'planned_run_configs':8,'successful_resource_head_forwards':104,'failed_resource_head_forward_attempts':13,
            'failed_resource_head_forwards_completed':12,'resource_total_completed_head_forwards':116,
            'new_global_forwards':0,'optimizer_steps':0,'real_training_steps':0,'GPU_calls':0},
        'resource_failure_record':str(C.HERE/'342_result_CPU计量首次实现失败.json'),
        '未执行':['真实训练','正式run入口','GPU','SSH','队列','heldout文件读取','正式run指标与G Δ'],
        '未核':['主控审阅本包','当前适配器全部合成合同逐项重跑','完整逐算子FLOPs','EMA残差唯一原因','跨平台二进制一致','远端训练环境身份'],
        'source_sha256':C.sources()}
    C.write('342_阶段5_result.json',result)
    summary=f'''# 342 阶段5中文汇总

审查包已按342第六节六项逐条写盘：{link(target)}。本轮用户明确放行阶段5；本包等待主控审阅，真实训练未授权且0步执行。

主线仍是“已有检测的可拒绝解释与漏检补全”。固定raw全局检测器，用N/S/G/R×2seed检验逐candidate对应/null残余接口。12项APs/主B方向都严格为正才保留有限方向，mixed就停，不加轮、不改标签/窗/阈值。

既有工程预检：6470条development、1096条val均齐；阶段5重新核了7576条SHA。旧文件{len(unchanged)}项SHA不变。M0、损失/融合parity、优化器路由、两seed四臂冒烟与8组续跑均有证据。主控独立范围和仅阅JSON的范围分别登记。

development overflow采用新值：seed1 474图/3047溢出GT，seed2 468图/3199溢出GT；同窗1443/3235。唯一#1466非正框保留fullP与native预算，PW排除后36候选，M0无合法边、owner=-1，GW=UW=0，训练标签不变。

论文baseline固定raw AP 0.139241918/0.148237083；EMA诊断AP 0.165982759/0.164042129。结合配置/solver与日志，主控基本确定历史日志评的是EMA；12项最大残差0.000557261/0.000583622，原因仍未核，主线不切EMA。

实际head参数N/R=894137，S=1028025，G=894147；两seed指定单窗CPU时延与部分算子FLOPs已计量。成功104次无梯度head前向；首次计量有13次尝试（12次完成，1次失败），原记录与另存修复保留。无global前向、无优化步。完整FLOPs覆盖与累计缓存墙钟未核，GPU资源均NA。

八份预定配置已列：每run6轮、batch16、1218步、19410曝光，共8run/9744步/155280曝光；G仅10个class_null用×10学习率。正式训练入口、完整shuffle/1218步恢复未执行；train344仅CPU有界验证，直接运行退出。

开训前还差：主控审阅本包与未核项；裁定是否补核当前适配器hidden/完整继承合同与FLOPs；另行授权正式入口及真实训练。训练后G Δ与科学优劣尚未执行。证据入口：{link(C.HERE/'342_阶段5_result.json')}、{link(C.HERE/'342_合同逐条证据表.json')}。
'''
    C.text('342_阶段5_中文汇总.md',summary)
    print(json.dumps({'status':'package written; stop for review','path':str(target),'contracts':dict(countstatus),'storage':storage,'old_SHA_checks':len(unchanged)},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
