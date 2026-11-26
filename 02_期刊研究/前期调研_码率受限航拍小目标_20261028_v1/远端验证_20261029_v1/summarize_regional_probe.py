"""Verify every archived regional artifact, report outcomes, and keep the research goal active."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone,timedelta
import hashlib,json,shutil,tarfile
local=Path(__file__).resolve().parent;base=local.parent;folder=local/'完成结果_20261029_v1'
archive=folder/'regional_probe_completed.tar.gz';receipt=json.loads((folder/'regional_archive_receipt.json').read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
target=(folder/'regional_raw').resolve();target.mkdir(exist_ok=True)
with tarfile.open(archive,'r:gz') as tar:
    for m in tar.getmembers():
        name=PurePosixPath(m.name)
        assert not name.is_absolute() and '..' not in name.parts and '\\' not in m.name and ':' not in m.name
        assert m.isfile() or m.isdir();path=target.joinpath(*name.parts).resolve();assert target in path.parents
        if m.isdir():path.mkdir(parents=True,exist_ok=True);continue
        path.parent.mkdir(parents=True,exist_ok=True)
        with tar.extractfile(m) as src,path.open('wb') as dst:shutil.copyfileobj(src,dst)
root=target/'restoration_probe_v1';manifest=json.loads((root/'artifact_manifest.json').read_text())
for row in manifest:
    p=target.joinpath(*PurePosixPath(row['path']).parts)
    assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
complete=json.loads((root/'completed.json').read_text());boot=json.loads((root/'bootstrap_completed.json').read_text())
fixed=json.loads((root/'fixed_choice_metrics.json').read_text());loss=json.loads((root/'loss_analysis.json').read_text())
freeze=json.loads((root/'frozen.json').read_text());assert sha(root/'run_probe.py')==freeze['script_sha256'] and sha(root/'protocol.json')==freeze['protocol_sha256']
assert complete['status']=='COMPLETED' and complete['detector_forwards']==3800 and complete['training_updates']==0
now=datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-7))).isoformat()
def m(arm,name):return complete['metrics'][arm][name]
def matches(arm,name):return next(v for v in m(arm,name)['matches'] if v['confidence']==.25 and v['iou']==.5)
names=['subset_00','subset_15','alpha_025','alpha_050','alpha_075','independent','quadratic','finite_loss_best','per_image_alpha_loss_best']
labels={'subset_00':'JPEG不恢复','subset_15':'全恢复','alpha_025':'固定全图alpha0.25','alpha_050':'固定全图alpha0.50','alpha_075':'固定全图alpha0.75','independent':'GT独立选区','quadratic':'GT二阶联合选区','finite_loss_best':'GT有限动作损失最优','per_image_alpha_loss_best':'GT逐图全局强度损失最优'}
lines=['## Material Passport','','- Origin Skill: academic-research-suite / experiment-agent + deep-research',
 '- Origin Mode: completed finite experiment + research-question refinement','- Origin Date: 2026-10-29；America/Los_Angeles',
 '- Verification Status: 归档与251个文件SHA核验；模型/损失接口和JPEG预测复现；单seed诊断，未形成部署算法',
 '- Version Label: regional_probe_v1_completed','',
 '区域恢复判别：二阶联合候选未通过，继续寻找主线','',f'整理时间：{now}','',
 '一、先给裁决','',
 '【观测】DA上独立选区的小目标AP24.9135%，二阶联合24.8208%，联合相对独立−0.0926pp，配对图像95%区间[-0.2750,0.2045]pp。',
 '【裁决】当前2x2四区、冻结FBCNN、Q20、固定YOLO26s checkpoint的结果，不支持将“显式二阶交互”作为算法主线。停止该增量开发，不训练二阶策略，不事后放宽0.5pp筛选门槛，不换网格继续追正结果。',
 '【边界】这是有范围的阴性判别，不证明所有恢复器/分辨率/空间粒度都没有相互作用；区间也不覆盖训练seed或独立场景变化。',
 '【保留线索】GT独立选区比最强固定全局alpha高0.6115pp，但选择使用GT，不能当作新方法或部署收益。只有剩余作用空间，尚未找到有独立新意且可部署的算法主线；持续目标保持ACTIVE。','',
 '二、实际完成了什么','',
 '复用D0/DA固定100轮EMA和相同100张Q20真实载荷。FBCNN作者commit54d1831927506b3247e2d4d245abb4f4dab1a1cd，color release权重SHA固定，自动QF、原始分辨率、FP32、无切块；输出clamp并按作者风格四舍五入uint8，缓存无损PNG。',
 '区域在原尺寸定义，x/y互补线性羽化构成4个非负权重且和为1，过渡带宽为图像各方向尺寸的5%。组合后按同一规则量化到uint8，再进入原先唯一一次letterbox；空集和全选精确对应JPEG/全恢复，padding没有额外残差。',
 '枚举16个子集，另加alpha0.25/0.50/0.75，共19个输入×100图×2模型=3800次正式检测前向；每图batch1。100张唯一恢复图中1张复用容量缓存，本阶段新恢复99次。',
 '额外容量检查2次恢复（含一次重复）；失败接口预检1次检测前向，修正后成功预检2次。与本轮codec800前向合计：正式科学检测4600次、接口3次、冻结恢复101次（唯一源图100）；新增训练更新0。',
 f"本次正式区域阶段耗时{complete['elapsed_seconds']/60:.2f}分钟。后续区间/固定分配选区AP全部重用保存预测，无额外模型前向。",
 '容量最大图2000x1500已分配峰值4086615040字节，保留峰值4269801472字节；该数字是容量预检，不冒充全阶段实测峰值。',
 '正式前预检曾因YOLO便利加载器不保留box/cls/dfl超参数而失败；从实际冻结训练启动回执补回7.5/0.5/1.5后，两个模型的基准动态/固定分配损失相同，JPEG逐框预测与旧结果完全一致。失败源码、日志与修正回执全部保留；训练文件和权重未更改。','',
 '三、主要结果','',
 '小目标主指标为640输入坐标面积严格<32²的AP50:95、maxDet500、内部COCO式crowd；不是原图标准AP_s或官方VisDrone挑战分数。GT选择项仅作诊断。',
 '条件 | D0小目标AP(%) | DA小目标AP(%) | DA总体AP(%) | DA原图面积AP_s(%)']
for name in names:lines.append(f"{labels[name]} | {100*m('D0',name)['AP_small_input']:.4f} | {100*m('DA',name)['AP_small_input']:.4f} | {100*m('DA',name)['AP']:.4f} | {100*m('DA',name)['AP_small_original']:.4f}")
lines+=['','source参考：D0输入面积小目标AP24.4239%，DA24.7060%。GT独立选区的点分数稍高于参考，也不能宣称真实算法已经超越参考。',
 '固定全局alpha0.75是在诊断结果中观察到最强者，不是已在调参集锁定的部署参数；逐图GT选alpha同样不可部署。',
 '有限动作损失最优只保证所定义损失在16个动作中最低。DA其AP24.8108%还略低于独立选区24.9135%，再次说明它不是AP上界。','',
 'DA预定confidence0.25、IoU0.5：','条件 | 小目标TP | 小目标FN | 全部FP/图']
for name in ['subset_00','subset_15','alpha_075','independent','quadratic']:
    v=matches('DA',name);lines.append(f"{labels[name]} | {v['small_TP']} | {v['small_FN']} | {v['FP_per_image']:.2f}")
lines+=['','【观测】独立选区和全恢复在此阈值的小目标TP都为2323，FP/图也接近。独立AP更高不能直接描述为“该阈值少漏了更多目标”；应同时看置信度排序、严格IoU、类别宏平均与GT身份转移。',
 '本轮保留了confidence0.05/0.1/0.25/0.5及IoU0.5/0.75的完整统计，未按最好的阈值选取结论。','',
 '四、不确定性和实际GT转移','',
 'DA、2000次源图成对bootstrap、单位pp；固定checkpoint和GT诊断动作条件下的探索性逐项区间：','比较 | 点差 | 95%区间']
for name,v in boot['contrasts'].items():
    a,b=v['paired_image_bootstrap95_pp'];lines.append(f"{name} | {v['point_pp']:+.4f} | [{a:+.4f}, {b:+.4f}]")
lines+=['','七个区间未校正为同时覆盖；最强全局alpha为诊断后选定，相关对比应按条件性探索解释。没有多seed、独立航次或新图像外推保证。',
 '相对Q20的同一小目标身份变化（confidence0.25、IoU0.5）：','条件 | 丢掉Q20已有命中 | 新命中 | 找回source命中而Q20漏掉者/222']
for name,v in boot['transitions_conf025_iou05'].items():lines.append(f"{labels[name]} | {v['jpeg_hits_lost']} | {v['new_hits_vs_jpeg']} | {v['recovered_source_hit_jpeg_missed']}/222")
lines+=['','【推断】恢复有真命中恢复，也有旧命中丢失；不能把净TP或单个更高AP当成零风险改善。GT独立选区并未找回最多的那222个压缩新增漏检，研究目标仍需与评价对象对齐。','',
 '五、为什么不继续二阶增量','',
 f"DA独立与二阶动作有{loss['arms']['DA']['dynamic']['independent_vs_quadratic_different_images']}图不同；平均相对16动作损失最优的regret从{loss['arms']['DA']['dynamic']['mean_loss_regret']['independent']:.6f}降至{loss['arms']['DA']['dynamic']['mean_loss_regret']['quadratic']:.6f}。",
 f"在未用于构造系数的三/四区组合上，效用近似MAE从{loss['arms']['DA']['dynamic']['mean_holdout_additive_MAE']:.6f}降至{loss['arms']['DA']['dynamic']['mean_holdout_quadratic_MAE']:.6f}。这些是损失代理的拟合改善，不是检测增益。",
 'GT独立/二阶均依赖真实有效GT计算native one2many v8DetectionLoss；原生损失没有内部COCO crowd ignore，且动态分配/软目标分母会随输入变化，因此它只是一种监督代理。',
 f"将JPEG的分配框、软目标、前景掩码和分母固定后重算损失，DA二阶动作有{loss['arms']['DA']['dynamic_vs_fixed_quadratic_different_images']}图改变。用这些固定分配损失选择动作（事后探索，仍复用既有预测）得到：",
 '固定分配选择 | DA小目标AP(%)']
for name,v in fixed['metrics']['DA'].items():lines.append(f"{name} | {100*v['AP_small_input']:.4f}")
lines+=['','固定分配下二阶与独立AP差约+0.0015pp，也没有恢复出二阶必要性。这不是证明动态分配造成了原问题，只说明选择标签定义会显著改变动作，而二阶损失拟合优势未转化为AP优势。',
 '本轮没有用后验loss、单个正例、额外网格或新训练挽救C10。','',
 '六、剩余问题、最近工作及可能增量','',
 '【问题线索C11，未批准为算法主线】普通JPEG适配后，GT局部选择相对全局处理尚有作用空间；部署可见的廉价任务信号，能否预测哪些区域值得恢复？',
 '这与“区域交互不为零”不同；本次数据支持先检查独立决策，而不是继续拟合成对项。',
 'TaskGuard已包含真实区域反事实、伪标签输入梯度与整图效用预测；PixelRL已有空间动作和不处理动作；RFDF已有识别相关区域与多代理融合；LQTARanker已有全图检测收益排序。因此“局部门控/学习效用/用检测指导恢复”均不能单独作为新增贡献。',
 '【待验证】TaskGuard式一阶伪梯度和原始/恢复检测响应是否已经足够。如果简单无训练代理达到这部分局部空间，直接采用已有机制，继续停止新算法包装。',
 '【可能增量，但未认可】只有在定位到现有任务信号的具体失败后，才讨论新的区域效用监督或可部署预测。没有将“使用区域反事实训练一个小网络”直接算作创新。RFDF完整方法仍未取得，其机制重合风险没有排除。',
 '【最小下一步，尚未运行】复用本次PNG与权重，比较无GT区域伪梯度选择、区域检测响应选择和整图伪梯度选择；不拟合新网络。需要时预算两模型各100图，共最多600次检测前向和200次输入反向，恢复0次、训练更新0。参数/阈值在运行前冻结，选择不得读GT；GT只用于事后评价。',
 '若这些现成信号不足，也仍须核最近方法与改进逻辑；不能由代理失败自动推出新方法有价值。','',
 '七、可核验资产和来源','',f'归档：{archive}',f'归档SHA256：{receipt["sha256"]}',
 f'原始结果：{root/"completed.json"}',f'区间及逐类AP：{root/"bootstrap_completed.json"}',f'固定分配选择：{root/"fixed_choice_metrics.json"}',f'损失诊断：{root/"loss_analysis.json"}',
 f'逐图19动作损失、全部预测、100张无损恢复图、源码/方案/失败日志均在：{root}',
 f'本地已逐一核验{len(manifest)}个归档文件的大小和SHA；两个冻结脚本/方案SHA与正式回执一致。',
 'TaskGuard（方法与附录选读）：https://arxiv.org/html/2609.08011v1',
 'PixelRL（此前方法p2–4选读）：https://ojs.aaai.org/index.php/AAAI/article/view/4240',
 'RFDF（仅出版商摘要/片段）：https://www.sciencedirect.com/science/article/pii/S0263224126031349',
 'LQTARanker（仅出版商预览，online日期未独立核清）：https://www.sciencedirect.com/science/article/pii/S0925231226022137',
 'FBCNN源码：https://github.com/jiaxi-jiang/FBCNN/tree/54d1831927506b3247e2d4d245abb4f4dab1a1cd','',
 '当前结论：C10二阶必要性未通过；C11仅为有实测依据的进一步问题；整体“值得继续的独立算法主线”目标尚未完成。']
report=base/'18_区域恢复判别结果与主线裁决_20261029.txt';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
verification=dict(status='ARCHIVE_AND_ALL_251_FILES_VERIFIED',sha256=receipt['sha256'],bytes=receipt['bytes'],files=len(manifest),training_updates=0)
(folder/'regional_backup_verification.json').write_text(json.dumps(verification,indent=2))
ledgerpath=local/'execution_ledger.json';ledger=json.loads(ledgerpath.read_text(encoding='utf-8-sig'))
ledger.update(region_probe=dict(status='COMPLETED',scientific_detector_forwards=3800,new_restorer_forwards=99,unique_restoration_images=100,training_updates=0,local_backup=verification,
    C10_joint_necessity_screen='NOT_PASSED',P3_deployable_selection_validated=False),goal_status='active',mainline_validated=False,
    current_lead='C11 independent regional utility predictability, not yet an approved novel algorithm',snapshot_time=now,latest_report=str(report))
ledgerpath.write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf-8')
board=dict(goal_status='active',snapshot_time=now,C10=dict(status='DO_NOT_DEVELOP_QUADRATIC_POLICY',reason='Joint minus independent -0.0926pp; 95% image CI [-0.2750,0.2045]; fixed-assignment comparison also near zero',scope='four regions, frozen FBCNN, Q20, YOLO26s fixed100, diagnostic100 single seed'),
           C11=dict(status='EVIDENCE_LEAD_ONLY',question='Can deployment-visible simple task signals predict useful independent restoration regions?',known_potential='GT independent minus strongest fixed-global alpha +0.6115pp; not deployable',algorithm_increment_approved=False,next='frozen no-GT pseudo-gradient and detector-response controls before any new policy training'))
(base/'主线目标_20261029_v1/candidate_board.json').write_text(json.dumps(board,ensure_ascii=False,indent=2),encoding='utf-8')
index=base/'00_调研报告.txt';old=index.read_text(encoding='utf-8-sig')
index.write_text('最新主线裁决（2026-10-29）：18_区域恢复判别结果与主线裁决_20261029.txt。四区二阶增量未通过，不训练该策略；GT独立区域选择尚有空间但新颖性/可部署性未证实。持续目标ACTIVE。简单编码对照见17。以下旧状态保留历史。\n\n'+old,encoding='utf-8')
goaltext=base/'主线目标_20261029_v1/00_目标与验收标准.txt';old=goaltext.read_text(encoding='utf-8')
goaltext.write_text('更新：冻结区域判别已完成，C10二阶候选未通过，详情见18。旧“进行中”段落保留为启动时记录；当前C11仅为剩余问题线索，整体目标仍ACTIVE。下一步600次前向/200次输入反向的无GT代理对照尚未运行。\n\n'+old,encoding='utf-8')
print(json.dumps(dict(status='REGIONAL_REPORT_AND_ACTIVE_GOAL_LEDGER_UPDATED',report=str(report),files_verified=len(manifest),goal_complete=False)))
