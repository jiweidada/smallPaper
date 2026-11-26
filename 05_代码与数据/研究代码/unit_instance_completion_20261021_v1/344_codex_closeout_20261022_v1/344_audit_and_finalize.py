"""Seal 344 closeout only; no phase-5 package and no training dispatch."""
import importlib,json,ast,time
C=importlib.import_module('344_context')

def link(name,root=None):
    p=(C.HERE if root is None else root)/name
    return f'[{name}]({p.as_posix()})'

def main():
    start=time.perf_counter()
    legacy=json.loads((C.S343/'343_result_本轮不覆盖与语法核对_v2.json').read_text(encoding='utf-8'))
    seal=json.loads((C.S343/'343_交付封存SHA256_v2.json').read_text(encoding='utf-8'))
    expected={r['path']:r['expected_sha256'] for r in legacy['entries']}
    for p,v in seal['artifacts'].items():
        if p in expected:assert expected[p]==v['sha256']
        expected[p]=v['sha256']
    for p,v in seal.get('source_sha256',{}).items():
        if p in expected:assert expected[p]==v
        expected[p]=v
    # Do not open any heldout file, even if an unrelated historical manifest names it.
    assert not any('heldout' in p.lower() or 'holdout' in p.lower() for p in expected)
    checks=[]
    for n,(p,h) in enumerate(sorted(expected.items()),1):
        actual=C.sha(p);assert actual==h,('old file changed',p)
        checks.append({'path':p,'expected_sha256':h,'actual_sha256':actual})
        if n%1024==0 or n==len(expected):print(f'344 immutable audit {n}/{len(expected)}',flush=True)
    syntax=[]
    for p in C.HERE.rglob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'));syntax.append(str(p))
    assert (C.HERE/'344_geometry/geom342.py').read_text(encoding='utf-8').startswith((C.S343/'343_geometry_v2/geom342.py').read_text(encoding='utf-8'))
    assert C.sha(C.SPEC/'344_授权_1466裁定_EMA诊断_3b收尾.txt')=='95ca0e5c0d9dc479f5fb9167909996eb8886bc6e52ccc91081164309d37653e6'
    C.write('344_result_旧文件不覆盖与语法审计.json',{'status':'已验证','checked_old_files':len(checks),'old_9077_ledger_entries':len(legacy['entries']),
        '343_sealed_artifact_entries':len(seal['artifacts']),'entries':checks,'new_Python_AST_paths':syntax,
        '343_geometry_prefix_byte_text_preserved':True,'344_authorization_unchanged':True,
        '已执行':{'old_file_SHA_checks':len(checks),'new_Python_AST_parses':len(syntax),'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
        'wall_seconds':time.perf_counter()-start,'未执行':['真实训练','GPU','SSH','队列','heldout文件读取','阶段5'],
        '未核':['主控独立复核'],'source_sha256':C.sources()})
    def read(n):return json.loads((C.HERE/n).read_text(encoding='utf-8'))
    complete=read('344_result_6470补全与非M0对账.json');scan=read('344_result_非正框全量扫描.json')
    independent=read('344_result_1466独立验收与6PNG_v2.json');visual=read('344_result_6PNG本轮目视.json')
    native=read('344_result_1466_native预算核对.json');bound=read('344_result_train344全量输入与入口验收.json')
    smoke=read('344_result_train344四臂CPU冒烟与续跑.json');ema=read('344_result_EMA诊断_v2.json')
    coco=read('344_result_独立pycocotools_baseline_v2.json')
    assert len(complete['per_entry'])==6470 and len(scan['fullP_hits'])==1 and len(independent['png_paths'])==6 and len(visual['entries'])==6
    assert smoke['已执行']['head_forwards']==198 and smoke['已执行']['optimizer_steps']==192
    assert ema['已执行']['global_forward_calls']==1096 and bound['已执行']['successful_bound_loads']==6470
    failures=[
        {'receipt':'344_result_EMA诊断_实现失败.json','reason':'新增诊断参数量常数少写2；前向0次。v2改为与实际EMA named_parameters逐项总数核对，严格加载794项，实测10231997。','resolved_by':'344_result_EMA诊断_v2.json'},
        {'receipt':'344_result_pycocotools首次实现失败.json','reason':'本机pycocotools.loadRes要求info元数据；v2仅在内存补info={}，标注/预测/指标参数不改。','resolved_by':'344_result_独立pycocotools_baseline_v2.json'},
        {'receipt':'344_result_1466独立验收_实现失败.json','reason':'本机Pillow无Image.Resampling；v2使用已安装版本的Image.BILINEAR/NEAREST枚举。首次PNG写入0张。','resolved_by':'344_result_1466独立验收与6PNG_v2.json'},
    ]
    for f in failures:assert read(f['receipt'])['status']=='实现失败'
    aggregate={
        'CPU_global_forwards_EMA_only':1096,'CPU_raw_global_forwards':0,
        'CPU_head_forwards':198,'bounded_CPU_optimizer_steps':192,'native_loss_adapter_calls':6336,
        'smoke_resume_checkpoint_saves':8,'smoke_resume_checkpoint_loads':8,
        'new_development_cache_writes':1,'development_fullP_scope_scans':6470,'development_non_M0_old340_comparisons':6470,
        'train344_successful_full_dataset_loads':6470,'train344_successful_smoke_loads':80,'train344_successful_native_check_loads':1,
        'native_PostProcessor_calls':1,'EMA_COCO_evaluate_calls':2,'independent_pycocotools_COCO_evaluate_calls':2,
        'PNG_written':6,'PNG_visually_viewed':6,'real_training_steps':0,'GPU_calls':0,
    }
    limits=[
        '344是317非正框报错合同的明示例外：唯一seed2 #1466 query62 raw h<=0。fullP/native预算保留，P_W排除，IoU=0且M0无合法边；坐标未修正。该图G_W/U_W均为0，训练标签不变。',
        '主线仍使用checkpoint[model] raw。EMA只作CPU val诊断，未落盘预测，未进入任何缓存/M0/训练输入。EMA接近历史日志但12项未精确复现，剩余原因未核。',
        '本机Windows.pyd与历史receipt绑定Linux.so并非同一二进制；跨平台逐位等价未核。',
        'development两seed |U_W|>4：474图/3047溢出GT、468图/3199溢出GT；不变更4查询、标签或窗。',
        '两seed同窗1443/3235；冻结FP32 top300与1e-6完整窗语义保留，#448/#3222/#6383不修改。',
        'L_det分母为max(1,batch内完整U_W目标数之和)，沿用主控对296/317的解读；配方/阈值/学习率未改变。',
        '主线8个真实run及R对N/S/G的12项严格为正科学判据未执行。预检只核工程接口。',
    ]
    result={'status':'344三项已完成，等待主控核对；阶段5未授权','scope':'CPU-only closeout, phase 3b completion + EMA diagnosis + train344 binding',
        'items':{'1466_exception_and_6470':'已验证','independent_geometry_negative_controls_and_six_PNGs':'已验证且已目视',
                 'train344_full_dataset_SHA_binding':'已验证','two_seed_four_arm20step_and_bitwise_resume':'已验证',
                 '1466_batch_four_arm_finite_losses':'已验证','EMA_diagnosis':'已执行1096次CPU前向；历史12项尚未精确复现',
                 'independent_pycocotools_raw_baseline':'已验证12项完全一致','phase5_review_package':'未执行，等主控'},
        'development_stats':complete['stats'],'same_window_count':1443,
        'EMA_AP_raw_log':[{'seed':r['seed'],'raw_AP':r['raw_stats12']['AP'],'EMA_AP':r['EMA_stats12']['AP'],
                          'log_AP':r['training_log_stats12']['AP'],'EMA_minus_log_AP':r['EMA_minus_log']['AP'],
                          'max_abs_12_item_EMA_minus_log':r['max_abs_EMA_minus_log']} for r in ema['per_seed']],
        'EMA_conclusion':ema['conclusion'],'fixed_engineering_failures_preserved':failures,'known_limits_for_future_phase5':limits,
        '已执行':aggregate,'未执行':['真实训练','8个正式run','GPU','SSH','队列','heldout3236文件读取','阶段5审查包'],
        '未核':['主控对344的独立复核','EMA与历史12项残差的唯一原因','跨平台二进制逐位一致','主线科学优劣'],
        'gate':'停下等主控；放行后才写342_开训前审查包.md；真实训练仍需另行明确授权。',
        'evidence_paths':[str(C.HERE/n) for n in (
            '344_result_6470补全与非M0对账.json','344_development6470_逐条SHA清单.json','344_result_非正框全量扫描.json',
            '344_result_1466独立验收与6PNG_v2.json','344_result_6PNG本轮目视.json','344_result_1466_native预算核对.json',
            '344_result_train344全量输入与入口验收.json','344_result_train344四臂CPU冒烟与续跑.json',
            '344_result_EMA诊断_v2.json','344_result_独立pycocotools_baseline_v2.json','344_result_旧文件不覆盖与语法审计.json')],
        'source_sha256':C.sources()}
    C.write('344_result_总交付.json',result)
    e1,e2=result['EMA_AP_raw_log']
    summary=f'''# 344 一页中文汇总

344 授权三项已完成，现停在主控核对闸门。工程预检已支持提交主线最终开训审查；阶段5与真实训练均未执行。

6470条development输入已齐：6469条继续引用343只读缓存，补建seed2 #1466一条。全部条目SHA与旧340非M0字段对账通过，train344逐条实际加载通过。两个seed的U_W>0为1385/1389，总U_W为6791/6854；|U_W|>4为474/468图，溢出GT为3047/3199；同窗1443/3235。

全量fullP仅命中1个非正框：seed2 #1466、query62、fullP索引259。按344保留原坐标与fullP预算，native逐类排名24，max100后仍保留；P_W排除，M0合法边0、owner=-1。该图G_W=U_W=0，训练标签不变。独立几何/M0证书、5项反向测试、6张PNG与本轮目视均通过。

train344仅改新缓存/SHA绑定与P_W显式排除，collate和L_assoc使用相同mask。配方、损失公式、优化器、学习率路由保持原样。双seed各前32窗、四臂各20步与8组续跑全部通过；20步日志与343逐位一致。另含#1466的16图batch在四臂前向loss均有限。实际198次head前向、6336次loss适配器调用、192次有界CPU冒烟优化步；真实训练0步。

| seed | raw AP | EMA AP | 历史日志 AP | EMA−日志 |
|---|---:|---:|---:|---:|
| 1 | {e1['raw_AP']:.12f} | {e1['EMA_AP']:.12f} | {e1['log_AP']:.12f} | {e1['EMA_minus_log_AP']:.12f} |
| 2 | {e2['raw_AP']:.12f} | {e2['EMA_AP']:.12f} | {e2['log_AP']:.12f} | {e2['EMA_minus_log_AP']:.12f} |

EMA按checkpoint['ema']['module']严格加载，完成val548×2=1096次CPU前向；预测仅留内存供诊断，未进入缓存/M0/训练输入。EMA大幅缩小历史AP差距，但12项未精确复现，最大绝对残差0.000557261/0.000583622，原因未完全核定。主线继续按296用raw。另用本机独立pycocotools重评343 raw预测，12项与343完全相同。

本机Windows.pyd与历史Linux.so的二进制差异、EMA剩余残差、4查询溢出与#1466明示例外均须写进后续阶段5“已知限制”。3次工程实现失败（参数量常数、pycocotools info元数据、旧Pillow枚举）保留原记录与修正diff，均已用另存v2完成验证。旧文件{len(checks)}个SHA检查通过。

主控独立核对344后才写342_开训前审查包.md；审查包通过后仍需另行授权真实训练。R对N/S/G的12项科学判据未执行，不由预检推断。

证据入口：{link('344_交接索引.md')}；逐项状态与调用数：{link('344_result_总交付.json')}。
'''
    C.text('344_一页中文汇总.md',summary)
    lines=['# 344 交接索引','',
        '状态：344授权三项已完成，停下等主控核对。阶段5未执行，真实训练仍未授权。', '',
        f'先读 {link("344_一页中文汇总.md")} 与 {link("344_result_总交付.json")}。本目录独占新建，旧342/343与规格文件只读。','',
        '## 6470条与 #1466','',
        f'- {link("344_result_非正框全量扫描.json")}：6470条fullP扫描，仅1个授权实例；未入fullP的raw非正查询另计。',
        f'- {link("344_cache_development_seed2/344_001466.pt")}：唯一补建缓存；其余6469条按下面清单引用343原文件。',
        f'- {link("344_development6470_逐条SHA清单.json")}：6470条文件SHA/payload SHA/来源SHA和非正mask索引。',
        f'- {link("344_result_6470补全与非M0对账.json")}：全部6470条旧340非M0字段逐位核对、新统计与逐图披露。',
        f'- {link("344_result_1466独立验收与6PNG_v2.json")}：独立NumPy几何/代价/M0证书/反向测试。',
        f'- {link("344_result_6PNG本轮目视.json")}：6张PNG路径、SHA与逐张目视结论。',
        f'- {link("344_result_1466_native预算核对.json")}：native PostProcessor逐位一致；坏框逐类第24，预算后仍在fullP。','',
        '## train344绑定、有限冒烟与续跑','',
        f'- {link("train344.py")}，{link("344_train344_vs_train342.diff")}：SHA清单固定绑定；不读取旧340或EMA作为训练输入。',
        f'- {link("344_geometry/geom342.py")}，{link("344_geom342_vs343.diff")}：343已有函数原文保留，只增344明示例外接口。',
        f'- {link("344_result_train344全量输入与入口验收.json")}：6470次实际加载、9项反向检查、入口静态检查与配方AST对照。',
        f'- {link("344_result_train344四臂CPU冒烟与续跑.json")}：两seed四臂20步、8组续跑逐位、#1466 batch四臂有限loss；完整逐步日志。',
        f'- {link("344_smoke_train344.py")}：只用前32窗的有界CPU测试，#1466另做一次16图batch前向。','',
        '## raw/EMA/历史与独立评价','',
        f'- {link("344_result_EMA诊断_v2.json")}：两seed12项完整并列、差值、实际权重路径与state SHA；1096次CPU前向。',
        f'- {link("344_ema_diagnostic_v2.py")}：EMA预测仅内存评价，未生成预测缓存。',
        f'- {link("344_result_独立pycocotools_baseline_v2.json")}：独立安装的pycocotools在343 raw baseline预测上12项完全一致。',
        'EMA大幅缩小历史差距，但未逐项精确复现。剩余原因未核；不调整预处理/阈值/评估设置来凑数，主线保持raw。','',
        '## 实现失败记录与封存','']
    for f in failures:lines.append(f'- {link(f["receipt"])}：{f["reason"]} 对应通过记录 {link(f["resolved_by"])}。')
    lines.extend(['',f'- {link("344_result_旧文件不覆盖与语法审计.json")}：{len(checks)}个旧文件SHA一致、新Python语法通过。',
        f'- {link("344_交付封存SHA256.json")}：本目录最终清单，封存文件本身不纳入自身SHA。','',
        '## 后续闸门与已知限制',''])
    for x in limits:lines.append('- '+x)
    lines.extend(['','下一步仅等主控独立核对与阶段5放行；本轮不生成342_开训前审查包.md，不给出GPU或真实训练命令。',''])
    C.text('344_交接索引.md','\n'.join(lines))
    artifacts={str(p):{'sha256':C.sha(p),'bytes':p.stat().st_size} for p in sorted(C.HERE.rglob('*')) if p.is_file()}
    C.write('344_交付封存SHA256.json',{'status':'344已交付，等主控；阶段5未执行','artifacts':artifacts,
        '已执行':aggregate,'未执行':result['未执行'],'未核':result['未核'],'source_sha256':C.sources()})
    print(json.dumps({'status':'PASS; stop for controller','old_SHA_checks':len(checks),'new_artifacts_sealed':len(artifacts),
        'calls':aggregate,'EMA':result['EMA_AP_raw_log']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
