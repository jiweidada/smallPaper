"""Exclusive-create handoff at the explicit 343 pending-main-control gates."""
import importlib,json,time
C=importlib.import_module('343_context')

def read(p):return json.loads(C.Path(p).read_text(encoding='utf-8'))
def link(p):return f'[{C.Path(p).name}]({C.Path(p).as_posix()})'

def main():
    start=time.perf_counter();expected={};origins={};conflicts=[]
    manifests=[C.S2/'342_交付清单_SHA256.json',C.S2/'342_闸门封存清单_SHA256.json',
        C.S3/'342_阶段3_交付封存SHA256.json',C.S3/'342_阶段3_封存补充SHA256.json']
    for manifest in manifests:
        m=read(manifest)
        for field in ('artifacts','source_sha256'):
            for path,value in m.get(field,{}).items():
                digest=value['sha256'] if isinstance(value,dict) else value
                if path in expected and expected[path]!=digest:conflicts.append({'path':path,'before':expected[path],'after':digest})
                expected[path]=digest;origins.setdefault(path,[]).append(str(manifest))
    pre=read(C.HERE/'343_result_3b来源预检.json')
    for r in pre['entries']:
        expected[r['path']]=r['current_340_file_sha256'];origins[r['path']]=['343_result_3b来源预检.json（本轮预检封存）']
    assert not conflicts,conflicts
    checked=[];bad=[]
    for path,digest in expected.items():
        current=C.sha(path);ok=current==digest
        checked.append({'path':path,'expected_sha256':digest,'actual_sha256':current,'equal':ok,'evidence':origins[path]})
        if not ok:bad.append(path)
    assert not bad,bad
    oldgeometry=(C.S3/'342_geometry_stage3_v2/geom342.py').read_text(encoding='utf-8')
    newgeometry=(C.HERE/'343_geometry/geom342.py').read_text(encoding='utf-8')
    assert newgeometry.startswith(oldgeometry)
    C.write('343_result_旧缓存与源码不变核对.json',{'status':'已验证','checked_count':len(checked),'mismatches':bad,
        'entries':checked,'approved_geometry_original_text_preserved_as_prefix':True,
        'note':'本轮340整文件封存不等于历史raw final300已被SHA覆盖；来源闸门仍待主控。',
        '已执行':{'file_SHA_comparisons':len(checked),'geometry_prefix_comparisons':1,'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
        '未执行':['任何旧文件写入','development重建','真实训练','阶段5'],'未核':['主控独立核对'],'source_sha256':C.sources()})
    evalresult=read(C.HERE/'343_result_阶段4全量评估.json');history=[];hist_sources={}
    solver=C.Path(r'C:\Users\jiwei\Desktop\DEIM\engine\solver\det_solver.py')
    solverlines=solver.read_text(encoding='utf-8').splitlines()
    code=[{'line':i+1,'text':line.strip()} for i,line in enumerate(solverlines) if 'module = self.ema.module if self.ema else self.model' in line]
    hist_sources[str(solver)]=C.sha(solver)
    for seed in (1,2):
        log=C.DATA/f'本次24轮双基线模型下载_20261009_v1/seed{seed}/run/log.txt'
        config=C.DATA/f'本次24轮双基线模型下载_20261009_v1/seed{seed}/bundle/resolved.yml'
        lines=log.read_text(encoding='utf-8').splitlines()
        selected=[(i+1,json.loads(line)) for i,line in enumerate(lines) if line.strip() and json.loads(line).get('epoch')==23]
        assert len(selected)==1;line,row=selected[0]
        cfg=read(config);actual=evalresult['per_seed'][seed-1]['cases']['baseline']['stats12']['AP']
        history.append({'seed':seed,'log_path':str(log),'log_line':line,'epoch':23,'training_log_stats12':row['test_coco_eval_bbox'],
            'training_log_AP_full_precision':row['test_coco_eval_bbox'][0],'cached_raw_AP':actual,
            'AP_delta':actual-row['test_coco_eval_bbox'][0],'resolved_use_ema':cfg['use_ema'],
            'identity_limit':'本轮缓存是checkpoint[model] raw。resolved启用EMA，当前本机solver评估优先EMA；历史日志是否实际由同一solver/EMA产生未核，不能据此确定偏差原因。'})
        hist_sources[str(log)]=C.sha(log);hist_sources[str(config)]=C.sha(config)
    C.write('343_result_阶段4历史AP精确对照.json',{'status':'数值已核；偏差原因未核','per_seed':history,
        'current_local_solver_evaluation_selection':code,'replaces':'343_result_阶段4全量评估.json中仅用于对照的截断training_log_AP；本轮12项COCO统计未更改。',
        '已执行':{'training_log_reads':2,'resolved_config_reads':2,'local_solver_source_reads':1,'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
        '未执行':['切换EMA评估','修改评估设置','新模型前向','真实训练'],'未核':['历史训练日志模型选择身份','AP差异因果'],
        'source_sha256':C.sources()|hist_sources})
    mapping=read(C.HERE/'343_result_阶段4映射与融合_v3.json')
    geo=read(C.HERE/'343_result_阶段4Geo0资格.json');boot=read(C.HERE/'343_result_阶段4bootstrap.json')
    direction=read(C.HERE/'343_result_阶段4方向规则.json');closeups=read(C.HERE/'343_result_阶段4Geo0近景PNG.json')
    unresolved=[
        {'id':'3b_source','basis':'343第二节1：若找不到能核对的原始来源，停下写待主控，不得自行重跑模型。',
         'finding':'6470条340内置子集SHA全部通过，raw final300与339全部逐位一致；但两版内置SHA均不覆盖final300_boxes/logits。只有seed1前32条有阶段2历史整文件SHA，其余6438条缺此历史绑定。',
         'needed':'指定覆盖raw final300的历史清单，或明确接受现有子集SHA+339逐位对照+本轮新封存来源链。',
         'dependent_work':'development重建、全量独立验收、旧340差异/新统计、新标签冒烟及续跑均未执行。'},
        {'id':'formal_TP_policy','basis':'343第三节4：317第7节有歧义处停下写待主控，不得自选口径。',
         'finding':'新增覆盖参照、先对全GT还是资格组匹配、同分顺序及IoU并列顺序未固定。',
         'proposal_not_adopted':'统一输出按同真类score降序对全部GT做1对1 IoU>=.5匹配，再筛资格组；新增/丢失相对同seed global-only统一后处理baseline；同分用固有flatten索引、IoU并列用annotation_id。',
         'needed':'确认该口径或给出主控口径；未收到答复前不执行正式B/FP/原TP新增丢失等计数。'}]
    C.write('343_阶段3b_result.json',{'status':'待主控；重建未执行','source_preflight':str(C.HERE/'343_result_3b来源预检.json'),
        'preflight_summary':pre['summary'],'待主控':[unresolved[0]],
        '已执行':pre['已执行'],'未执行':['development3235×2新M0重建','全量独立验收','旧340字段差异与新Wilson/overflow统计','新标签四臂20步冒烟及续跑','6张development PNG','真实训练','GPU','阶段5'],
        '未核':['6438条raw final300的历史SHA绑定'],'source_sha256':C.sources()})
    evidence=[C.HERE/n for n in ('343_result_阶段4映射与融合_v3.json','343_result_阶段4全量评估.json',
        '343_result_阶段4历史AP精确对照.json','343_result_阶段4Geo0资格.json','343_result_阶段4Geo0近景PNG.json',
        '343_result_阶段4bootstrap.json','343_result_阶段4方向规则.json','343_result_旧缓存与源码不变核对.json')]
    checklist=[
        {'item':'1 融合/映射','status':'已验证','evidence':str(evidence[0])},
        {'item':'2 Windows绑定源码评估器/12项COCO','status':'已验证；.pyd与.so差异已披露','evidence':str(evidence[1])},
        {'item':'3 正式B/全GT Geo0 TP/IoU.75/FP/原TP新增丢失','status':'待主控；仅资格组已验证','evidence':str(evidence[3])},
        {'item':'4 配对bootstrap','status':'构造数据1000次已验证；正式B部分待主控，训练后数据未执行','evidence':str(evidence[5])},
        {'item':'5 十二项方向规则','status':'5类构造用例已验证；科学判定未执行','evidence':str(evidence[6])},
        {'item':'6 baseline与历史AP对照','status':'已执行；AP未复现历史值，原因未核','evidence':str(evidence[2])},
        {'item':'7 低分new4/随机new4合成端到端','status':'两seed已验证，低分12项指标逐位相同','evidence':str(evidence[1])}]
    C.write('343_阶段4_result.json',{'status':'部分已验证；正式计数待主控，阶段4未完成','checklist':checklist,
        'geometry':{'independent_max_abs_diff':mapping['max_independent_abs_diff'],'round_trip_max_abs_diff':mapping['max_round_trip_abs_diff'],
            'two_seed_cache_entries':1096,'same_score_native_topk_reordered_images':len(mapping['native_topk_tie_reordered_images']),
            'low_new4_score_max_abs_diff':max(r['low_new4_score_max_abs_diff'] for r in mapping['per_image']),
            'frozen_complete_tolerance':'保持1e-6，不改G_W或选窗'},
        'baseline_AP_comparison':history,'tiny_Geo0_eligibility':geo['stats'],'待主控':[unresolved[1]],
        'visual_QA':{'viewed_mapping_PNG':mapping['png_paths'],'viewed_Geo0_overview_PNG':geo['png_paths'],
            'viewed_Geo0_closeup_PNG':closeups['png_paths'],'observation':'左右独立映射位置一致；资格组红框在固定窗内；seed1#534和seed2#210资格组为0，右近景无红框。PNG为几何构造/资格显示，不代表补全预测。'},
        'repair_history':[str(C.HERE/'343_result_映射首轮失败.json'),str(C.HERE/'343_result_映射第二轮失败.json')],
        '已执行':{'successful_mapping_case_checks':1096,'successful_native_PostProcessor_calls':1096,
            'val_COCO_evaluate_calls':6,'constructed_bootstrap_COCO_evaluate_calls':10,
            'constructed_bootstrap_replications':1000,'bootstrap_COCO_accumulate_calls':boot['已执行']['bootstrap_COCO_accumulate_calls'],
            'direction_rule_calls':5,'independent_Geo0_group_checks':1096,'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
        '未执行':['正式B及其余有歧义计数','任何训练后产物评估','真实训练','GPU','SSH','队列','heldout3236读取','阶段5'],
        '未核':['主控独立复核','AP差异因果','Windows.pyd与Linux.so一致性'],
        'result_evidence':{str(p):C.sha(p) for p in evidence},'source_sha256':C.sources()})
    C.text('343_阶段3b_中文汇总.md',f'''阶段3b尚未重建，按343第二节1停在来源核对。

已逐条读取6470个340缓存和6470个339缓存。两版内置子集SHA全部通过，raw final300_boxes/logits全部逐位一致。关键限制是两版内置SHA均不覆盖这两个原始张量；只有seed1前32条可绑定阶段2历史整文件SHA，其余6438条缺历史原始来源SHA。本轮计算的SHA只用于当前封存，不冒称历史证明。

证据：{link(C.HERE/'343_result_3b来源预检.json')}；{link(C.HERE/'343_阶段3b_result.json')}。

待主控指定历史清单，或裁定是否接受“有效子集SHA＋339原始张量逐位对照＋本轮新封存”的来源链。此前提交的问题尚未收到答复。development重建、6470条独立验收、旧340差异表、新Wilson/overflow统计、新标签四臂冒烟/续跑和6张PNG均未执行。旧溢出数字未更新。

阶段3放行和G_W的1e-6冻结容差已作为输入接受；没有改容差、窗、阈值或标签定义。本轮global前向、head前向、optimizer步数均为0；未启动真实训练、GPU或阶段5。
''')
    ap='\n'.join(f'| {r["seed"]} | {r["cached_raw_AP"]:.12f} | {r["training_log_AP_full_precision"]:.12f} | {r["AP_delta"]:.12f} |' for r in history)
    C.text('343_阶段4_中文汇总.md',f'''阶段4部分已验证，正式B计数按343第三节4待主控，尚未完成。

两个seed共1096条val v2缓存通过共享geom342反变换、独立中心/半宽高构造、往返及错误格式测试；最大独立差{mapping['max_independent_abs_diff']:.3g}，往返差{mapping['max_round_trip_abs_diff']:.3g}。批准的geom342原文完整保留，仅在新目录另存反变换扩展。6张映射PNG及6张资格近景PNG均已目视检查。冻结1e-6容差保持。

只用记录的using_tiny_metrice=False、using_vtiny_metrice=False、max_dets=100和本机Windows绑定源码评估器。.pyd与receipt的Linux.so不同。两seed的低分new4与baseline共12项统计逐位相同，随机new4评估完成。

| seed | 本轮raw AP | epoch23日志AP | 差值 |
| --- | --- | --- | --- |
{ap}

差异如实保留，未调整设置。resolved启用EMA，当前本机solver评估优先EMA；历史运行身份及偏差因果未核，未切换EMA或新做模型前向。

tiny Geo0资格组独立核对为586/625个GT；这不是正式新增TP。B新增覆盖参照、匹配及并列顺序未固定，未自行采用口径。配对bootstrap的1000×548共享抽样在构造数据上完成，8次独立完整克隆评价的precision/recall逐位相同。12项方向规则的全正、零、负、跨seed抵消、G未有效校准脚注5类构造通过；没有科学效果判定。

首两轮映射测试把预测逐位相同误作指标合同而失败；原生topk同分顺序和sigmoid末位差已记录，未修改融合算子来凑一致。完整证据及待主控项见{link(C.HERE/'343_阶段4_result.json')}。本轮所有模型前向和optimizer步数为0。阶段5未执行。
''')
    lines=['本轮交接：阶段3b待主控；阶段4部分已验证、正式计数待主控。两项尚未完成，不进入阶段5。','',
        '授权依据：'+link(C.SPEC/'343_授权_阶段3放行_阶段4与阶段3b.txt')+'。','',
        '阶段结果与汇总：','']
    for name in ('343_阶段3b_result.json','343_阶段3b_中文汇总.md','343_阶段4_result.json','343_阶段4_中文汇总.md'):
        lines.append('- '+link(C.HERE/name))
    lines+=['','需主控裁定：','',
        '1. 3b来源：6470条内置子集SHA有效且与339原始张量逐位相同，但仅32条有历史整文件SHA覆盖raw；其余6438条需历史清单，或明确接受现有来源链。依据343第二节1必须停。',
        '2. 正式计数：B新增覆盖参照、全GT/资格组的匹配先后、同分及IoU并列顺序。提议仅作为待裁定文本保存在result，未实施。依据343第三节4必须停。','',
        '阶段4逐项证据：','']
    lines.extend('- '+link(p) for p in evidence)
    lines+=['','主控目视优先看：','']
    lines.extend('- '+link(p) for p in mapping['png_paths']+closeups['png_paths'])
    lines+=['','原始/失败产物全部保留。有效映射结果是v3；v1/v2失败记录仅用于复盘。','',
        f'旧文件与缓存SHA核对{len(checked)}项均未变；本轮raw模型/新头前向及optimizer步数均为0。无真实训练、GPU、SSH、队列或阶段5。','',
        '完整当前目录SHA清单：'+link(C.HERE/'343_交付封存SHA256.json')+'。']
    C.text('343_交接索引.md','\n'.join(lines)+'\n')
    artifacts={str(p):{'sha256':C.sha(p),'bytes':p.stat().st_size} for p in C.HERE.rglob('*') if p.is_file()}
    C.write('343_交付封存SHA256.json',{'status':'本轮待主控封存；不是阶段5审查包','artifacts':artifacts,
        'source_sha256':C.sources(),'待主控':unresolved,
        '已执行':{'artifact_file_hashes':len(artifacts),'protected_old_file_hashes':len(checked),'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
        '未执行':['development重建','正式B计数','真实训练','阶段5'],'未核':['主控独立核对'],
        'wall_seconds_finalization':time.perf_counter()-start})
    print(json.dumps({'status':'待主控','protected_files_equal':len(checked),'artifacts_sealed':len(artifacts),
        'baseline_AP':[h['cached_raw_AP'] for h in history],'index':str(C.HERE/'343_交接索引.md')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
