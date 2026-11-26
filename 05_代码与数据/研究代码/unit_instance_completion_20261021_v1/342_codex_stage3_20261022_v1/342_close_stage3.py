"""Seal stage 3 and stop. No evaluation or training invocation."""
import importlib,json,ast,re
C=importlib.import_module('342_stage3_context')

def read(name):return json.loads((C.HERE/name).read_text(encoding='utf-8'))

def main():
    q=read('342_result_val548独立验收.json');assert q['已执行']['complete_cache_loads']==1096
    manifest=read('342_val548_逐条SHA清单.json')['entries'];assert len(manifest)==1096
    assert all(C.sha(row['path'])==row['file_sha256'] for row in manifest)
    # All 434 pre-existing source/spec hashes, previous delivery, and 32 old
    # validation entries are checked without altering their historical reports.
    protected=json.loads((C.PREV/'342_输入快照.json').read_text(encoding='utf-8'))['source_sha256']
    sealed=json.loads((C.PREV/'342_闸门封存清单_SHA256.json').read_text(encoding='utf-8'))['artifacts']
    protected.update({p:r['sha256'] for p,r in sealed.items()})
    for row in read('342_result_阶段3输入预检.json')['rows']:protected[row['cache_path']]=row['cache_file_sha256']
    changed=[p for p,h in protected.items() if C.sha(p)!=h];assert not changed,changed
    geom=C.HERE/'342_geometry_stage3_v2'/'geom342.py'
    assert geom.read_bytes().startswith((C.PREV/'geom342.py').read_bytes())
    raw=[read(f'342_raw_seed{s}_{tag}_result.json') for s in (1,2) for tag in ('part1','resumed')]
    assert sum(r['已执行']['global_forward_calls'] for r in raw)==1096
    assert all(r['raw_global_state_sha_before']==r['raw_global_state_sha_after'] and r['raw_global_all_grad_none'] for r in raw)
    timings=[{'receipt':f'342_raw_seed{s}_{tag}_result.json','seconds':read(f'342_raw_seed{s}_{tag}_result.json')['elapsed_seconds']}
             for s in (1,2) for tag in ('part1','resumed')]
    build_names=['342_full_seed1_part1_result.json','342_full_seed2_part1_result.json',
        '342_full_seed1_v2_part1_result.json','342_full_seed1_v2_resumed_result.json',
        '342_full_seed2_v2_part1_result.json','342_full_seed2_v3_recovered_result.json']
    timings += [{'receipt':n,'seconds':read(n)['elapsed_seconds']} for n in build_names]
    # seed2 v2 interrupted segment has a recorded completed-entry count but no
    # elapsed-time receipt. Do not invent its duration.
    storage={}
    for seed in (1,2):
        rawfiles=list((C.HERE/f'342_raw_val_seed{seed}').glob('342_*.pt'))
        complete=[r for r in manifest if r['seed']==seed]
        storage[str(seed)]={'raw_entries':len(rawfiles),'raw_bytes':sum(p.stat().st_size for p in rawfiles),
                           'complete_entries':len(complete),'complete_bytes':sum(r['bytes'] for r in complete)}
    paths={'geom342':str(geom),'builder':str(C.HERE/'342_build_val_v3.py'),
           'M0':str(C.HERE/'342_m0_exact.py'),'independent':str(C.HERE/'342_val_independent.py'),
           'validator':str(C.HERE/'342_validate_val_v3.py'),
           'complete_caches':[str(C.HERE/f'342_cache_val_v2_seed{s}') for s in (1,2)],
           'raw_caches':[str(C.HERE/f'342_raw_val_seed{s}') for s in (1,2)]}
    result={'status':'已验证（阶段3 val548×2工程项）；待主控','gate':'STOP_AFTER_STAGE3',
        'paths':paths,'stats':q['stats'],'storage':storage,'timing_receipts':timings,
        'timing_scope':'各进程实测墙钟；存在并行重叠，不相加冒称总墙钟。中断失败段耗时未核。',
        'old341_summary':q['old341_summary'],'old341_32_comparison':q['old341_32_comparison'],
        'PNG_ids_per_seed':q['PNG_ids_per_seed'],'png_paths':q['png_paths'],
        'M0_definition':{'candidate_class':'fullP instance label; no query argmax substitution',
            'primary':'maximum cardinality on same-class native FP32 IoU>=0.5 edges',
            'secondary':'exact sum of native FP32 matcher cost values (dyadic rationals converted to integers)',
            'tertiary':'lex owner-GT-index vector sorted by original flatten(query,class) index; unmatched sentinel M',
            'proof':'242 brute-force small cases + per-entry independent maximum matching and exact reduced-cost dual certificate',
            'integer_storage':'certificate big integers stored as decimal strings; computation remains integer exact'},
        'preserved_files':{'checked':len(protected),'changed':changed,'approved_geometry_prefix_byte_equal':True},
        'visual_QA':{'inspected_final_PNGs':12,'observer':'Codex same execution session; not independent main-controller review',
            'observation':'两seed原图窗口与裁片位置对应；缓存与独立裁片目视一致；空G_W窗无K_g红圈；未见明显行列转置。'},
        'repair_history':['M0候选类及精确lex修复（用户明确授权）；旧缓存不覆盖。',
            '首版64条完整条目因大整数LONG4无法weights_only读取，保留为失败产物；正式缓存改用字符串证书。',
            'seed2原始query中3图7个非正尺寸query均未进入fullP；原始query转换/选窗恢复与旧管线逐位一致，候选正面积检查未放宽。',
            '验收重跑命名冲突按独占新建规则停止，改用新PNG前缀；未覆盖。'],
        '已执行':{'raw_global_forward_calls':1096,'raw_model_loads':4,
            'complete_cache_M0_solves':1160,'valid_complete_entries':1096,'preserved_failed_v1_entries':64,
            'exact_M0_small_cases':242,'old_val_real_M0_unit_cases':32,
            'successful_final_independent_validation':q['已执行'],
            'source_and_old_cache_preservation_hash_checks':len(protected),'final_cache_file_SHA_checks':1096,
            'head_forwards':0,'optimizer_steps':0,'PNG_final_generated_and_inspected':12},
        '未执行':['真实训练','GPU','SSH','队列','heldout3236读取','阶段4评估/融合/指标','阶段5最终审查包',
                  'development3235×2按修正M0重建与重新审查','四臂冒烟/续跑在修正M0标签上的重新执行'],
        '未核':['主控独立核对本阶段','中断失败段耗时','旧development标签修正后的逐条影响'],
        '待主控':['确认阶段3统计、M0逐项变化、两seed各6张PNG；确认后才进入阶段4。',
                  '开训前需安排development缓存采用同一修正M0版本；现有旧标签缓存不可当作修正M0合同已通过。'],
        'source_sha256':C.sources()}
    C.json_out('342_阶段3_result.json',result)
    table=[]
    names=[('G_W>0','G_W_positive'),('U_W>0','U_W_positive'),('|U_W|>4','U_W_gt4')]
    for label,key in names:
        cols=[]
        for seed in ('1','2'):
            z=q['stats'][seed][key];lo,hi=z['wilson95']
            cols.append(f"{z['count']}/548；{100*z['rate']:.2f}% [{100*lo:.2f}, {100*hi:.2f}]")
        table.append('| '+label+' | '+' | '.join(cols)+' |')
    summary='\n'.join(['# 342 阶段3中文汇总','',
        '状态：val548×2完整缓存与独立验收已完成，停在阶段3闸门，待主控。真实训练及阶段4/5未执行。','',
        '| 指标 | seed1：比例及Wilson 95%区间 | seed2：比例及Wilson 95%区间 |',
        '|---|---|---|',*table,'',
        f"U_W GT总数：seed1={q['stats']['1']['total_U_W_GT']}、seed2={q['stats']['2']['total_U_W_GT']}；四槽溢出GT：{q['stats']['1']['overflow_GT']}、{q['stats']['2']['overflow_GT']}。仅描述val，不与旧development统计混用。",'',
        '原始预测共1,096次CPU前向，所有冻结模型state不变、grad均None；完整缓存从预测复用，无额外模型前向。两seed均实测32条停下后恢复；第二seed还从174条完成中断恢复。',
        '窗/G_W/U_W/K_g/crop及M0合法边、最大基数、精确整数最优性证书已对1,096条逐条复核。K_g最大绝对差2.98e-8；crop最大绝对差seed1=4.10e-5、seed2=4.35e-5，低于运行前固定的FP32采样误差界。','',
        '经用户授权，M0按候选实例类修复，并落实原flatten索引上的精确三级lex。旧val32非M0字段全部逐位相同；30图owner变化，7图U_W变化，annotation_id差异见总结果。原缓存保留。','',
        '最终缓存目录为342_cache_val_v2_seed1/2；活动脚本为342_build_val_v3.py，活动几何为342_geometry_stage3_v2/geom342.py。请勿使用保留的首版64条失败产物。',
        '两seed各6张PNG、逐条SHA、脚本diff、失败与修复记录均见342_阶段3_交接索引.md。','',
        '前次两处表述已更正：[-0.349,1.303]只指seed1前16窗；旧恢复代码下一步会重设LR，新写法是状态清理，不能算实际更新错误修复。','',
        '开训前新增缺口：旧development3235×2仍沿旧M0生成，尚未按修正版本重建/复核；旧四臂冒烟记录也未在新标签下重跑。本阶段不得被当作训练授权。'])+'\n'
    C.text_out('342_阶段3_中文汇总.md',summary)
    index=f'''# 342 阶段3交接索引（停在闸门）

主结果：[342_阶段3_result.json](342_阶段3_result.json)。一页汇总：[342_阶段3_中文汇总.md](342_阶段3_中文汇总.md)。阶段4/5及真实训练均未执行。

活动源：`342_build_val_v3.py`、`342_geometry_stage3_v2/geom342.py`、`342_m0_exact.py`、`342_val_independent.py`、`342_validate_val_v3.py`。原规格、311_v2/320_v2、train340、stage1_cache_v3、上阶段交付和旧341缓存均未修改。

| 内容 | 证据 |
|---|---|
| 两处汇报更正、阶段1/2放行范围 | 342_放行与更正记录.json |
| M0修复及字段级差异授权 | 342_M0修复放行记录.json |
| 旧M0跨类owner：15个/10图，其中1个owner在G_W | 342_result_阶段3输入预检.json、342_result_M0独立核查.json、342_M0跨类证据_*.png |
| 242例精确M0穷举与旧val32非M0逐位对照 | 342_result_阶段3单元验收.json |
| 全部1,096条独立窗/G_W/U_W/K_g/crop/M0证书 | 342_result_val548独立验收.json |
| 逐条文件、payload、图像、checkpoint、标注、M0代价SHA | 342_val548_逐条SHA清单.json |
| seed1/2各6张最终PNG | 342_val_v3_seed1_独立复核_*.png、342_val_v3_seed2_独立复核_*.png |
| 两seed各548条正式缓存 | 342_cache_val_v2_seed1/、342_cache_val_v2_seed2/ |
| 冻结模型、断点、调用次数与耗时 | 342_raw_seed*_result.json、342_full_seed*_result.json（总结果整合了中断段计数） |
| 精确M0及共享几何差异 | 342_M0_新旧算法.diff、342_geom342_阶段3扩展.diff、342_geom342_原始query输入域修复.diff |
| 大整数序列化失败与修复 | 342_result_缓存序列化实现失败.json、342_build_val_v2_序列化修复.diff |
| seed2未入fullP的原始query输入域误报及修复 | 342_result_全局退化框诊断.json、342_result_seed2原始query验证实现失败.json、342_result_原始query输入域修复验证.json |

正式缓存使用`gt_short_side_640`保留旧341的G_W对齐语义，另存`gt_short_side_640_all`与全GT annotation_id一一对应，避免后续tiny组索引混淆。精确M0代价矩阵、合法边、owner、原flatten索引及可验证的整数对偶证书均随条目保存。大整数以十进制字符串持久化，计算并未改成浮点近似。

选窗仍调用stage1_cache_v3的固定选择器，原始300-query输入全部保留；没有通过阈值或剔框修复窗口。两seed3图7个非正尺寸原始query均未进入fullP；正式候选仍要求有限、正面积。

注意：首版目录342_cache_val_seed1/2各32条为保留的序列化失败产物，不能用作活动输入。第一版验收生成的旧PNG和失败结果只属历史记录。以本索引列出的v2缓存、v3构建/验收脚本与v3 PNG为准。

主控需核对：两seed统计及Wilson区间、旧val32的7图U_W变化、两seed各6张PNG、精确M0合同。收到阶段3放行前不进入阶段4。

开训前还需修正development3235×2的旧M0标签与训练输入版本，并重新审查相关合同；此工作本阶段未执行。旧阶段2四臂冒烟/续跑已获认可，但不是新M0标签上的重跑结果。阶段4映射必须通过geom342的局部归一化cxcywh反变换；该反变换及评估本阶段未执行。
'''
    C.text_out('342_阶段3_交接索引.md',index)
    # Current audit enumerates active source lines; independent oracles are
    # explicitly exempt from calling shared geometry by the user's requirement.
    audit=[];pat=re.compile(r'\b(independent_cap|d_out|geometry|window_xyxy(?:_640)?|K_g)\b')
    for p in sorted(C.HERE.rglob('*.py')):
        src=p.read_text(encoding='utf-8-sig');ast.parse(src)
        active=p in [geom,C.HERE/'342_build_val_v3.py',C.HERE/'342_validate_val_v3.py',C.HERE/'342_m0_exact.py']
        kind='活动共享几何/调用入口' if active else ('独立对照，不调用共享实现' if p.name=='342_val_independent.py' else '历史版本或诊断/证据脚本')
        for line,text in enumerate(src.splitlines(),1):
            if pat.search(text):audit.append({'file':str(p),'line':line,'text':text.strip(),'kind':kind})
    C.json_out('342_result_geom_audit_阶段3.json',{'entries':audit,'source_sha256':C.sources(),
        '已执行':{'source_files':len(list(C.HERE.rglob('*.py'))),'matched_lines':len(audit)},
        '未执行':['阶段4反变换与评估'],'未核':['主控代码复核']})
    C.json_out('342_阶段3_来源与记录登记.json',{'source_sha256':C.sources(),
        'execution_records':[str(p) for p in C.HERE.glob('342_*result.json')],
        'diagnostic_records':[str(p) for p in C.HERE.glob('342_result*.json')],
        'note':'历史失败/更正记录保持不可变；执行来源和最终状态以阶段3总结果及本登记为准。',
        '已执行':{'provenance_registry':1},'未执行':['真实训练'],'未核':['主控复核']})
    files=list(C.HERE.rglob('*'))
    sealed={str(p):{'bytes':p.stat().st_size,'sha256':C.sha(p)} for p in files if p.is_file()}
    C.json_out('342_阶段3_交付封存SHA256.json',{'artifacts':sealed,'source_sha256':C.sources(),
        '已执行':{'artifact_sha_count':len(sealed),'optimizer_steps':0},'未执行':['阶段4至5','真实训练'],
        '未核':['主控独立复核'],'note':'包含原始/正式/失败缓存及全部证据，不含本清单自身；未覆盖任何已有产物。'})
    print(json.dumps({'status':result['status'],'storage':storage,'protected_files':len(protected),'sealed_files':len(sealed)},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
