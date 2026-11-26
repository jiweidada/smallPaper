"""Publish the completed 345 handoff exclusively; no model execution here."""
import importlib, json, re, time
from pathlib import Path
C=importlib.import_module('345_context')

def link(path, label=None):
    p=Path(path)
    return f'[{label or p.name}]({p.as_posix()})'

def readonly_audit():
    started=time.perf_counter(); checked={}; rows=[]
    for seal_path, field in ((C.HERE/'345_停止交付封存SHA256_v2.json','artifacts'),
                             (C.S5/'342_阶段5_交付封存SHA256.json','artifacts'),
                             (C.S5/'342_审查包证据文件SHA256_v2.json','files')):
        entries=C.read(seal_path)[field]
        for name, item in entries.items():
            expected=item if isinstance(item,str) else item['sha256']
            actual=C.sha(name)
            assert actual==expected, ('sealed old file changed',name)
            checked[name]=actual
        rows.append({'manifest':str(seal_path),'manifest_sha256':C.sha(seal_path),'verified_records':len(entries)})
    active=C.read(C.S5/'342_活动缓存7566_逐条SHA总清单.json')
    cache_bytes=0
    for r in active['entries']:
        p=Path(r['path'])
        assert p.stat().st_size==r['bytes'] and C.sha(p)==r['file_sha256'], str(p)
        cache_bytes+=r['bytes']
    assert len(active['entries'])==7566
    for item in active['source_manifests']:
        assert C.sha(item['path'])==item['sha256']
    bundle=C.read(C.HERE/'345_B_冻结入口绑定.json')
    for path, expected in bundle['sources'].items():
        assert C.sha(path)==expected, path
        checked[path]=expected
    checkpoint_paths={}
    for cfg in C.read(C.S5/'342_八run配置清单.json')['runs']:
        checkpoint_paths[cfg['global_checkpoint']]=cfg['global_checkpoint_sha256']
    for path, expected in checkpoint_paths.items():
        assert C.sha(path)==expected, path
        checked[path]=expected
    result={'status':'已验证','old_seals':rows,'unique_noncache_paths_checked':len(checked),
            'checked_noncache_sha256':checked,'active_cache_files_checked':7566,'active_cache_bytes_checked':cache_bytes,
            'raw_checkpoints_checked':len(checkpoint_paths),'wall_seconds':time.perf_counter()-started,
            '已执行':{'content_SHA_comparisons_including_repeated':sum(r['verified_records'] for r in rows)+len(bundle['sources'])+2+7566+2,
                       'cache_file_SHA_checks':7566,'head_forwards':0,'global_forwards':0,'optimizer_steps':0},
            '未执行':['重建缓存/改标签','真实训练','heldout图像或缓存读取'],'未核':['远端副本身份'],
            'source_sha256':C.sources()}
    C.write('345_result_只读旧文件与缓存封存核验.json',result)
    print('Read-only audit PASS: all 7566 active caches and prior sealed artifacts unchanged',flush=True)
    return result

def contract_rows(a):
    methods={1:'当前/冻结旧头加载同权重；对照计划字段及两层hidden；破坏值作反例',
             2:'独立NumPy逐类逐格cap/退回null；另枚举行列几何，检查守恒及质量拆分',
             3:'解析Bernoulli梯度对照；故意恢复旧m/u代数作反例',
             4:'J=0/全padding/空P有限性；SciPy匹配数及错误denom=4反例',
             5:'只改有效槽位；四字段同步排列/增删；只翻框必须被拒绝',
             6:'真实三损失分别autograd；直接mass正对照',
             7:'非零读出排列测试；NumPy类别计数；空行hook及破坏计数反例',
             8:'G/R同公共初态，逐字节检查两层hidden/输出/residual；破坏hidden反例',
             9:'改aux参数；生成预测逐字节比较；三个损失分别查class_null梯度',
             10:'将R residual替换成mass后与N逐字节比较；正常R hidden差异作正对照',
             11:'独立行列枚举；cxcywh/像素错误输入；两seed共4PNG及已完成目视',
             12:'一次真实raw global前向，克隆794项state逐字节比较；破坏state反例'}
    data=a['contracts'];plan_max=max(r['NumPy_max_abs_diff'] for r in data[1]['details']['rows'])
    geom_max=max(r.get('independent_geometry_max_abs_diff',0) for r in data[1]['details']['rows'])
    small_max=max(r['analytic_grad_abs_diff'] for r in data[2]['details']['rows'])
    permutation_max=max(x for r in data[6]['details']['rows'] for x in r['permutation_max_differences'].values())
    det_max=max(r.get('loss_det_abs_diff',0) for r in data[3]['details']['rows'])
    metrics={1:'全部比较字段差0',2:f'plan最大差{plan_max:.12g}；geometry最大差{geom_max:.12g}',
             3:f'解析梯度最大差{small_max:.12g}；前后向均有限',4:f'overflow检测损失差{det_max:g}；6目标匹配4、分母6',
             5:'输出差0；9次无效输入拒绝',6:'det/assoc对mass梯度0或None；mass损失梯度非零',
             7:f'排列最大差{permutation_max:.12g}（预定容差2e-5）；count精确',
             8:'两层hidden/输出/residual全部差0',9:'生成/预测差0；κ只接det',10:'替换mass后差0；正常R与N的hidden有非零差',
             11:'260次独立几何比较均≤1e-5；520反例通过；本项未单列实际最大差（未核）',
             12:'794项state差0；参数grad均None'}
    result=[]
    for r in data:
        n=r['contract'];assert r['status']=='已验证'
        path=C.HERE/(f'345_result_A{n:02d}.json' if n<=4 else f'345_result_A{n:02d}_v2.json')
        result.append({'contract':n,'status':'已验证','independent_or_negative_method':methods[n],
                       'numeric_result':metrics[n],'call_counts':r['call_counts'],
                       'evidence':{'path':str(path),'sha256':C.sha(path)},'source_sha256':r['source_sha256'],
                       'scope_note':'A1-4沿已通过原记录；A5-12沿345b重跑记录；本轮未再次运行A'})
    C.write('345_A逐项状态表_v3.json',{'status':'12项已验证，主控已放行','rows':result,
            'master_independent_scope':'两seed各前16窗A8独立重跑；A5/A7测试diff和十二项状态表已核。',
            '未核':['A11当次实际最大几何差未单列；通过原有≤1e-5断言，不补造数值'],
            'source_sha256':C.sources()})
    return result

def b_rows(b):
    allruns=[]
    for p in (C.HERE/'345_B_CPU_acceptance_runs').glob('345_*/345_run_result.json'):
        allruns.append(C.read(p))
    assert len(allruns)==32 and sum(r['counts_this_process']['optimizer_steps'] for r in allruns)==112
    initial_count=sum(r['initial_state_sha256']==C.read(C.S5/f"342_预定配置_{r['arm']}_seed{r['seed']}.json")['config']['untrained_initial_state_sha256'] for r in allruns)
    assert initial_count==32
    ck_count=len(list((C.HERE/'345_B_CPU_acceptance_runs').rglob('345_checkpoint*_receipt.json')))
    assert ck_count==64
    desc=[
        ('B1 数学与数据继承','直接调用train344函数；旧文件SHA另核，配方/损失/优化器/坐标函数不复制改写','0项旧源码改动','345_train344_to_train345.diff'),
        ('B2 配置与CLI绑定','8配置文件SHA取旧封存，内容与八run清单比较；CLI仅5参数；越权lr参数被拒绝','8配置绑定；1个CLI反例拒绝','345_B_冻结入口绑定.json'),
        ('B3 共享物化shuffle','每seed6轮完整排列；独立重新生成并核3235 ID、203批和末批3图','12个完整排列逐值相同','345_result_B_物化顺序独立验收.json'),
        ('B4 初态SHA','32个短验收进程均与各自预定初态state SHA比较','32/32相同','345_result_B_入口与CPU验收总表.json'),
        ('B5 完整断点与游标','独占xb保存model/optimizer/组lr/RNG/step/epoch/epoch内位置/shuffle；新进程从step3恢复','64份断点；8组从epoch1/batch1恢复','345_result_B_入口与CPU验收总表.json'),
        ('B6 SHA与预算守卫','配置/清单/shuffle篡改、既有run目录、1219步；附加正式拒绝mini与CLI覆写','7个反向用例全部拒绝；正向计算0次','345_result_B_五种篡改与附加守卫.json'),
        ('B7 数值日志与耗时','每步loss三项、各组lr、G十κ及clip后梯度；实际epoch耗时单独账本','112实际CPU优化步；耗时未作逐位声称','345_result_B_入口与CPU验收总表.json'),
        ('B8 八组跨轮恢复','连续6 vs 3+新进程3；递归原始字节比较model/optimizer/RNG及完整step日志字节','8/8通过；模型/优化器最大差0','345_result_B_入口与CPU验收总表.json'),
        ('B9 正式顺序两步','经正式CLI读取真实3235顺序；比较ID和collate后RGB/P全部张量字节SHA','8组×2步；同seed四臂指纹相同','345_result_B_正式顺序两步八组.json')]
    rows=[{'item':item,'status':'已验证','independent_or_negative_method':method,'numeric_result':metric,
           'evidence':{'path':str(C.HERE/name),'sha256':C.sha(C.HERE/name)},'source_sha256':C.sources()}
          for item,method,metric,name in desc]
    C.write('345_B逐项状态表.json',{'status':'已验证，待主控复核','rows':rows,'counts':b['counts'],
            'checkpoint_receipts':ck_count,'initial_SHA_checks':initial_count,
            '未执行':['完整1218步真实run','GPU','真实训练'],'未核':['主控独立复核B'],
            'source_sha256':C.sources()})
    return rows

def update_package(arows,brows,flops):
    text=(C.S5/'342_开训前审查包.md').read_text(encoding='utf-8')
    text=text.replace('# 342 开训前审查包\n','# 342 开训前审查包 v2\n',1)
    lines=text.splitlines()
    assert lines[2].startswith('日期：2026-10-22。')
    lines[2]='日期：2026-10-22。本文件是345/345b/345c完成后的活动v2；旧342包及所有失败/停止记录只读保留。状态：A十二合同主控已核，FLOPs补足及B的CPU验收已完成，待主控核对本次交付；真实训练未授权。'
    text='\n'.join(lines)+'\n'
    def replace_once(old,new):
        nonlocal text
        assert text.count(old)==1,old
        text=text.replace(old,new,1)
    replace_once('后续342/343/344落实几何、M0、计数和#1466明示例外。',
                 '后续342/343/344落实几何、M0、计数和#1466明示例外；345/345b/345c完成当前适配器合同、FLOPs补足和正式入口CPU验收。')
    def replace_row(label,replacement):
        nonlocal text
        prefix='| '+label+' |'
        old=[line for line in text.splitlines() if line.startswith(prefix)]
        assert len(old)==1,label
        text=text.replace(old[0],replacement,1)
    a8=link(C.HERE/'345_result_A08_v2.json');a9=link(C.HERE/'345_result_A09_v2.json')
    at=link(C.HERE/'345_A逐项状态表_v3.json');bt=link(C.HERE/'345_B逐项状态表.json')
    br=link(C.HERE/'345_result_B_入口与CPU验收总表.json')
    replace_row('G/R初态计划和预测同值、G的aux隔离',f'| G/R初态计划和预测同值、G的aux隔离 | 已验证 | 当前适配器两层hidden、输出、residual逐位相同；aux改变不动G生成；仅det给κ梯度。{a8}；{a9} |')
    replace_row('311守恒、empty/overflow、S排列/count、N无P与sg(m)',f'| 311守恒、empty/overflow、S排列/count、N无P与sg(m) | 已验证 | 345当前train344适配器已逐项执行；NumPy独立计划、解析极小u、合法扰动、三损失梯度分别核验；非适用臂有说明。{at} |')
    replace_row('G/R decoder hidden及所有继承合同在当前适配器上的完整重跑',f'| G/R decoder hidden及所有继承合同在当前适配器上的完整重跑 | 已验证 | A1–A12全部通过；A5/A7仅修测试构造，模型不改；主控已核状态表，并独立重跑两seed前16窗A8。{at}；{a8} |')
    replace_row('正式8run、正式科学判定和训练资源',f'| 正式8run、正式科学判定和训练资源 | 未验证 | 真实run/训练后效果/远端资源未执行；正式入口、物化shuffle和有界CPU恢复已验证，见下节B。{br} |')
    table='\n### 1.1 345当前适配器十二项合同\n\n| 项 | 状态 | 独立/反向方法 | 数值结果 | 实际调用 | 证据 |\n|---|---|---|---|---|---|\n'
    for r in arows:
        c=r['call_counts']
        counts=f"head完成{c['head_completions']}，loss{c['loss_calls']}，autograd{c['autograd_calls']}，反例{c['negative_controls']}"
        if c['NumPy_plan_calls']:counts+=f"，NumPy plan{c['NumPy_plan_calls']}"
        if c['global_forwards']:counts+=f"，global{c['global_forwards']}"
        table+=f"| A{r['contract']} | 已验证 | {r['independent_or_negative_method']} | {r['numeric_result']} | {counts} | {link(r['evidence']['path'])} |\n"
    table+='\n每项源文件SHA及全部调用计数保存在相应JSON和A状态表。A11未保存当次几何最大差标为未核，原260次≤1e-5断言与520反例均通过，不以阈值冒充实测最大值。4张PNG已目视，记录见 '+link(C.HERE/'345_result_A11四PNG目视.json')+'。\n'
    table+='\n### 1.2 B正式入口与CPU验收\n\n| 项 | 状态 | 独立/反向方法 | 实际结果 | 证据 |\n|---|---|---|---|---|\n'
    for r in brows:
        table+=f"| {r['item']} | 已验证 | {r['independent_or_negative_method']} | {r['numeric_result']} | {link(r['evidence']['path'])} |\n"
    table+='\nB实际执行32个正向工作进程、112次head前向/反向/优化步、1792次缓存加载、3584次native loss调用；global前向0、真实训练0。连续/恢复的确定性step日志逐字节一致；实际耗时是单独账本，不声称逐位相同。'+bt+'。\n'
    marker='主控核对范围：'
    assert text.count(marker)==1
    text=text.replace(marker,table+'\n'+marker,1)
    replace_once('上述是用户报告，不冒充本轮代跑。','上述是用户报告，不冒充本轮代跑。345的十二合同主控已放行，其中A8由主控在两seed前16窗独立重跑；FLOPs本次重跑与B为执行方实测，尚待主控独立复核。')
    start=text.index('## 3. 参数量');end=text.index('## 4. 八个run配置清单')
    section=text[start:end]
    section=section.replace('参数量与所列CPU计量已验证；完整逐算子FLOPs未验证。','参数量、CPU时延与本次观察算子的FLOPs覆盖已验证。时延沿旧342实测；FLOPs由345c另跑8次同输入前向补足。')
    section=section.replace('seed1已计FLOPs','seed1注册＋手算FLOPs').replace('seed2已计FLOPs','seed2注册＋手算FLOPs')
    replacement={str(r['registered_FLOPs']):str(r['registered_plus_manual_FLOPs']) for r in flops['rows']}
    for old,new in replacement.items():section=section.replace(old,new)
    old='FLOPs口径：PyTorch FlopCounterMode所注册的卷积/矩阵乘等算子，乘加按2 FLOPs；逐算子清单与未注册调用见证据。CPU融合attention、softmax、归一化和elementwise等可能未计入，数字不冒称完整FLOPs或完整部署成本。'
    assert section.count(old)==1
    section=section.replace(old,'FLOPs口径：已注册卷积/矩阵乘（乘加=2）＋逐算子形状手算补足；softmax为4N−rows，归一化为5N＋2rows并计affine，CPU融合attention/激活/elementwise均按数学原语计。所有观察到的dispatcher算子已分类。exp/log/erf等各按一个原语；比较/索引/内存/整数操作计0。这是明确口径下的算法FLOPs，不是硬件指令数或完整部署成本。')
    section+='\n345c完整逐事件/逐算子报告：'+link(C.HERE/'345_result_FLOPs覆盖补足_v2.json')+'。8个合计与失败前stdout完全相同；只把非有限标量元数据编码成字符串inf/-inf/nan，模型输出仍有限。计数AST不变证明与diff：'+link(C.HERE/'345_FLOPs序列化修订范围.json')+'、'+link(C.HERE/'345_FLOPs序列化修订.diff')+'。原部分JSON及实现失败收据保留，不作通过证据。\n\n'
    text=text[:start]+section+text[end:]
    replace_once('正式shuffle序列及完整run恢复尚未执行。','六轮正式shuffle已按seed物化并共享，跨轮中途恢复已完成8组CPU验收；完整1218步run仍未执行。')
    old='正式run入口未执行：train344保留CPU、<=20步、<=32窗断言且__main__直接退出。当前不能把该文件视为1218步启动程序；真实训练获单独授权后需另存正式入口、绑定数据/初态/每轮shuffle/完整恢复并验收，不能通过删除守卫直接启动。'
    new='正式控制流已另存 '+link(C.HERE/'train345.py')+'；train344守卫及源码完全保留。所有非CLI超参数从旧342预定配置读取，配置文件SHA取旧交付封存，内容与八run清单完全一致；旧配置里的入口/shuffle说明是其封存时状态，本次附加执行元数据以 '+link(C.HERE/'345_B_冻结入口绑定.json')+' 为准，配方数值不改。入口总预算硬上限1218；当前授权守卫只允许CPU，正式顺序≤2步、迷你验收≤6步。公开CLI仅arm/seed/run_dir/device/max_steps；验收的mode/新进程恢复由私有worker调用，不增加CLI超参数。恢复写入新的子run目录，父run只读，既有目录拒绝覆盖。六轮顺序为 '+link(C.HERE/'345_shuffle_development_seed1.json')+'、'+link(C.HERE/'345_shuffle_development_seed2.json')+'；mini文件仅验收，正式读取拒绝。真实训练需另行授权，本包不提供启动命令。'
    replace_once(old,new)
    replace_once('当前adapter完整hidden/继承合同重跑、完整FLOPs覆盖、累计缓存墙钟、远端训练环境身份均未核，不能用有限loss或参数量代替。',
                 '当前adapter合同与本次观察算子FLOPs补足已验证；A11当次几何最大差未单列（原断言通过）。累计缓存墙钟、远端训练环境身份未核。CPU短验收不替代训练后的指标、G有效性或远端数值验证。')
    replace_row('是否补核当前adapter的hidden/全部继承合成合同与FLOPs覆盖',f'| 当前adapter十二合同与FLOPs补足 | 已验证 | A由主控已放行；FLOPs八值与此前stdout一致，完整逐算子报告已落盘。{at}；{link(C.HERE/"345_result_FLOPs覆盖补足_v2.json")} |')
    replace_row('正式训练入口、shuffle物化与1218步完整恢复验收',f'| 正式入口、shuffle物化与跨轮恢复CPU验收 | 已验证 | 八组6 vs 3+新进程3严格逐位相同，八组真实顺序2步相同，五种篡改均拒绝；完整1218步未执行。{bt}；{br} |')
    replace_once('本包完成“开训前审查材料”交付；是否补核、是否批准正式入口及真实训练，由主控与用户裁定。当前没有启动任何真实训练，没有读取heldout文件。',
                 '本包完成345授权的A、FLOPs补足和B交付，现在停下等主控核对。是否批准真实训练由用户另行明确授权。当前真实训练0步；没有读取heldout图像/缓存，没有修改冻结模块或配方，没有提供训练启动命令。')
    text+='\n本次交付与SHA：'+link(C.HERE/'345_交接索引_v3.md')+'；'+link(C.HERE/'345_一页中文汇总_v3.md')+'；'+link(C.HERE/'345_阶段AB_result_v3.json')+'；'+link(C.HERE/'345_交付封存SHA256_v3.json')+'。旧包和旧停止索引保留为历史记录，以本v2和索引v3为本次交付。\n'
    C.text('342_开训前审查包_v2.md',text)

def main():
    a=C.read(C.HERE/'345_result_A合同总表_v2.json');f=C.read(C.HERE/'345_result_FLOPs覆盖补足_v2.json')
    b=C.read(C.HERE/'345_result_B_入口与CPU验收总表.json')
    assert len(a['contracts'])==12 and all(r['status']=='已验证' for r in a['contracts'])
    assert f['all_8_match_previous_stdout'] and b['counts']['optimizer_steps']==112
    assert not (C.HERE/'345_result_B实现失败_停止.json').exists()
    audit=readonly_audit()
    ar=contract_rows(a);br=b_rows(b)
    update_package(ar,br,f)
    C.text('345_一页中文汇总_v3.md',f'''# 345 一页中文汇总：A、FLOPs与B完成，待主控

主线仍是“已有检测的可拒绝解释与漏检补全”：冻结raw全局模型，比较N/S/G/R四臂×两seed。真实训练0步。

A十二合同沿已核原始/345b记录全部通过，主控已放行；本次不重复运行。A8两层hidden/输出/residual逐位同值。A5/A7仅修测试构造，错误的只翻框输入继续被拒绝。每项方法、差值、调用数与源码SHA见 {link(C.HERE/'345_A逐项状态表_v3.json')}。

FLOPs只修inf/-inf/nan元数据序列化，计数公式不变。8次重跑合计全部匹配原stdout：seed1 N/R 216758270、S 221100344、G 217204470；seed2 N/R 215882542、S 219681128、G 216266174。按声明的数学原语补足观察算子，不是硬件指令或完整部署成本。

B新入口train345直接继承train344数学函数，旧文件未改。2份正式六轮shuffle和2份mini顺序均已物化、封存；8组“连续6步 vs 3步＋新进程恢复3步”的model/optimizer/RNG/step日志逐位一致；8组正式顺序前2步的同seed四臂batch ID/张量SHA相同。五种篡改及两个附加反例全部拒绝。32个短验收工作进程，112次CPU前向/反向/优化步，1792次缓存加载；完整1218步未执行。耗时另记，不声称逐位相同。

旧封存和7566活动缓存SHA再次核验无差异。审查包v2保留#1466唯一非正框例外、新overflow 474/468图与3047/3199 GT、两seed同窗1443，以及raw AP 0.139242/0.148237与EMA历史日志的区别。EMA余差最大约0.0006、跨平台评估二进制、累计缓存墙钟仍未核；A11当次最大几何差未单列，只保留原≤1e-5断言通过证据。

当前停下等主控核对v2。入口仍只放行CPU正式顺序≤2步、mini≤6步。训练后G Δ和12项严格全正判定未执行；真实训练需另行授权。未使用GPU/SSH/队列，未读取heldout图像或缓存。本包没有训练启动命令。

活动审查包：{link(C.HERE/'342_开训前审查包_v2.md')}。交接入口：{link(C.HERE/'345_交接索引_v3.md')}。
''')
    C.text('345_交接索引_v3.md',f'''# 345 交接索引 v3（本次活动版本）

状态：A十二合同主控已核；345c FLOPs序列化修复及八值核对通过；B全部CPU短验收通过。A、FLOPs与B已完成，停下等主控。真实训练0步，需另行授权。

- {link(C.HERE/'342_开训前审查包_v2.md')}：完整六节审查包，沿旧包另存。
- {link(C.HERE/'345_一页中文汇总_v3.md')}：一页中文摘要。
- {link(C.HERE/'345_A逐项状态表_v3.json')}：A逐项方法/数值/调用数/源码SHA；A1–4原通过记录、A5–12的345b记录。
- {link(C.HERE/'345_result_FLOPs覆盖补足_v2.json')}：八组完整逐算子报告；{link(C.HERE/'345_FLOPs序列化修订.diff')} 仅改元数据编码。
- {link(C.HERE/'345_B逐项状态表.json')}、{link(C.HERE/'345_result_B_入口与CPU验收总表.json')}：B九项状态与八组恢复结果。
- {link(C.HERE/'train345.py')}、{link(C.HERE/'345_train344_to_train345.diff')}：新增入口与差异；train344及冻结数学源码未修改。
- {link(C.HERE/'345_B_冻结入口绑定.json')}、{link(C.HERE/'345_bindings_anchor.py')}：配置/清单/原raw checkpoint/初态/曲线/顺序/源码SHA信任链。
- {link(C.HERE/'345_result_B_物化顺序独立验收.json')}、{link(C.HERE/'345_result_B_正式顺序两步八组.json')}、{link(C.HERE/'345_result_B_五种篡改与附加守卫.json')}：顺序、batch张量指纹与反向用例。
- {link(C.HERE/'345_B_CPU_acceptance_runs')}：32个短验收目录，初态、64完整断点及收据、每步日志和单独耗时账本。
- {link(C.HERE/'345_result_只读旧文件与缓存封存核验.json')}：旧封存文件/7566缓存SHA重新核验。
- {link(C.HERE/'345_阶段AB_result_v3.json')}、{link(C.HERE/'345_交付封存SHA256_v3.json')}：本次计数与最终封存。

两个seed的正式shuffle分别见345_shuffle_development_seed1/2.json；mini只在验收worker中使用，正式加载拒绝。公开入口仅五个参数，当前CPU上限正式2步/mini6步；1218步预算硬上限保留。恢复以新目录承接已封存父断点，拒绝覆盖既有run。

原345_交接索引.md、v2停止索引、失败脚本和部分JSON均不覆盖，只是历史证据。本索引和审查包v2为本次交付。实际耗时不纳入确定性日志逐位声明。A11当次最大差未单列，原≤1e-5断言已过。GPU/真实训练启动命令未提供。
''')
    C.write('345_阶段AB_result_v3.json',{'status':'A/FLOPs/B完成，待主控；真实训练未授权',
            'A':{'status':'12项已验证，主控已放行','rerun_this_turn':False,'active_table':str(C.HERE/'345_A逐项状态表_v3.json')},
            'FLOPs':{'status':'已验证','rerun_forwards_this_turn':8,'all_8_match_previous_stdout':True,'rows':[{k:r[k] for k in ('seed','arm','registered_FLOPs','manual_FLOPs','registered_plus_manual_FLOPs')} for r in f['rows']]},
            'B':{'status':'已验证，待主控独立复核','rows':br,'actual_counts':b['counts'],'work_processes':32,'resume_cases':8,'formal_two_step_cases':8,'negative_cases':7},
            '已执行_本轮':{'head_forwards':120,'global_forwards':0,'backward_calls':112,'optimizer_steps_CPU_acceptance':112,'cache_loads_B':1792,'native_loss_calls_B':3584,'cache_file_SHA_checks_final':7566,'real_training_steps':0},
            'history_scope':'A历史实际调用及此前FLOPs失败的8次前向分别见旧收据；本轮计数不将继承结果冒充重跑。',
            '未执行':['真实训练','完整1218步run','GPU/SSH/队列','heldout图像或缓存读取','训练后指标/CI/12方向/G Δ'],
            '未核':['主控独立核对FLOPs完整报告和B','A11当次最大几何差未单列，只有≤1e-5断言','EMA余差具体原因','跨平台二进制逐位一致','累计缓存总墙钟','远端执行环境'],
            'read_only_audit':str(C.HERE/'345_result_只读旧文件与缓存封存核验.json'),
            'source_sha256':C.sources(),
            'authorization_sha256':{str(C.SPEC/name):C.sha(C.SPEC/name) for name in ('345_授权_继承合同重跑与正式入口验收.txt','345b_补充裁定_A5测试构造修正.txt','345c_补充裁定_FLOPs写盘修正.txt')}})
    # No original artifact is edited. Seal every new/existing deliverable, including
    # worker checkpoints; the known old partial JSON is kept as opaque bytes.
    paths=[p for p in C.HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    artifacts={str(p):{'sha256':C.sha(p),'bytes':p.stat().st_size} for p in sorted(paths)}
    C.write('345_交付封存SHA256_v3.json',{'status':'345完成交付，待主控核对；真实训练0',
            'artifacts':artifacts,'files':len(artifacts),'bytes':sum(r['bytes'] for r in artifacts.values()),
            'active_review':str(C.HERE/'342_开训前审查包_v2.md'),'active_index':str(C.HERE/'345_交接索引_v3.md'),
            'source_sha256':C.sources(),'real_training_steps':0})
    print('Review v2, Chinese summary, handoff v3 and exclusive SHA seal written',flush=True)

if __name__=='__main__':main()
