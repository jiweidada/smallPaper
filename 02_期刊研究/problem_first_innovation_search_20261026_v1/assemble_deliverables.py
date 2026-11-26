import json, hashlib, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
AUDIT=ROOT/'英文小论文/研究历史与文献审计_20261026_v1'
def read(name): return json.loads((OUT/name).read_text(encoding='utf-8-sig'))
def save(name,data): (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def flat(x):
    if x is None:return ''
    if isinstance(x,str):return x
    if isinstance(x,(int,float,bool)):return str(x)
    if isinstance(x,list):return '\n'.join(flat(v) for v in x)
    return '\n'.join(f'{k}: {flat(v)}' for k,v in x.items())
def pick(d,*names):
    for k in names:
        if d.get(k) is not None:return d[k]
    return ''

p1=read('local_P1_P3.json'); p45=read('local_P4_P5.json'); p810=read('local_P8_P10.json'); pr=read('local_P6_P7.json')
cards=read('problem_cards.json')
directions=json.loads((AUDIT/'direction_table_data.json').read_text(encoding='utf-8'))
root_sources={
 'cache_contract':('英文小论文/code/problem_definition_audit_20261026_v1/cache_contract_audit.json','$.seeds, $.decoder_configs；只读旧全量审计，不重新重放缓存'),
 'cap2':('英文小论文/code/postprocess_budget_precheck_20261026_v1/result.json','$.per_seed.{1,2}.arms及matching_ceiling_NOT_TP'),
 'nms':('英文小论文/code/postprocess_nms_control_20261026_v1/result.json','$.per_seed.{1,2}.{B0,B1}及delta_stats12'),
 'nms_protocol':('英文小论文/code/postprocess_nms_control_20261026_v1/protocol.txt','L4–11；全文'),
 'nms_code':('英文小论文/code/postprocess_nms_control_20261026_v1/run_control.py','L25–43、L64–105；静态读，未执行'),
 'semantic':('英文小论文/data/00_实验计划与证据索引/分类读出与分数预算核验_20261007_v2/04_本轮机制结果与目标裁决.txt','§1–4及限制；旧特权干预'),
 'native_code':('英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1/native/engine/deim/dfine_decoder.py','L315–359、L499、L884–905；静态读'),
 'v7_correction':('英文小论文/持续寻找研究主线_20261020_v2/360_v7_问题定义交付_独立核对_20261026.txt','§1–5；原数据契约/单位/匹配勘误'),
 'completion_raw':('英文小论文/code/unit_instance_completion_20261021_v1/349_cpu_val_20261023_v1/349_十二项判定_result.json','$.directions12, $.all12_strict_positive, $.footnotes'),
 'completion_metrics':('英文小论文/code/unit_instance_completion_20261021_v1/349_cpu_val_20261023_v1/349_val548指标与G校准.json','$.metrics, $.evaluator；本轮未执行349_evaluate.py'),
 'completion_report':('英文小论文/持续寻找研究主线_20261020_v2/351_主线val548结果_主控核对.txt','§1–5；旧建议仅作历史，不继承回相位/训练指令'),
 'scap':('英文小论文/data/00_实验计划与证据索引/双账号SCAP暂停与四任务快照_20261007_v1/四任务数据与同轮比较.md','L34–45；协议不一致与修复边界'),
 'ranking_code':('英文小论文/code/budget_ranking_20261007_v1/ranking_loss.py','L7–53；静态读，encoder dense_aux'),
 'swap':('英文小论文/持续寻找研究主线_20261020_v2/175_排序1024交换_完整结果回交.txt','§1–3，1024行与简单增删比较'),
 'old_history':('outputs/Claude独立复核交接_20261012_v1/01_全工作区研究总结_仅供参考.txt','L81–125；历史汇总，不冒充终点重算'),
 'full360':('英文小论文/持续寻找研究主线_20261020_v2/360_创新点整理_线程A_全量对照_20261026_v1.md','全文表格；L36差值方向错误另记，不使用其旧指令'),
 'prompt373':('英文小论文/持续寻找研究主线_20261020_v2/373_问题优先寻找创新点_提示词_20261026.txt','用户本轮明确指定的范围'),
 'summary372':('outputs/research_history_literature_audit_20261026_v1/372_研究方向与文献覆盖审计_20261026_v1_一页结论.txt','全文'),
 'directions372':('英文小论文/研究历史与文献审计_20261026_v1/direction_table_data.json','73行导航，原状态和否定范围保留'),
 'literature372':('英文小论文/研究历史与文献审计_20261026_v1/literature_table_data.json','312行身份/历史实际阅读范围，定向定位'),
 'workbook372':('outputs/research_history_literature_audit_20261026_v1/372_研究方向与文献覆盖审计_20261026_v1_两张明细表.xlsx','两张工作表只读ZIP/XML，未修改/重导出')
}

all_refs=[]
for x in p1['sources']:all_refs.append(dict(x,alias='P1:'+x['id'],locator=x.get('locator',''),scope=x.get('read_scope','')))
for x in p45['sources']:all_refs.append(dict(x,alias='P45:'+x['id'],locator=x.get('read_locator',''),scope=x.get('kind','')))
for k,x in p810['references'].items():all_refs.append(dict(x,alias='P810:'+k,locator=x.get('locator',''),scope=x.get('read_scope','')))
for k,(path,loc) in root_sources.items():all_refs.append({'alias':'root:'+k,'path':str(ROOT/path),'locator':loc,'scope':'当前根助手只读文件/字段；未运行被读程序'})
groups={}
for x in all_refs:
    path=str(Path(x['path'])); key=path.replace('\\','/').lower()
    g=groups.setdefault(key,{'path':path,'aliases':[],'locations':[],'reading_scope':[],'read_hashes':[]})
    g['aliases'].append(x['alias'])
    if x.get('locator'):g['locations'].append(x['alias']+' '+x['locator'])
    if x.get('scope'):g['reading_scope'].append(x['alias']+' '+x['scope'])
    if x.get('sha256_read_now'):g['read_hashes'].append(x['sha256_read_now'])
evidence=[]; alias_map={}
for ix,(_,g) in enumerate(sorted(groups.items()),1):
    p=Path(g['path']); assert p.is_file(),str(p)
    g['id']=f'E{ix:03d}';g['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    assert all(h==g['sha256'] for h in g['read_hashes']),('Changed source',str(p))
    for alias in g['aliases']:alias_map[alias]=g['id']
    evidence.append(g)
for c in cards:
    for ref in c['refs']: assert ref in alias_map,ref
    c['evidence_ids']=[alias_map[r] for r in c['refs']]

checks={}
for d in p1['trial_checks']:checks[d['id']]={'refs':['P1:'+x for x in d['sources']],'note':d['audit'],'grade':'见对应来源实际阅读范围；不等于原预测复现'}
for d in p45['directions']:checks[d['id']]={'refs':['P45:'+x for x in d['sources']],'note':flat({k:v for k,v in d.items() if k not in ['id','sources']}),'grade':'原件/静态源码与历史材料混合，按源标识实际范围'}
for d in p810['directions']:
    a=d['local_audit']; checks[d['id']]={'refs':['P810:'+x for x in a['refs']],'note':flat(a['facts'])+'\n'+a.get('remaining',''),'grade':a.get('evidence_grade','')}
root_drefs={'D010':['old_history'],'D041':['scap'],'D042':['semantic'],'D043':['ranking_code'],'D044':['swap'],'D054':['full360'],'D055':['completion_raw','completion_metrics','completion_report'],'D057':['full360'],'D071':['cap2'],'D072':['nms','nms_protocol','nms_code']}
for did,refs in root_drefs.items():checks[did]={'refs':['root:'+x for x in refs],'note':pr.get('other_directions',{}).get(did,flat([p.get('negative_scope','') for p in pr['problems'] if did in ['D055','D071','D072','D042']])),'grade':'根助手只读原结构化结果/协议/静态源码；D010/D054/D057明确保留历史汇总级别'}
assert set(checks)=={d['id'] for d in directions},('Coverage mismatch',set(checks)^{d['id'] for d in directions})

bundles=[('literature_Q01_ranking.json',['Q01']),('literature_Q02_geometry.json',['Q02']),('literature_Q02_input.json',['Q02']),('literature_Q03_augmentation.json',['Q03'])]
sources=[];logs=[]
for fn,qs in bundles:
    b=read(fn)
    for s in b['sources']:
        t=dict(s); t['origin_file']=fn; t.setdefault('problem_ids',qs); sources.append(t)
    for i,log in enumerate(b['search_log']):logs.append({'source_file':fn,'source_locator':f'$.search_log[{i}]','problem_ids':qs,**log})
    for i,log in enumerate(b.get('access_log',[])):logs.append({'source_file':fn,'source_locator':f'$.access_log[{i}]','problem_ids':qs,**log})
by_id={s['id']:s for s in sources}
snip_alias=by_id.pop('R62')
by_id['R41']['problem_ids']=['Q02','Q03']; by_id['R41']['alias_source_ids']=['R62'];by_id['R41']['additional_reading_R62']=snip_alias
sources=list(by_id.values()); sources.sort(key=lambda s:int(s['id'][1:]))
for c in cards:c['literature_ids']=list(dict.fromkeys('R41' if r=='R62' else r for r in c['literature_ids']))
assert all(r in by_id for c in cards for r in c['literature_ids'])

problem_header=['问题ID','优先级','不预设解法的问题定义','实际观测','模型、划分与口径','解释局限','竞争解释与判别证据','旧方案及否定范围','剩余缺口','状态与排序理由','原证据ID']
problem_rows=[[c['id'],c['priority'],c['definition'],c['observation'],c['model_scope'],c['explanation_limits'],c['competing'],','.join(c['old_ids'])+'\n'+c['negative_scope'],c['remaining_gap'],c['status']+'\n'+c['rank_reason'],', '.join(c['evidence_ids'])] for c in cards]
literature_header=['问题ID','来源ID','论文或代码身份及版本','实际阅读与原文定位','问题先例状态','已有办法及原文支持','适用边界','本地同协议状态','剩余差距及不确定性','原始来源']
literature_rows=[]
for s in sources:
    identity=f"{s['title']}\n{flat(s.get('authors',''))} {s.get('year','')}\n{flat(pick(s,'version','identity'))}"
    actual=flat(pick(s,'actual_read_scope','actual_reading_scope'))+'\n'+flat(pick(s,'locators','paper_locator'))
    if s['id']=='R41':actual+='\n兼R62：'+flat(pick(snip_alias,'actual_read_scope','actual_reading_scope'))
    prior=flat(pick(s,'same_problem_defined_in_literature','same_problem_definition','problem_definition_prior'))
    known=flat(s.get('known_solution',''))+'\n'+flat(s.get('supported_claims',''))
    boundary=flat(pick(s,'applicability_boundary','conditions','effective_conditions'))
    if s.get('method_weaknesses'):boundary+='\n'+flat(s['method_weaknesses'])
    local=flat(pick(s,'tested_same_local_protocol','local_same_protocol'))
    gap=flat(s.get('remaining_gap',''))
    url=flat(pick(s,'url','original_url','primary_url'))
    assert url.startswith('https://'),(s['id'],url)
    literature_rows.append([','.join(s['problem_ids']),s['id'],identity,actual,prior,known,boundary,local,gap,url])

evidence_header=['证据ID','实际读取的原文件','原文位置或JSON键','本轮实际阅读边界','分区索引别名','SHA256（交付组装时）']
evidence_rows=[[e['id'],e['path'],flat(e['locations']),flat(e['reading_scope']),', '.join(e['aliases']),e['sha256']] for e in evidence]
legacy_header=['审计ID','问题大类','已试方案','历史主状态','实际阶段','历史结果（原口径）','真正停止/保留理由','否定范围','本轮核对及限制','本轮证据ID','372原导航位置']
legacy_rows=[]
for d in directions:
    ch=checks[d['id']];rr=[alias_map[x] for x in ch['refs']]
    original=d['evidence'];loc=str(original['path'])+':'+str(original.get('line',''))
    legacy_rows.append([d['id'],d['family']+' '+d['family_name'],d['name'],d['status'],d['stage'],d['result'],d['stop_reason'],d['negative_scope'],ch['note']+'\n阅读程度：'+ch['grade'],', '.join(rr),loc])
search_header=['记录ID','问题ID','记录日期','检索式或访问目标','保留理由、结果或访问限制','完整原记录定位']
search_rows=[]
for ix,l in enumerate(logs,1):
    query=pick(l,'queries','query','url','target','request','search_query','action')
    if not query:query={k:v for k,v in l.items() if k not in ['source_file','source_locator','problem_ids']}
    result=pick(l,'result','outcome','results','status','note','notes','purpose','disposition','reason')
    search_rows.append([f'S{ix:03d}',','.join(l['problem_ids']),str(pick(l,'date','date_local','search_date') or '2026-10-26'),flat(query),flat(result),l['source_file']+' '+l['source_locator']])

tables=[
 {'name':'问题与归因','title':'问题与归因：三个深入问题，七个保留问题','headers':problem_header,'rows':problem_rows,'widths':[11,19,46,55,48,48,55,58,49,52,23]},
 {'name':'文献与已有办法','title':'问题、原文和已有办法：SNIP已合并身份','headers':literature_header,'rows':literature_rows,'widths':[13,11,57,64,44,62,64,57,60,68]},
 {'name':'原始证据索引','title':'实际读取的本地来源及阅读边界','headers':evidence_header,'rows':evidence_rows,'widths':[11,90,78,72,33,69]},
 {'name':'旧方案适用范围','title':'73条历史条目的状态与否定范围','headers':legacy_header,'rows':legacy_rows,'widths':[11,28,40,25,42,58,58,59,72,26,90]},
 {'name':'检索与访问记录','title':'定向检索与访问记录：含失败和部分阅读','headers':search_header,'rows':search_rows,'widths':[11,14,17,92,93,76]}
]
save('research_tables_data.json',{'tables':tables,'metadata':{'covered_families':10,'covered_direction_ids':len(checks),'problem_cards':len(cards),'deep_problems':3,'literature_records':len(sources),'paper_identities':len([s for s in sources if s['id']!='R64']),'search_and_access_records':len(logs),'method_status':'M1仅条件式设想，未满足新颖性和方法有效性证明','SNIP_identity_dedup':'R62并入R41，原记录保留'}})
save('evidence_registry.json',{'sources':evidence,'alias_to_evidence_id':alias_map})
save('literature_comparison.json',{'sources':sources,'search_and_access_logs':logs,'identity_aliases':{'R62':'R41'},'code_record_ids':['R64']})
save('problem_cards_resolved.json',cards)

text=['问题—文献—已有解法逐项说明','源自本轮实际阅读范围；SNIP R62并入R41；作者代码R64不计论文。','']
for s,row in zip(sources,literature_rows):
    text += [f"{s['id']} / {row[0]}：{s['title']}",'身份与版本：'+row[2],'原始链接：'+row[9],'实际阅读与定位：'+row[3],'问题先例：'+row[4],'已有办法与支持：'+row[5],'适用边界：'+row[6],'本地同协议状态：'+row[7],'剩余缺口：'+row[8],'']
text+=['检索与访问原记录（包含完整检索式和失败）','']
for i,l in enumerate(logs,1):text += [f'S{i:03d} '+json.dumps(l,ensure_ascii=False),'']
(OUT/'03_文献与已有办法_逐项说明.txt').write_text('\n'.join(text),encoding='utf-8')

corrections=[]
corrections.extend(p1['audit_additions']);corrections.extend(p45['corrections_and_cautions']);corrections.extend(p810['corrections']);corrections.extend([{'topic':'根助手补充','finding':x} for x in pr['corrections']])
save('corrections.json',corrections)
(OUT/'04_证据补充与口径更正.txt').write_text('证据补充与口径更正；不覆盖任何旧文件\n\n'+'\n\n'.join(f'{i+1}. '+flat(x) for i,x in enumerate(corrections)),encoding='utf-8')

summary='''373执行结论：尚未解释的问题与方法状态

本轮按十类问题横向核对73条研究条目，形成10张问题卡，深入其中3项。它们都有真实现象，但尚不能宣布新方法成立。

优先1：已有候选的选择损失与类别歧义。
BASE24 raw两seed的tiny各有145个GT存在真类几何候选却在top300全丢。硬cap2损失tiny TP39/36；传统类内NMS虽提升总体AP约.92/.95pp，tiny TP仅净+4/−2且FP增加。问题可测，类别歧义、跨query重复、重选和真实输出数的贡献尚未分开。先比较Soft-NMS及已有独立分类/排序方案；已有先例并不代表本地问题已解决。

优先2：极小框几何失败的形成环节。
同组4952个tiny中3715/3727在原始query就没有IoU≥.5的框，排序无法补救。输入与训练尺度、预算/初始化、FDR支持/监督归属、真实标签精度仍是竞争解释。NWD、Sized L1、TOLF、SAHI等未找到当前DEIM同协议验收。整数标签不证明均匀±.5px噪声；旧集合期望失败只限制该实现。当前是比较/测量缺口，不先设计新集合损失或几何头。

优先3：增强保护的真实收益来自哪些曝光变化。
12轮目标保护APs两seed均正且胜随机，但旧5096个sub4组的TP@.5/几何支持并非均正。APs与sub4定义不同，这不是结果矛盾。随机保护未严格匹配密度/标签存活；成熟单seed取消ZoomOut的APs反而下降。SNIP、Scale Match、原生DEIM和DST已有尺度/反馈办法，标准比较与长期机制仍缺证。

仅保留一个条件式方法设想M1：若部署可得语义能区分真类歧义、背景与跨query重复，才考虑在固定框和query数下分别建模类别再判决和重复关系，再联合选择。它与cap2、编码器pairwise排名及已被复制的交换标签不同；但尚未证明超出DCR+NMS、Rank-DETR/MDS等既有组合，不能称新创新点。如果简单办法已解释残差、部署语义不可分、联合不胜同输入独立方案，或损害抵消新增，即推翻。它只能针对选择残差，不能解决约3700个几何不足tiny的大多数。

关键更正已另存：截断319次不等319个漏检；D035是4/5轮早停；D023正式门零修改；D030为BASE12、D037为epoch053 EMA局部诊断；360的一处C与均值控制差值方向倒置。

正式表格含：问题与归因、文献与已有办法、实际原证据索引、73条旧方案适用范围、定向检索/访问记录。具体数字和原文位置见各表及02/03详细说明。部分文献仅摘要或原PDF片段，已明确标记；未把下载或解析当全文已读。

本轮只读证据与文献、写新交付。未跑实验、模型前向、预检、重评、profiling或训练，未访问远端/旧队列，未新增test分析，未自动转回相位机制。AI辅助与共享上下文分工已披露。交付后停止。
'''
(OUT/'00_简洁结论.txt').write_text(summary,encoding='utf-8')

verification={'scope':'元数据、来源、覆盖与现有聚合数字的只读核对；不运行评价/模型','directions_expected':len(directions),'directions_checked':len(checks),'all_73_covered':set(checks)=={d['id'] for d in directions},'families':sorted({d['family'] for d in directions}),'problem_refs_resolved':True,'source_paths_exist':True,'source_read_hashes_match':True,'sources_count':len(evidence),'literature_records':len(sources),'paper_identities':len(sources)-1,'search_records':len(logs),'new_model_forwards':0,'new_evaluations':0,'new_training_steps':0,'new_test_analysis':False,'workbook':'pending'}
save('content_verification.json',verification)
print(json.dumps(verification,ensure_ascii=False))
