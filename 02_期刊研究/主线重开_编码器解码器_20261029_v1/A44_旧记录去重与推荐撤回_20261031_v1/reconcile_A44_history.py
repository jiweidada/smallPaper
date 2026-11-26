"""Reconcile A44 with existing records; no model, network or training operation."""
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('C:/Users/jiwei/Desktop/小论文投稿')
BASE = ROOT/'英文小论文/主线重开_编码器解码器_20261029_v1'
D = BASE/'A43_ECDet三组蒸馏最小验证_20261031_v1'
N = D/'next_candidate_A44_v1'
OUT = Path(__file__).resolve().parent
REPORT = BASE/'106_A44重复路线核查与推荐撤回.txt'
RECEIPT = BASE/'107_A44撤回与历史记录校验.json'
STATUS = 'WITHDRAWN_AS_NEW_MAINLINE_AFTER_HISTORY_COLLISION_REVIEW'
CID = 'A44_QUERY_CONDITIONAL_SCALE_READ_ALLOCATION'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()


def write(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def locate(path, phrase):
    lines=path.read_text(encoding='utf-8').splitlines()
    hits=[i+1 for i,s in enumerate(lines) if phrase in s]
    assert hits,(path,phrase)
    return {'file':str(path),'line':hits[0],'anchor':phrase}


def main():
    if RECEIPT.exists():
        x=read(RECEIPT)
        assert x['status']=='PASS'
        for p,h in x['current_files_sha256'].items():
            assert sha(Path(p))==h,p
        print(json.dumps({'status':'ALREADY_CORRECTED_VERIFIED','receipt':str(RECEIPT)},ensure_ascii=False))
        return
    OUT.mkdir(exist_ok=True)
    before=OUT/'before_correction'
    before.mkdir(exist_ok=True)
    targets={
        BASE/'01_候选判别表.json':'01_候选判别表.json',
        BASE/'15_当前状态与结果校验.json':'15_当前状态与结果校验.json',
        N/'A44_candidate_proposal.json':'A44_candidate_proposal.json',
        N/'A44_候选问题与最小验证路线.txt':'A44_候选问题与最小验证路线.txt',
        ROOT/'agent.md':'agent.md',
        ROOT/'实验入口.md':'实验入口.md',
        D/'CONTINUE_A43.txt':'CONTINUE_A43.txt'
    }
    snapshot_manifest=OUT/'before_correction_manifest.json'
    if snapshot_manifest.exists():
        saved=read(snapshot_manifest)
        snapshots=saved['snapshots']
        for a in snapshots.values():
            assert sha(Path(a['preserved_as']))==a['sha256']
    else:
        snapshots={}
        for p,name in targets.items():
            q=before/name
            if q.exists():
                assert q.read_bytes()==p.read_bytes()
            else:
                q.write_bytes(p.read_bytes())
            snapshots[str(p)]={'preserved_as':str(q),'sha256':sha(q)}
        write(snapshot_manifest,{'snapshots':snapshots,'scope':'Original bytes preserved before correction.'})
    def original(p):
        return Path(snapshots[str(p)]['preserved_as'])
    old105=BASE/'105_A43终点与目标完成审计回执.json'
    old105sha=sha(old105)
    signed=read(old105)['files_sha256']
    resolutions={}
    for name,digest in signed.items():
        p=Path(name)
        resolved=original(p) if str(p) in snapshots else p
        assert sha(resolved)==digest, name
        resolutions[name]={'preserved_at':str(resolved),'sha256':digest,
                           'current_path_will_change':str(p) in snapshots}
    assert len(resolutions)==53

    ledger=ROOT/'英文小论文/研究历史与文献审计_20261026_v1/direction_table_data.json'
    legacy=read(ledger)
    assert len(legacy)==73
    byid={r['id']:r for r in legacy}
    selected=['D013','D020','D024','D025','D026','D027','D070']
    relevant={i:{k:byid[i].get(k) for k in ['id','name','status','stage','hypothesis','result','stop_reason','negative_scope','source','anchor','evidence_level']} for i in selected}
    h=ROOT/'英文小论文/文献_相关原始论文_20261004/H候选阶段裁决_反例与强裁剪基线_20261006.txt'
    hi=ROOT/'英文小论文/文献_相关原始论文_20261004/H接口阶段裁决_DSH97至98_20261006.txt'
    review=ROOT/'outputs/chat_improvement_review_20261011_v1/改进与测试方向复盘.txt'
    independent=ROOT/'英文小论文/data/00_实验计划与证据索引/二分量尺度候选_20261006_v1/Codex独立复核与CPU预诊断_v1/DEIM结构改点地图与二分量候选独立裁决_20261006_v1.txt'
    local=ROOT/'英文小论文/data/00_实验计划与证据索引/单一候选空间读取与近邻自审_20261008_v1/00_算子与归因自审结论.txt'
    spec=ROOT/'英文小论文/data/00_实验计划与证据索引/P2终点对主要严格定位缺口的归档验收_20261008_v1/13_局部几何读取的可检验规格_未执行.txt'
    oldsource=[
        locate(h,'当前优先候选是解码器跨尺度与尺度内点权重的归一化约束'),
        locate(h,'原生B本身已经动态'),
        locate(h,'主要区别是constant与input-dependent尺度门'),
        locate(h,'AdaMixer等分层/尺度门控先例'),
        locate(hi,'H张量尚未执行'),
        locate(review,'终止该P2细节保留机制的扩展'),
        locate(review,'预设“多数P3头共同近常数/同网格”条件命中0/85'),
        locate(review,'将采样点朝真实目标中心平移'),
        locate(independent,'V2：已有MSAD、ScaleKD'),
        locate(local,'增加融合前局部特征通路'),
        locate(spec,'局部多尺度读取/细特征回归在旧资料中已有近邻')
    ]
    comparisons=[
        {'recent':'A42定位专用/共享细尺度读出','old_ids':['D020','D024'],
         'overlap':'细尺度局部几何读取、任务侧几何支路及专用Encoder/Decoder已在旧路线中讨论和测试。',
         'old_decision':'D020两seed提前停止，D024固定12轮细节必要性对照未通过；均有限定范围。',
         'verdict':'方法族重复；更换到ECDet宿主不能自动建立新问题/新贡献。',
         'limit':'不声称A42与旧实现逐行等价；A42这次完整12轮结果仍是新取得且已核验的项目结果。'},
        {'recent':'A43实例query监督早期空间特征','old_ids':['V2_cross_scale_KD_review'],
         'overlap':'跨尺度/低分辨率检测蒸馏在旧独立复核中已作为V2讨论，已知MSAD/ScaleKD先例。',
         'old_decision':'旧V2为必要性未建立、降备选，不是全部蒸馏已实测失败。A43当前固定12轮另有阴性证据。',
         'verdict':'不能把通用蒸馏动机当新方向；本次实验结论保留。',
         'limit':'没有证据把旧V2说成A43同一target/投影/教师的精确实验。'},
        {'recent':'A44细网格占据条件化采样尺度归属','old_ids':['D025','D026','D027'],
         'overlap':'旧H已在同一3/6/3、12点MSDA框架讨论尺度质量、静态与input-dependent门和简单对照，已明确AdaMixer先例。采样集中/移点机制也有旧限定阴性。',
         'old_decision':'H没有完成性能验证；D026/D027否定的是特定集中/移点前提。不能外推所有采样变体。',
         'verdict':'撤回A44作为新主线/下一优先训练候选：新草案没有说明为何不再落入旧路线及为何旧不推进理由已被新证据解除。',
         'limit':'A44更改尺度所属，H更改权重归一化；未证明数学等价，也没有找到A44精确slot3变体已经训练的记录。'}
    ]
    rules={
        'purpose':'Prevent relabelling old routes as new mainlines without imposing exhaustive causal proof before a pilot.',
        'indexes':[str(ledger),str(BASE/'01_候选判别表.json')],
        'cheap_required_fields_before_recommendation':['旧路线/编号和原裁决','这次问题、作用位置/变量究竟有何实质不同','支持重新投入的新增依据及可区分的简单对照'],
        'insufficient_reopening_reasons':['只换DEIM/ECDet/YOLO宿主','只换层、阈值、点数或起名字','把同族未测试的小变体自动视作新主线','以一次小验证很便宜代替查重和改动理由'],
        'reopening_scope':'同方法族可在实质新问题/证据或用户明确重开下重新评估，但必须承认继承关系和旧裁决；不要求先证明所有已有方法无效。',
        'automatic_training':False,
        'current_priority_candidate':None
    }
    evidence={'status':'DOCUMENT_RECONCILIATION','client_date':'2026-10-31',
              'review_scope':'Indexed73 historical units and21 recent candidate records; selected7 related entries and specific report passages. Not all raw experiments or full PDFs re-read/recomputed.',
              'legacy_count':73,'related_entries':relevant,'primary_project_passages':oldsource,
              'comparisons':comparisons,'future_dedup_rules':rules,
              'central_point':'User correction supported at research-route/method-family level. A44 is withdrawn; exact-variant empirical failure is not fabricated.',
              'current_metrics_scope':'A42/A43 verified numbers unchanged; this audit does not rerun AP.',
              'new_model_forwards':0,'new_optimizer_updates':0,'new_training_jobs':0,
              'new_web_searches':0,'remote_access_this_task':False}
    write(OUT/'历史路线碰撞对照.json',evidence)
    event={'phase':'A44_HISTORY_RECONCILIATION','recorded_utc':datetime.now(timezone.utc).isoformat(),
           'client_date':'2026-10-31','status':STATUS,'candidate_id':CID,
           'reason':'Repeated method family with no substantiated reason to reopen old route; prior history was not reconciled before recommendation.',
           'correction_scope':'Withdraw recommendation, not falsify an unrun variant as trained failure.',
           'mainline_validated':False,'current_priority_candidate':None,
           'pilot_recommended':False,'training_allocation':0,
           'A43_metrics_and_negative_result_retained':True,
           'comparison':str(OUT/'历史路线碰撞对照.json'),'report':str(REPORT),'receipt':str(RECEIPT),
           'new_model_forwards':0,'new_optimizer_updates':0}
    board_path=BASE/'01_候选判别表.json'
    state_path=BASE/'15_当前状态与结果校验.json'
    board=read(original(board_path)); state=read(original(state_path))
    old_counter={k:state[k] for k in ['model_forwards','training_updates','formal_detector_training_updates','capacity_only_optimizer_updates','formal_training_epochs']}
    row=next(r for r in board['candidates'] if r['id']==CID)
    row.update(status=STATUS,pilot_recommended=False,increment_approved=False,training_allocation=0,
               evidence=str(REPORT),withdrawal=event,
               pilot_recommendation_scope='WITHDRAWN. Historical unexecuted plan only.',
               next='Do not launch or relabel; require substantive new problem/evidence relative to old records before reopening.')
    board.update(status='NO_NEW_MAINLINE_A44_WITHDRAWN_AFTER_HISTORY_RECONCILIATION',
                 current_priority_candidate=None,current_priority_question='尚未找到新的可推荐主线；A44因旧路线重复且无重新投入依据而撤回。',
                 next_exploratory_question='新候选须先完成旧编号、旧裁决、实质差异的简明对照。',
                 mainline_selected_for_pilot=False,selection_stage_complete=False,
                 mainline_validated=False,latest_report=str(REPORT),latest_update_utc=event['recorded_utc'],
                 goal_status='complete',
                 goal_status_scope='Local history-reconciliation task fulfilled; broader research mainline is NOT complete/validated.',
                 remaining_deliverable='整体研究选题未完成；当前无优先候选，无新增实验安排。',
                 A44_proposal=row,A44_recommendation_withdrawal=event,
                 new_empirical_evidence='A43已核验阴性保持；本轮仅旧记录查重与推荐更正，无新实验。',
                 next_decision_protocol={'status':'HISTORY_DEDUP_BEFORE_NEW_RECOMMENDATION','required_comparison':rules['cheap_required_fields_before_recommendation'],'training_allocation':0})
    write(board_path,board)
    state.update(recorded_utc=event['recorded_utc'],goal_status='complete',goal_status_scope=board['goal_status_scope'],
                 latest_report=str(REPORT),current_priority_question=board['current_priority_question'],
                 mainline_selected_for_pilot=False,selection_stage_complete=False,mainline_validated=False,
                 current_user_task_status='HISTORY_RECONCILED_A44_WITHDRAWN_NO_NEW_MAINLINE',
                 latest_authoritative_phase=event,current_goal_turn=event,current_user_task_phase=event,
                 A44_proposal=row,A44_recommendation_withdrawal=event)
    state.setdefault('authoritative_phase_history',[]).append(event)
    state.setdefault('verification',{})['A44_history_reconciliation']=str(RECEIPT)
    state['goal_completion_tool_receipt']=str(OUT/'goal_completion_tool_receipt.json')
    write(state_path,state)

    proposal_path=N/'A44_candidate_proposal.json'
    proposal=read(original(proposal_path))
    proposal['prior_status_before_correction']=proposal['status']
    proposal.update(status=STATUS,pilot_recommended=False,mainline_validated=False,novelty_cleared=False,
                    training_allocation_now=0,recommendation_withdrawal=event,
                    historical_collision_records=['D025/H','D026','D027','D020/D024 family history'])
    proposal['minimal_validation_plan']['execution_status']='WITHDRAWN_HISTORICAL_PLAN_NOT_CURRENT_RECOMMENDATION'
    proposal['minimal_validation_plan']['not_launched']=True
    write(proposal_path,proposal)
    text_path=N/'A44_候选问题与最小验证路线.txt'
    oldtext=original(text_path).read_text(encoding='utf-8')
    text_path.write_text('## Material Passport\n类型：旧研究记录核对后的推荐更正，用户端日期2026-10-31。\n当前状态：A44作为新主线/下一优先训练候选的推荐已撤回；未训练，不冒称已实测失败。\n原因：旧H/D025及相关采样、细尺度读取路线未先核对；本草案没有给出摆脱旧裁决的新问题或新证据。\n完整更正：'+str(REPORT)+'\n以下完整保留撤回前的历史草案，不代表当前推荐或实验安排。\n\n——撤回前原文——\n'+oldtext,encoding='utf-8')
    write(N/'A44_current_status_override.json',event)
    with (D/'CONTINUE_A43.txt').open('w',encoding='utf-8') as f:
        f.write(original(D/'CONTINUE_A43.txt').read_text(encoding='utf-8'))
        f.write('\nSUPERSEDING HISTORY CORRECTION: A44 recommendation WITHDRAWN after user pointed to prior records. See106report/107receipt. Current priority=null; no new training. Do not rerun old finalize_A43_report.py as current updater: it would restore a superseded recommendation. A43 numeric results/checkpoints unchanged; original105 signed versions are preserved in the new correction snapshot mapping.\n')

    intro='''【2026-10-31选题查重更正：A44推荐已撤回】
当前没有新的优先主线。A44“query条件化尺度读取分配”回到了已讨论的尺度门控/采样/细尺度读取方法族，缺少重新投入依据，已撤回；未启动新实验。
后续推荐先给出三项简明对照：旧编号与裁决；本次实质差异；支持重新投入的新依据和简单对照。只换宿主、层、阈值、点数或名字不算通过查重；不要求先证明所有已有方法无效。
旧记录入口：英文小论文/研究历史与文献审计_20261026_v1/direction_table_data.json；近期候选：英文小论文/主线重开_编码器解码器_20261029_v1/01_候选判别表.json。
更正与证据：英文小论文/主线重开_编码器解码器_20261029_v1/106_A44重复路线核查与推荐撤回.txt。
本段是选题约束；实验授权仍以用户当前指令为准。下面运行状态均按各自日期理解，不能当作当前活性或自动恢复依据。

'''
    (ROOT/'agent.md').write_text(intro+original(ROOT/'agent.md').read_text(encoding='utf-8'),encoding='utf-8')
    (ROOT/'实验入口.md').write_text(intro+original(ROOT/'实验入口.md').read_text(encoding='utf-8'),encoding='utf-8')
    report='''## Material Passport
类型：ARS项目旧记录核对与推荐更正；用户端日期2026-10-31。
本轮只读既有材料并更正状态：0新文献检索、0远端访问、0模型/张量调用、0训练或优化器更新；未重算旧AP。

结论
用户关于重复旧路线的核心指正成立。我在提出A44前，没有把项目完整旧台账的相关裁决作为排除条件。把同一方法族细化成“固定预算下搬移一个slot”，不足以作为新主线重新推荐。
A44推荐撤回；当前优先候选置空。并没有找到一条新的可推荐主线。A43已核验的实验结果和阴性裁决继续有效。
此处不编造“slot3这个精确变体已被训练否定”：没有查到这样的记录。撤回原因是旧路线重复、缺少解释旧不推进理由为何已改变的新依据。

查重范围
旧方向索引73个研究单元、近期候选表21条；重点核对7个相关旧条目及H、独立复核、局部读取和10月11日复盘中的对应段落。不是所有旧实验、原始预测或PDF重新审阅。
两份索引有交叉，不能相加称94个独立方向；旧台账也不是所有条目都等于完整训练失败。

具体对照
1. 细尺度局部几何读取与专用Encoder/Decoder
旧D020/fine_residual已有两seed提前停止的负结果，未完成原24轮；旧D024的双seed12轮相对BASE有小幅数值收益，但相对退化控制不能同时支持细节必要性，明确终止该机制扩展。
近期A42的“定位专用/共享细尺度读出”有同族继承关系。换为ECDet宿主不能单独算新的问题或贡献。A42本次固定12轮实验数据仍是真实的新增项目结果，不被追溯改成旧实验。
来源：outputs/chat_improvement_review_20261011_v1/改进与测试方向复盘.txt，第6、18–19、41–44行；旧台账D020/D024；10月8日局部几何读取规格与自审。

2. 尺度质量、尺度内权重和输入条件化尺度门
旧D025/H已经在原生[3,6,3]、每head12点MSDA中分析尺度质量，并规定固定尺度bias与input-dependent尺度门的简单对照。旧记录明确原生B本已动态，也列出AdaMixer等先例。
A44改变点的尺度所属，H改变权重归一化；没有证明两者数学等价。但二者落在已反复讨论的尺度利用/读取分配路线中。我没有给出新的独立问题或解除旧不推进理由的证据，就再次推荐，这一步不成立。
H实际停在接口/准备阶段，不能写成H完整训练失败。旧D026/D027分别是特定采样集中前提和朝GT移点前提的限定阴性，同样不能外推所有注意力/采样变体无效。
来源：文献_相关原始论文_20261004/H候选阶段裁决_反例与强裁剪基线_20261006.txt，阶段判断及第1、2、5、7节；H接口阶段裁决_DSH97至98_20261006.txt；复盘第92–98行。

3. 蒸馏与近期A43
旧二分量候选独立复核第6节V2已经讨论跨尺度/低分辨率检测蒸馏及先例，并裁决必要性未建立、降备选。它不是A43同一教师/目标/投影的原样实验，也不是“所有蒸馏都失败”。
A43这次完整12轮对照仍有效：Q的APs低于原版和区域蒸馏，按冻结门槛停止。旧V2的存在本应在选题继承关系中说明，而不是把通用蒸馏动机当成新的研究方向。
来源：data/00_实验计划与证据索引/二分量尺度候选_20261006_v1/Codex独立复核与CPU预诊断_v1/DEIM结构改点地图与二分量候选独立裁决_20261006_v1.txt；104/105及A43独立计分文件。

已执行的更正
- 01候选表、15当前状态：A44 withdrawn，pilot_recommended=false，training_allocation=0，current_priority_candidate=null。
- A44 JSON状态与文本开头加入撤回说明；原草案保留为历史未执行规格，不可当作当前推荐或任务队列。
- 实验入口与agent.md加入旧编号/原裁决/实质差异的廉价查重步骤，优先读完整旧台账，再看近期记录。此步骤不扩展为无限的因果预证明或新增审批。
- 未改A42/A43训练源码、数据、权重、预测、计分或指标；未启动新实验。当前未查询远端实时状态。
- 104/105原件保留；被更新的105签名输入保留原字节副本，107记录完整解析映射，旧审计不能充当A44当前推荐的有效性证明。

以后重新提出同族方案的最低要求
必须先列旧编号与原裁决、实质新问题/变量/位置、支持重新投入的新增依据和可区分简单对照。换宿主、阈值、点数、层或名字本身不足以重新晋级。若没有实质差异，就明确尚未找到新主线，不再将旧路线小变体作为新方向安排测试。

完整逐条对照、证据锚点、原状态快照及SHA解析见同目录A44_旧记录去重与推荐撤回_20261031_v1；本次校验回执为107_A44撤回与历史记录校验.json。
'''
    REPORT.write_text(report,encoding='utf-8')

    # Verify all current states and every signed old input after the correction.
    b=read(board_path);s=read(state_path);p=read(proposal_path)
    current=next(r for r in b['candidates'] if r['id']==CID)
    assert b['current_priority_candidate'] is None and not b['mainline_selected_for_pilot']
    assert current['status']==STATUS and current['pilot_recommended'] is False
    assert current['training_allocation']==p['training_allocation_now']==0
    assert p['status']==STATUS and p['pilot_recommended'] is False
    assert p['minimal_validation_plan']['not_launched']
    assert all(s[k]==value for k,value in old_counter.items())
    assert s['A43_terminal_verification']==read(original(state_path))['A43_terminal_verification']
    assert sha(old105)==old105sha
    for name,val in resolutions.items():
        assert sha(Path(val['preserved_at']))==val['sha256'],name
    for target,meta in snapshots.items():
        assert sha(Path(meta['preserved_as']))==meta['sha256']
    assert STATUS in text_path.read_text(encoding='utf-8') or '推荐已撤回' in text_path.read_text(encoding='utf-8')
    assert 'A44推荐已撤回' in (ROOT/'agent.md').read_text(encoding='utf-8')
    assert 'A44推荐已撤回' in (ROOT/'实验入口.md').read_text(encoding='utf-8')
    evidence_paths=[ledger,h,hi,review,independent,local,spec]
    result={'status':'PASS','phase':'A44_HISTORY_RECONCILIATION','client_date':'2026-10-31',
            'recorded_utc':datetime.now(timezone.utc).isoformat(),
            'recommendation_withdrawn':True,'current_priority_candidate':None,
            'mainline_validated':False,'exact_variant_training_failure_claimed':False,
            'A43_metrics_and_terminal_evidence_unchanged':True,
            'unchanged_model_counters':old_counter,
            'new_model_forwards':0,'new_optimizer_updates':0,'new_training_jobs':0,
            'new_web_searches':0,'remote_access':False,
            'signed_old_receipt':str(old105),'signed_old_receipt_sha256':old105sha,
            'signed_105_all53_inputs_resolved':resolutions,
            'snapshots_before_correction':snapshots,
            'source_evidence_sha256':{str(p):sha(p) for p in evidence_paths},
            'current_files_sha256':{str(p):sha(p) for p in list(targets)+[REPORT,OUT/'历史路线碰撞对照.json',snapshot_manifest,N/'A44_current_status_override.json',Path(__file__)]},
            'scope':'Documentation/recommendation correction; no new numerical reproduction, no universal rejection of all variants.',
            'goal_completion_receipt':str(OUT/'goal_completion_tool_receipt.json')}
    write(RECEIPT,result)
    print(json.dumps({'status':'PASS','A44_recommendation':'WITHDRAWN','priority':None,
                      'signed105_inputs_preserved':len(resolutions),'changed_files':len(targets),
                      'new_training':0,'report':str(REPORT)},ensure_ascii=False))


if __name__=='__main__':
    main()

