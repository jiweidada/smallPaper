"""Publish the first completed real epoch; keep prior receipts and frozen runs."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import shutil

P=Path(__file__).resolve().parent
BASE=P.parent
D=BASE/'A54_场景尺度约束候选_20261101_v1'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def meta(path):return dict(bytes=path.stat().st_size,sha256=sha(path))

def save(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def replace(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    receipt_path=P/'A60_B_epoch1_milestone_preservation_receipt_v1.json';assert not receipt_path.exists()
    parent=P/'A60_closest_prior_live_preservation_receipt_v1.json';old=json.loads(parent.read_text())
    for name,m in old['files'].items():assert meta(Path(name))==m,name
    proof=json.loads((P/'B_epoch1_bundle_local_verification_v1.json').read_text())
    assert proof['status']=='PASS_EXISTING_B_EPOCH1_BUNDLE_7_PAYLOADS_AND_13751_EVENTS'
    assert proof['original_epochs_completed']==1 and not proof['AP_available']
    remote_audit=json.loads((D/'A60_B_epoch1_readonly_audit_v1.json').read_text());assert remote_audit['exit']==0
    audit=json.loads(remote_audit['stdout']);assert audit['status']=='PASS_ORIGINAL_B_EPOCH1_REAL_JOURNAL_AUDIT'
    wrapper=json.loads((D/'A60_formal_observe_013.json').read_text());assert wrapper['exit']==0
    r=json.loads(wrapper['stdout']);b=r['arms']['B'];counts=b['journal_counts']
    assert r['source_binding'] and r['supervisor']['live'] and r['supervisor_exit'] is None
    assert r['supervisor']['pid']==30238 and r['supervisor']['actual_start_ticks']=='30772425'
    assert b['process']['live'] and b['process']['pid']==30239 and b['process']['actual_start_ticks']=='30772460'
    assert b['exit'] is None and b['completed_epochs']==1 and b['progress']['epoch']==2
    assert not r['arms']['M']['exists'] and not r['arms']['Q']['exists']
    assert counts['started']>=6471 and counts['updates']>=809
    scope=json.loads((P/'protocol_FROZEN_v1.json').read_text())
    for name,h in scope['source_sha256'].items():assert sha(P/name)==h,name
    assert sha(P/'protocol_FROZEN_v1.json')==r['protocol_sha256']
    mutable=[BASE/'01_候选判别表.json',BASE/'15_当前状态与结果校验.json',P/'CURRENT_A60.json']
    before=P/'before_B_epoch1_publish_v1';before.mkdir();resolution={}
    for f in mutable:
        dest=before/f.name;shutil.copyfile(f,dest);assert meta(dest)==old['files'][str(f)]
        resolution[str(f)]=dict(resolved_path=str(dest),**meta(dest))
    n=counts['started'];u=counts['updates']
    first=dict(status=proof['status'],epoch=1,images=6471,forwards=6471,backwards=6471,optimizer_updates=809,
        mean_loss=proof['mean_loss'],minimum_time_MLP_gradient_L1=proof['minimum_time_MLP_gradient_L1'],
        losses_gradients_finite=True,original_journal_events=13751,
        verification=str(P/'B_epoch1_bundle_local_verification_v1.json'),bundle_dir=str(P/'epoch1_B_remote_v1'),
        AP_available=False,full_formal_cross_arm_input_match_verified=False,mainline_validated=False)
    hp=P/'A60_turn_handoff_B_epoch1_v1.json'
    current=json.loads(mutable[2].read_text())
    current.update(status='A60_B_EPOCH1_AUDITED_ORIGINAL_B2_LIVE',recorded_utc=r['recorded_utc'],
        previous_goal_turn_classification='progress',current_goal_turn_classification='progress: completed first real epoch plus saved-journal verification',
        original_supervisor=r['supervisor'],arms=r['arms'],source_binding_verified=True,
        observed_formal_model_forwards=n,observed_formal_updates=u,completed_formal_epochs=1,
        global_observed_model_forwards=276643+n,global_observed_all_optimizer_updates=233606+u,
        global_observed_formal_optimizer_updates=233518+u,global_completed_formal_epochs=289,
        first_epoch_verification=first,latest_report=str(P/'06_B首轮训练核验与等待首评.txt'),
        latest_handoff=str(hp),next_readonly_receipt='A60_formal_observe_014.json',
        next_endpoint='B4 first AP, then original frozen12 B learning gate; M/Q still unstarted.',
        mainline_validated=False,candidate_has_pilot_support=False,training_protocol_modified=False,
        scope='Original B completed one full epoch and continues epoch2. Integrity/operational evidence only; no AP or candidate efficacy yet.')
    text=(
        'A60 B首轮真实训练核验\n'
        f"时间：服务器报告UTC {r['recorded_utc']}；绝对钟差不用于估算速度。\n"
        '数据/宿主：VisDrone6471训练/548验证，受控R50-FPN与六级DynamicHead，一次整段框解码；不是完整作者FlowDet复现。\n'
        '候选问题：同样的尺寸偏采样扰动框下，几何进度条件能否超出物理时刻条件？目前仍待验证。\n'
        '本次已证实：原B首轮6471次整模型训练前向、6471次反向、809次优化器更新；808组各8张，最后1组7张。13751条原始事件与整轮记录逐项相符。\n'
        f"首轮平均损失{proof['mean_loss']:.9f}；809次记录中的损失/梯度均有限，时间MLP梯度每次均正。损失值只说明数值/流程状态，不作为检测效果依据。\n"
        '归档：7个不可变有效负载及manifest，原始事件前缀/初始化/前8图输入摘要/整轮输入流摘要已本地核验。未重跑模型、未加载新检查点、未增加训练或COCO评估。\n'
        '输入证据边界：已记录完整B首轮输入流摘要；正式M/Q尚未开训，因此跨组实际完整输入相同还未核验。容量检查的输入一致性不冒充正式训练核验。\n'
        f"实时快照：原监督器30238/start30772425、B训练30239/start30772460均匹配且存活；已开始第2轮。日志累计前向开始{n}、完成{counts['completed']}、更新返回{u}，均为运行中下界。\n"
        'AP：尚无。第4/8/12轮验证，固定第12轮原始模型判定；B12 AP≥8且APs≥3才允许未启动M/Q继续。\n'
        '后续门槛不变：Q−B APs≥0.5pp、Q−M APs≥0.3pp、Q−B AP≥−0.2pp、Q/M中位模型延迟≤1.10；三组各≤12轮，无自动重试/参数补救。\n'
        '最近方法与风险：ConsistencyDet所读方法已覆盖非线性噪声日程/输入预条件化/少步框解码；CEOSOD全文仍未取得，不能排清重合。M/Q时间条件统计也不同，不能仅凭短测胜出认定唯一几何原因。详见05补查报告与02b矩阵增补。\n'
        '研究裁决：训练流程已完整跨过首轮；主线有效性、创新边界和投入价值尚未判定，总体目标active。A58等历史裁决、旧资产和423条文献主索引保留。\n')
    with (P/'06_B首轮训练核验与等待首评.txt').open('x',encoding='utf-8') as f:f.write(text)
    replace(mutable[2],current)
    board=json.loads(mutable[0].read_text());board['A60_execution']=current;board['A60_B_epoch1_verification']=first
    for row in board['candidates']:
        if row.get('id')=='A60_GEOMETRIC_PROGRESS_BOX_DECODER':
            row.update(status=current['status'],formal_observed_calls=n,formal_observed_updates=u,
                B_completed_epochs=1,B_epoch1_verification=first,evidence_scope=current['scope'])
    board.update(status=current['status'],latest_update_utc=r['recorded_utc'],latest_report=current['latest_report'],
        goal_status='active',mainline_validated=False,pilot_validation_status='B completed1, original B2 live; M/Q unstarted; no AP')
    replace(mutable[0],board)
    state=json.loads(mutable[1].read_text());state['A60_execution']=current;state['A60_B_epoch1_verification']=first
    state.update(status=current['status'],recorded_utc=r['recorded_utc'],latest_update_utc=r['recorded_utc'],
        previous_goal_turn_classification='progress',current_goal_turn='PROGRESS_FIRST_FULL_REAL_EPOCH_AND_ORIGINAL_B2_VERIFIED_LIVE',
        model_forwards=276643+n,training_updates=233606+u,detector_optimizer_updates_including_capacity=233606+u,
        formal_detector_training_updates=233518+u,formal_optimizer_updates_observed=233518+u,
        formal_training_epochs=289,cap=512185,counts_terminal=False,training_counts_terminal=False,
        full_forward_cap_closed=False,counters_observed_lower_bounds_while_job_live=True,
        goal_status='active',mainline_validated=False,latest_report=current['latest_report'],
        pilot_validation_status='Original B2 live after audited first epoch, no AP; M/Q unstarted')
    assert state['training_updates']-state['formal_detector_training_updates']==88
    replace(mutable[1],state)
    snapshot=P/'formal_live_observation_013_v1.json';save(snapshot,r)
    handoff=dict(status=current['status'],goal_status='active',mainline_validated=False,
        previous_goal_turn_classification='progress',this_goal_turn_classification='progress',
        recorded_remote_utc=r['recorded_utc'],recorded_client_utc=datetime.now(timezone.utc).isoformat(),
        canonical=str(P/'CURRENT_A60.json'),snapshot=str(snapshot),first_epoch_verification=first,
        original_supervisor=r['supervisor'],original_B=b['process'],actual_journal_lower_bound=counts,
        B_completed_epochs=1,B_current_epoch=2,M_started=False,Q_started=False,AP_available=False,
        global_model_forwards_lower_bound=276643+n,global_all_updates_lower_bound=233606+u,
        global_formal_updates_lower_bound=233518+u,discarded_capacity_updates=88,formal_epochs=289,
        reserved_global_forward_cap=512185,reserved_global_all_update_cap=262730,
        remaining_action='Continue only original supervisor. Next milestone B4 first AP; fixed12 learning gate controls unstarted M/Q. No rescue/restart or extra experiment.',
        next_readonly_query=str(P/'original_formal_readonly_query_v1.json'),bridge_session_id=19671,next_readonly_receipt='A60_formal_observe_014.json',
        do_not_rerun=['prepare/launch/freeze/publish/collect scripts','geometry/capacity/CPU fixture/checkpoint audits','B first-epoch audit already consumed','A58 jobs/audits'],
        prior_risk='CEOSOD full method unresolved; ConsistencyDet selected methods already constrain novelty. M/Q condition-statistic confound remains a mechanism limit, no extra arm now.',
        new_verification_forwards=0,new_verification_updates=0,new_COCO_evaluations=0,
        missing_for_goal_completion='Matched fixed12 detector outcomes, novelty/feasibility judgment and a defensible mainline remain unproven.')
    save(hp,handoff)
    for name,m in resolution.items():assert meta(Path(m['resolved_path']))==old['files'][name]
    files=set(Path(name) for name in old['files'])
    files.update(f for f in P.rglob('*') if f.is_file() and '__pycache__' not in f.parts)
    files.update(f for f in D.glob('A60*.json') if f.is_file())
    save(receipt_path,dict(status=current['status'],goal_status='active',mainline_validated=False,
        parent_receipt=str(parent),parent_receipt_sha256=sha(parent),parent_files_verified_before_publish=len(old['files']),
        parent_path_resolution=resolution,files={str(f):meta(f) for f in sorted(files)},
        scope='First-epoch evidence and explicit canonical snapshot updates; frozen experiment/source/protocol and prior result assets untouched.'))
    print(json.dumps(dict(status=current['status'],B_epoch1_verified_forwards=6471,B_epoch1_verified_updates=809,
        B_live_forwards_lower_bound=n,B_live_updates_lower_bound=u,B_completed_epochs=1,global_model_forwards=276643+n,
        global_all_updates=233606+u,global_formal_updates=233518+u,global_epochs=289,AP_available=False,
        parent_files_verified=len(old['files']),preserved_files=len(files),goal_status='active',mainline_validated=False),ensure_ascii=False))

if __name__=='__main__':main()
