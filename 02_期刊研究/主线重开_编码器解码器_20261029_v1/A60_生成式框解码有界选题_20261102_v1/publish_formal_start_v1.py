"""Publish a dated live snapshot, preserving the complete prior A58 receipt chain."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import shutil

P=Path(__file__).resolve().parent
BASE=P.parent
ENG=BASE.parent
D=BASE/'A54_场景尺度约束候选_20261101_v1'
P59=BASE/'A59_非均匀采样与坐标一致性_20261102_v1'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def save(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)

def meta(path):return dict(bytes=path.stat().st_size,sha256=sha(path))

def main():
    assert not (P/'CURRENT_A60.json').exists()
    parent=BASE/'A58_区域范围路由候选_20261102_v1/A58_fixed12_terminal_preservation_receipt_v1.json'
    old=json.loads(parent.read_text())
    for name,m in old['files'].items():
        path=Path(name);assert path.is_file() and meta(path)==m,'Prior receipt mismatch: '+name
    wrapper=json.loads((D/'A60_formal_observe_001.json').read_text());assert wrapper['exit']==0
    observed=json.loads(wrapper['stdout'])
    assert observed['source_binding'] and observed['supervisor']['live']
    assert observed['arms']['B']['process']['live'] and observed['arms']['B']['journal_counts']['updates']>=25
    assert observed['supervisor_exit'] is None and observed['arms']['B']['exit'] is None
    assert not observed['arms']['M']['exists'] and not observed['arms']['Q']['exists']
    scope=json.loads((P/'protocol_FROZEN_v1.json').read_text())
    assert observed['protocol_sha256']==sha(P/'protocol_FROZEN_v1.json')
    before=P/'before_A60_canonical_publish_v1';before.mkdir()
    roots=[BASE/'01_候选判别表.json',BASE/'15_当前状态与结果校验.json']
    resolutions={}
    for path in roots:
        dest=before/path.name;shutil.copyfile(path,dest);assert meta(dest)==old['files'][str(path)]
        resolutions[str(path)]=dict(resolved_path=str(dest),**meta(dest))
    a60F=sum(a['journal_counts']['started'] for a in observed['arms'].values())
    a60U=sum(a['journal_counts']['updates'] for a in observed['arms'].values())
    a60E=sum(a['completed_epochs'] for a in observed['arms'].values())
    current=dict(status='A60_FIXED12_B_RUNNING',goal_status='active',mainline_validated=False,
        candidate_has_pilot_support=False,candidate_selected_for_pilot=True,
        recorded_utc=observed['recorded_utc'],snapshot_is_live_lower_bound=True,
        candidate='尺度归一化几何进度条件化的单步框解码',
        question='相同尺寸偏采样扰动框下，几何进度条件能否超出物理时刻条件？',
        dataset='VisDrone6471train/548val',
        host='Controlled R50-FPN + DiffusionDet six-stage DynamicHead; linear CFM baseline, not complete author FlowDet reproduction',
        report=str(P/'04_正式验证启动与当前判别.txt'),research_report=str(P/'03_候选问题_最近方法_最小验证.txt'),
        matrix=str(P/'02_核心文献矩阵与阅读范围_v1.json'),protocol=str(P/'protocol_FROZEN_v1.json'),
        protocol_sha256=sha(P/'protocol_FROZEN_v1.json'),initialization_sha256=scope['initialization_sha256'],
        capacity=dict(status='PASS',model_forwards=9,discarded_updates=6,peak_allocated_bytes=3717671936,
            result=str(P/'capacity_local_verification_v1.json'),weights_reused=False),
        original_supervisor=observed['supervisor'],arms=observed['arms'],source_binding_verified=True,
        observed_formal_model_forwards=a60F,observed_formal_updates=a60U,completed_formal_epochs=a60E,
        remaining_authorized_queue=['M','Q'],queue_conditions='B fixed12 AP>=8 and APs>=3; failures cancel unstarted arms; no retry',
        full_reserved_forward_cap=512185,global_observed_model_forwards=276643+a60F,
        global_observed_all_optimizer_updates=233606+a60U,
        global_observed_formal_optimizer_updates=233518+a60U,
        global_discarded_capacity_updates=88,global_completed_formal_epochs=288+a60E,
        next_endpoint='B4 first AP; primary decision remains fixed12 B/M/Q, not early-curve selection',
        scope='Actual live training verified. No new AP yet, no efficacy, convergence, global novelty or final mainline claim.',
        previous_A58='Retired exact recipe; no files/models/verdicts changed',
        A59='Bounded primary review, inverse mapping already in closest sources; no training allocated; not a trained negative',
        old_route_stops_preserved=['A42','A43','A44','A46','A47','A51','A54','A58'])
    # A compact public-facing report; all detailed raw readings remain separate.
    text=(
        'A60正式验证启动与当前判别\n'
        f"时间：{observed['recorded_utc']}（UTC，运行中快照，下界）\n"
        '当前问题：极小框的相对定位阶段在普通线性框插值中占比很低；这是否构成单步decoder可改善的训练条件瓶颈，仍待检测结果验证。\n'
        '候选：尺度归一化几何进度条件化的单步框解码。不是新encoder架构或已证实主线。\n'
        '数据：VisDrone，6471训练/548验证，640×640，相同标签与图序；不删除密集图GT。\n'
        '算法：受控R50-FPN + DiffusionDet六级DynamicHead，线性CFM基线依据FlowDet；不是完整作者FlowDet复现。110565396参数，不能称轻量化。\n'
        'B：普通线性插值，物理时刻；M：尺寸偏插值，实际物理时刻；Q：与M相同的扰动框，几何进度条件。三组参数和推理调用数相同。\n'
        'M/Q对照的目的：分离普通尺寸偏采样与条件变量的贡献。两组实际完整输入流必须核对，当前只有B开训，尚不能声称跨组实际输入核验完成。\n'
        '容量：9次整模型前向/6次丢弃更新，902GT密集图通过，峰值allocated约3.462GiB。容量权重未复用。\n'
        f"实际启动：原监督器PID{observed['supervisor']['pid']}/start{observed['supervisor']['actual_start_ticks']}；"
        f"原B训练PID{observed['arms']['B']['process']['pid']}/start{observed['arms']['B']['process']['actual_start_ticks']}，双方身份匹配且存活。\n"
        f'已有正式记录：{a60F}次整模型前向开始，{a60U}次优化器更新返回，完成{a60E}轮；无AP。\n'
        '训练：各组fresh同SHA初始化/seed20261102，microbatch1累积8（最后7张取平均），固定12轮原始模型，无EMA/最佳轮选择；第4/8/12轮评估。\n'
        '停止：B12 AP<8或APs<3则取消未启动M/Q；异常/单组12小时或总36小时上限停止，无自动重试/续训/换参数。\n'
        '继续门槛：Q−B APs≥0.5pp、Q−M APs≥0.3pp、Q−B AP≥−0.2pp、Q/M中位模型延迟≤1.10，全部通过才继续投入。\n'
        '口径：COCO-style十类/maxDets300/原图裁剪GT面积small≤1024/class0区域并集IoA过滤，不是官方VisDrone挑战分数。\n'
        '证据边界：几何计算成立、真实GPU能运行；解码器是否因此漏检、Q是否有效、最近工作是否完全不重合均未证明。CEOSOD仅摘要/README阅读，全文重合仍有风险。\n'
        '历史：A58终点和裁决原样保留；A59未立项而非训练阴性；423条文献主索引未改。总体目标active。\n')
    with (P/'04_正式验证启动与当前判别.txt').open('x',encoding='utf-8') as f:f.write(text)
    save(P/'CURRENT_A60.json',current)
    board=json.loads(roots[0].read_text(encoding='utf-8-sig'))
    board['historical_top_level_before_A60']={k:board.get(k) for k in ['status','current_host_candidate','model_candidates','existing_reference','current_priority_question','next_decision_protocol']}
    assert not any(c.get('id','').startswith('A60_') for c in board['candidates'])
    a59=dict(id='A59_NONUNIFORM_SAMPLING_COORDINATE_REVIEW',name='非均匀采样与原坐标一致性',
        status='NOT_SELECTED_AFTER_BOUNDED_PRIMARY_REVIEW_NO_TRAINING',selected_for_training=False,model_forwards=0,optimizer_updates=0,
        report=str(P59/'01_原文核查与暂不立项.txt'),scope='FOVEA already original-space losses and RoI-related coordinate correction. Not global warp equivalence or trained negative.')
    a60=dict(id='A60_GEOMETRIC_PROGRESS_BOX_DECODER',name=current['candidate'],status=current['status'],
        question=current['question'],arms=scope['arms'],selected_for_training=True,mainline_validated=False,novelty_cleared=False,
        dataset=current['dataset'],algorithm=current['host'],report=current['report'],matrix=current['matrix'],
        formal_protocol=current['protocol'],capacity_result=current['capacity'],training_allocation=235542,
        formal_observed_calls=a60F,formal_observed_updates=a60U,continuation_gates=scope['continuation_gates'],
        historical_difference='Condition variable in generative box training; no new local read/teacher/scale route/region attention; old stops retained.',
        evidence_scope='Geometry and capacity passed, actual B training live; no new AP or mainline efficacy.')
    board['candidates'].extend([a59,a60]);board['A59_review']=a59;board['A60_execution']=current
    board.update(status=current['status'],latest_update_utc=observed['recorded_utc'],goal_status='active',mainline_validated=False,
        current_priority_candidate=a60['id'],current_priority_question=current['question'],current_host_candidate=current['host'],
        model_candidates=[current['host']],existing_reference=current['initialization_sha256'],
        mainline_selected_for_pilot=True,candidate_selected_for_pilot=True,
        mainline_selected_for_pilot_scope='Candidate selected for bounded testing, not scientific acceptance.',
        latest_report=current['report'],pilot_validation_status='B live, M/Q conditional queue; no AP',training_running=True,
        next_decision_protocol=scope['continuation_gates'])
    state=json.loads(roots[1].read_text(encoding='utf-8-sig'))
    state['A59_review']=a59;state['A60_execution']=current
    state.update(recorded_utc=observed['recorded_utc'],latest_update_utc=observed['recorded_utc'],goal_status='active',
        status=current['status'],current_phase='A60_FIXED12_FORMAL',phase='A60_FIXED12_FORMAL',
        current_goal_turn='PROGRESS_A59_REVIEW_A60_GEOMETRY_CAPACITY_AND_FORMAL_B_START',previous_goal_turn_classification='progress',
        current_priority_candidate=a60['id'],current_priority_question=current['question'],current_host_candidate=current['host'],
        current_user_task_phase='Bounded formal pilot for next distinct mainline candidate',current_user_task_status='training running, no AP',
        mainline_validated=False,mainline_selected_for_pilot=True,candidate_selected_for_pilot=True,
        mainline_selected_for_pilot_scope='Candidate selected for testing; mainline efficacy is not established.',
        latest_report=current['report'],latest_authoritative_phase=str(P/'CURRENT_A60.json'),
        model_forwards=276643+a60F,cap=512185,training_updates=233606+a60U,
        formal_detector_training_updates=233518+a60U,formal_optimizer_updates_observed=233518+a60U,
        capacity_only_optimizer_updates=88,discarded_capacity_optimizer_updates_observed=88,
        detector_optimizer_updates_including_capacity=233606+a60U,formal_training_epochs=288+a60E,
        training_counts_terminal=False,counts_terminal=False,counters_observed_lower_bounds_while_job_live=True,
        full_forward_cap_closed=False,cap_scope='Historical closed276643 + A60 reserved235542; maxima512185. Observed counts from original live journals, not allocated as completed.',
        formal_training_epochs_scope='Historical288 throughA58 plus A60 fully completed epochs; no partial epoch rounded up.',
        training_update_scope='All optimizer updates including88 discarded capacity updates; formal count is separate.',
        model_forward_scope='Historical276634 + A60 capacity9 + actual formal forward-start journals; excludes module-only CPU fixtures.',
        pilot_validation_status='Original B live; M/Q unstarted conditional queue; no new AP',training_running=True)
    state['A60_CPU_and_manifest_ops']=dict(detector_constructors=1,detector_forwards=0,
        reference_backbone_constructors=1,backbone_module_forwards=2,head_module_forwards=2,
        standalone_clock_training_calls=15,GT_free_clock_input_calls=1,synthetic_matcher_calls=6,
        criterion_only_forward_calls=2,criterion_only_backwards=2,manifest_header_reads=7019,manifest_annotation_reads=7019,
        caveat='These are A60 CPU/interface preparation operations, not whole-detector forward counts or formal optimizer updates.')
    for path,value in zip(roots,[board,state]):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    # Parent historical files resolve byte-for-byte through the two explicit snapshots.
    for name,resolution in resolutions.items():assert meta(Path(resolution['resolved_path']))==old['files'][name]
    files=set(roots)
    for folder in [P,P59]:
        files.update(f for f in folder.rglob('*') if f.is_file() and '__pycache__' not in f.parts)
    archive=ENG/'文献_相关原始论文_20261004'
    for pattern in ['A59*','A60*']:
        files.update(f for f in archive.glob(pattern) if f.is_file())
        files.update(f for f in (archive/'_text').glob(pattern) if f.is_file())
    files.update(f for f in D.glob('A60*.json') if f.is_file())
    receipt=dict(status=current['status'],goal_status='active',mainline_validated=False,
        recorded_utc=datetime.now(timezone.utc).isoformat(),parent_receipt=str(parent),parent_receipt_sha256=sha(parent),
        parent_files_verified_before_publish=len(old['files']),parent_path_resolution=resolutions,
        snapshot=str(P/'CURRENT_A60.json'),scope='Phase non-cache artifacts, new primary source archives and canonical roots; no deletion or previous result mutation.',
        files={str(f):meta(f) for f in sorted(files)})
    save(P/'A60_formal_start_preservation_receipt_v1.json',receipt)
    print(json.dumps(dict(status=current['status'],parent_files_verified=len(old['files']),new_preserved_files=len(files),
        global_forwards_lower_bound=state['model_forwards'],global_updates_lower_bound=state['training_updates'],
        reserved_global_forward_cap=512185,goal_status='active',mainline_validated=False),ensure_ascii=False))

if __name__=='__main__':main()
