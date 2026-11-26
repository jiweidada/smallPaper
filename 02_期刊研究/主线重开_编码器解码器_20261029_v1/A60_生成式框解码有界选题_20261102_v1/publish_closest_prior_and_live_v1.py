"""Append closest-prior evidence and publish a live snapshot without altering runs."""
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
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()

def meta(path):return dict(bytes=path.stat().st_size,sha256=sha(path))

def save(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def replace(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    assert not (P/'A60_closest_prior_live_preservation_receipt_v1.json').exists()
    parent=P/'A60_formal_start_preservation_receipt_v1.json'
    appendix=P/'A60_formal_start_append_receipt_v1.json'
    parents=[json.loads(f.read_text()) for f in [parent,appendix]]
    for receipt in parents:
        for name,m in receipt['files'].items():assert meta(Path(name))==m,name
    raw=D/'A60_formal_observe_006.json';wrapper=json.loads(raw.read_text());assert wrapper['exit']==0
    r=json.loads(wrapper['stdout']);b=r['arms']['B']
    assert r['source_binding'] and r['supervisor_exit'] is None and r['supervisor']['live']
    assert r['supervisor']['pid']==30238 and r['supervisor']['actual_start_ticks']=='30772425'
    assert b['process']['pid']==30239 and b['process']['actual_start_ticks']=='30772460' and b['process']['live']
    assert b['exit'] is None and b['completed_epochs']==0
    assert not r['arms']['M']['exists'] and not r['arms']['Q']['exists']
    scope=json.loads((P/'protocol_FROZEN_v1.json').read_text())
    assert sha(P/'protocol_FROZEN_v1.json')==r['protocol_sha256']
    for n,h in scope['source_sha256'].items():assert sha(P/n)==h,n
    artifact=P/'closest_prior_supplement_v1/ConsistencyDet_v5.html'
    assert b'arXiv:2404.07773v5' in artifact.read_bytes()
    commit=json.loads((P/'closest_prior_supplement_v1/CEOSOD_default_commit.json').read_text())[0]['sha']
    assert commit=='2a3a49d1c193ff5996f1f39c41df8914eccd65b1'
    probe=json.loads((P/'closest_prior_supplement_v1/CEOSOD_publisher_access_probe_v1.json').read_text())
    assert all(not x['is_PDF'] for x in probe)
    mutable=[BASE/'01_候选判别表.json',BASE/'15_当前状态与结果校验.json',P/'CURRENT_A60.json']
    before=P/'before_closest_prior_publish_v1';before.mkdir();resolution={}
    for f in mutable:
        dest=before/f.name;shutil.copyfile(f,dest)
        assert meta(dest)==parents[0]['files'][str(f)]
        resolution[str(f)]=dict(resolved_path=str(dest),**meta(dest))
    counts=b['journal_counts'];n=counts['started'];u=counts['updates']
    current=json.loads(mutable[2].read_text())
    current.update(status='A60_ORIGINAL_B1_LIVE_CLOSEST_PRIOR_SUPPLEMENT',recorded_utc=r['recorded_utc'],
        source_timestamp_scope='Server UTC as reported; client fetch timestamps are separate machine clocks. No cross-machine ETA inference.',
        previous_goal_turn_classification='progress',current_goal_turn_classification='progress with live-process verified wait',
        original_supervisor=r['supervisor'],arms=r['arms'],source_binding_verified=True,
        observed_formal_model_forwards=n,observed_formal_updates=u,completed_formal_epochs=0,
        global_observed_model_forwards=276643+n,global_observed_all_optimizer_updates=233606+u,
        global_observed_formal_optimizer_updates=233518+u,global_completed_formal_epochs=288,
        closest_prior_supplement=str(P/'02b_核心文献矩阵增补_ConsistencyDet_v1.json'),
        closest_prior_report=str(P/'05_最近方法补查与当前主张边界.txt'),
        CEOSOD_full_method_read=False,CEOSOD_default_pin_unchanged=commit,
        novelty_cleared=False,training_protocol_modified=False,
        mechanism_identifiability_limit='M/Q condition distributions and within-image query uniformity also change; a win does not uniquely establish a geometric cause. No fourth arm allocated.',
        next_endpoint='Continue original B; first AP at4, frozen12 learning gate. No early-curve rescue or repeat.')
    replace(mutable[2],current)
    board=json.loads(mutable[0].read_text());board['A60_execution']=current
    for row in board['candidates']:
        if row.get('id')=='A60_GEOMETRIC_PROGRESS_BOX_DECODER':
            row.update(status=current['status'],formal_observed_calls=n,formal_observed_updates=u,
                matrix_supplement=current['closest_prior_supplement'],closest_prior_report=current['closest_prior_report'],
                novelty_cleared=False,evidence_scope='Real B training live, no AP; selected ConsistencyDet v5 methods checked; CEOSOD full-method risk unresolved.')
    board.update(status=current['status'],latest_update_utc=r['recorded_utc'],goal_status='active',mainline_validated=False,
        pilot_validation_status='B1 original live, M/Q unstarted; no AP',
        closest_prior_supplement=current['closest_prior_supplement'],closest_prior_report=current['closest_prior_report'])
    replace(mutable[0],board)
    state=json.loads(mutable[1].read_text());state['A60_execution']=current
    state.update(status=current['status'],recorded_utc=r['recorded_utc'],latest_update_utc=r['recorded_utc'],goal_status='active',
        previous_goal_turn_classification='progress',current_goal_turn='PROGRESS_CLOSEST_PRIOR_METHOD_CHECK_AND_LIVE_VERIFIED_WAIT',
        mainline_validated=False,model_forwards=276643+n,training_updates=233606+u,
        detector_optimizer_updates_including_capacity=233606+u,formal_detector_training_updates=233518+u,
        formal_optimizer_updates_observed=233518+u,formal_training_epochs=288,
        cap=512185,counts_terminal=False,training_counts_terminal=False,full_forward_cap_closed=False,
        counters_observed_lower_bounds_while_job_live=True,pilot_validation_status='Original B1 live; no AP',
        closest_prior_supplement=current['closest_prior_supplement'],closest_prior_report=current['closest_prior_report'])
    replace(mutable[1],state)
    snapshot=P/'formal_live_observation_006_v1.json';save(snapshot,r)
    handoff=dict(status=current['status'],goal_status='active',mainline_validated=False,
        previous_goal_turn_classification='progress',this_goal_turn_classification='progress',
        recorded_remote_utc=r['recorded_utc'],recorded_client_utc=datetime.now(timezone.utc).isoformat(),
        snapshot=str(snapshot),canonical=str(P/'CURRENT_A60.json'),original_supervisor=r['supervisor'],original_B=b['process'],
        formal_journal_lower_bound=counts,completed_epochs=0,M_started=False,Q_started=False,B_AP_available=False,
        global_model_forwards_lower_bound=276643+n,global_all_updates_lower_bound=233606+u,
        global_formal_updates_lower_bound=233518+u,discarded_capacity_updates=88,formal_epochs=288,
        reserved_global_forward_cap=512185,reserved_global_all_update_cap=262730,
        remaining_action='Wait for original B4 first evaluation; preserve fixed12 B learning gate then original M/Q. No new training launch or diagnostic rerun.',
        next_readonly_query=str(P/'original_formal_readonly_query_v1.json'),bridge_session_id=19671,
        next_readonly_receipt='A60_formal_observe_007.json',
        research_progress='Added selected latest ConsistencyDet methods, tightened novelty/mechanism claims; CEOSOD exact DOI/publisher confirmed but normal accesses gave empty202; full-method overlap remains unresolved.',
        new_review_model_forwards=0,new_review_optimizer_updates=0,training_code_or_gates_modified=False,
        phase_local_matrix_supplement=current['closest_prior_supplement'],scope_report=current['closest_prior_report'],
        do_not_rerun=['launch/prepare/freeze scripts','capacity','geometryD0','CPU fixture/checkpoint checks','past A58 jobs or audits'],
        no_completion_audit_passed='All matched fixed12 outcomes and reliable novelty/feasibility judgment are still missing; keep full research goal active.')
    hp=P/'A60_turn_handoff_closest_prior_v1.json';save(hp,handoff)
    for path,m in resolution.items():assert meta(Path(m['resolved_path']))==parents[0]['files'][path]
    files=set(Path(x) for receipt in parents for x in receipt['files'])
    files.update(f for f in P.rglob('*') if f.is_file() and '__pycache__' not in f.parts)
    files.update(f for f in D.glob('A60_formal_observe_*.json') if f.is_file())
    save(P/'A60_closest_prior_live_preservation_receipt_v1.json',dict(status=current['status'],goal_status='active',mainline_validated=False,
        parents=[dict(path=str(x),sha256=sha(x)) for x in [parent,appendix]],
        parent_files_verified_unchanged_before_publish=sum(len(x['files']) for x in parents),parent_path_resolution=resolution,
        files={str(f):meta(f) for f in sorted(files)},
        scope='Phase evidence append and three explicit canonical snapshot updates; no experiment/plan/source/metric changes or prior result deletion.'))
    print(json.dumps(dict(status=current['status'],B_forwards_started=n,B_updates=u,B_completed_epochs=0,
        global_forwards_lower_bound=276643+n,global_updates_lower_bound=233606+u,
        source_binding=True,prior_files_verified=sum(len(x['files']) for x in parents),preserved_files=len(files),
        goal_status='active',mainline_validated=False),ensure_ascii=False))

if __name__=='__main__':main()
