"""Publish completed original A60 evidence, ready for the explicitly requested goal pause.

Run only after all three local verifiers and the final organizer have succeeded.
This file never constructs a model, starts remote work, or changes training.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

P = Path(__file__).resolve().parent
B = P.parent
D = B/'A54_场景尺度约束候选_20261101_v1'
OUT = P/'本轮A60实验汇总_20261103_v1'
ROOTS = [B/'01_候选判别表.json', B/'15_当前状态与结果校验.json', P/'CURRENT_A60.json']
PARENT = P/'A60_M12_Q_start_preservation_receipt_v1.json'
BACKUP = P/'before_A60_organized_pause_publish_v1'
HANDOFF = P/'A60_turn_handoff_ORGANIZED_PAUSE_READY_v1.json'
RECEIPT = P/'A60_organized_pause_ready_preservation_receipt_v1.json'
STATUS = 'A60_ALL_THREE_FIXED12_VERIFIED_ORGANIZED_USER_PAUSE_READY'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def meta(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4 << 20), b''):
            h.update(block)
    return {'bytes':path.stat().st_size, 'sha256':h.hexdigest()}

def create(path, value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')

def main():
    assert not RECEIPT.exists() and not HANDOFF.exists() and not BACKUP.exists()
    n = int(sys.argv[1])
    original = subprocess.run([sys.executable,str(P/'read_observer_receipt_v1.py'),
                               '--number',str(n),'--wait','5'],
                              capture_output=True,text=True,check=True)
    saved = json.loads(original.stdout)
    observation = saved['observation']
    summary_path = OUT/'01_三组完整指标与判据.json'
    summary = read(summary_path)
    manifest = read(OUT/'06_汇总文件SHA256清单.json')
    close_proof = read(P/'OWNED_READONLY_BRIDGE_CLOSED_AFTER_A60_v1.json')
    assert close_proof['session_id'] == 3711 and close_proof['exit_code'] == 0
    assert summary['status'] == 'ALL_ORIGINAL_A60_FIXED12_VERIFIED_AND_ORGANIZED'
    assert summary['goal_pause_ready'] and not summary['mainline_validated']
    assert observation['source_bound'] and observation['protocol_sha256'] == meta(P/'protocol_FROZEN_v1.json')['sha256']
    assert observation['supervisor_exit'] == summary['original_supervisor_exit']
    assert observation['supervisor_exit']['status'] == 'ALL_THREE_FIXED12_COMPLETED'
    assert observation['supervisor_exit']['exit'] == 0 and observation['supervisor_exit']['remaining_queue'] == []
    assert not observation['supervisor']['live']
    expected = {'started':78514,'completed':78514,'updates':9708,'backwards':77652,
                'epochs':12,'train_forwards':77652,'eval_forwards':822,'latency_forwards':40}
    for arm in ['B','M','Q']:
        a = observation['arms'][arm]
        assert not a['process']['live'] and a['exit']['exit'] == 0
        assert a['worker_reaped']['exit'] == 0 and a['worker_reaped']['wait_completed']
        assert a['result']['counts'] == expected and a['completed_epochs'] == 12
        assert a['result']['metrics'] == summary['fixed12_metrics'][arm]
        v = read(P/(arm+'_fixed12_local_verification_v1.json'))
        assert v['status'] == 'PASS_ORIGINAL_'+arm+'_FIXED12_ARTIFACTS_AND_LOCAL_MODEL_BACKUP'
        assert v['metrics'] == summary['fixed12_metrics'][arm] and v['original_counts'] == expected
    assert len(summary['all_12_epoch_training_records_and_4_8_12_metrics']) == 36
    assert sum(x['metrics'] is not None for x in summary['all_12_epoch_training_records_and_4_8_12_metrics']) == 9
    actual_files = {x.relative_to(OUT).as_posix() for x in OUT.rglob('*') if x.is_file()}
    assert actual_files == set(manifest) | {'06_汇总文件SHA256清单.json'}
    for rel, expected_meta in manifest.items():
        file = OUT/rel
        assert file.resolve().is_relative_to(OUT.resolve()) and meta(file) == expected_meta, rel
    totals = summary['formal_counts_total']
    assert totals == {k:3*v for k,v in expected.items()}
    protocol = read(P/'protocol_FROZEN_v1.json')
    before = protocol['global_before']
    global_final = {'model_forwards':before['model_forwards']+totals['started'],
        'all_optimizer_updates':before['all_optimizer_updates']+totals['updates'],
        'formal_optimizer_updates':before['formal_optimizer_updates']+totals['updates'],
        'discarded_capacity_optimizer_updates':before['discarded_capacity_optimizer_updates'],
        'formal_epochs':before['formal_epochs']+totals['epochs']}
    assert global_final == protocol['global_with_full_reserved_cap']
    parent = read(PARENT)
    aliases = {}
    for path in ROOTS:
        keys = [k for k in parent['files'] if Path(k).name == path.name
                and Path(k).exists() and os.path.samefile(k,path)]
        assert keys
        actual = meta(path)
        for key in keys:
            assert actual == parent['files'][key], key
        aliases[str(path)] = keys
    board,state,current = [read(x) for x in ROOTS]
    now = datetime.now(timezone.utc).isoformat()
    report = OUT/'00_本轮结论与暂停说明.txt'
    evidence = {'summary':str(summary_path),'report':str(report),'index':str(OUT/'02_文件索引与本地权重位置.json'),
        'manifest':str(OUT/'06_汇总文件SHA256清单.json'),'fixed12_metrics':summary['fixed12_metrics'],
        'fixed12_delta':summary['fixed12_delta'],'fixed12_gates':summary['fixed12_gates'],
        'candidate_has_pilot_support':summary['candidate_has_pilot_support'],'mainline_validated':False,
        'scope':'All three original arms fixed12 raw, nine original 4/8/12 evaluation rows, 36 loss rows, source and model backups.',
        'original_supervisor_exit':summary['original_supervisor_exit'],'formal_counts_total':totals,
        'original_source_observation':saved['receipt'],'no_more_experiments':True}
    current.update({'status':STATUS,'goal_status':'active','goal_status_record_scope':'Actual tool status before requested final pause action.',
        'goal_pause_ready':True,'goal_pause_required_next':True,'pause_condition_met':True,
        'conditional_pause_request_fulfilled':True,'goal_is_not_complete':True,'mainline_validated':False,
        'candidate_has_pilot_support':summary['candidate_has_pilot_support'],'recorded_client_utc':now,
        'recorded_utc':observation['utc'],'snapshot_is_live_lower_bound':False,'training_running':False,
        'original_supervisor':observation['supervisor'],'arms':observation['arms'],'currently_running_arm':None,
        'source_binding_verified':True,'observed_formal_model_forwards':totals['started'],
        'observed_formal_updates':totals['updates'],'completed_formal_epochs':totals['epochs'],
        'global_observed_model_forwards':global_final['model_forwards'],
        'global_observed_all_optimizer_updates':global_final['all_optimizer_updates'],
        'global_observed_formal_optimizer_updates':global_final['formal_optimizer_updates'],
        'global_completed_formal_epochs':global_final['formal_epochs'],
        'remaining_authorized_queue':[],'post_A60_training_queue':[],
        'candidate_Q_tested':True,'fixed12_comparison_available':True,
        'organized_terminal_evidence':evidence,'report':str(report),'latest_report':str(report),'latest_handoff':str(HANDOFF),
        'latest_preservation_receipt':str(RECEIPT),'read_only_bridge_session_id':None,
        'owned_readonly_bridge_closed':close_proof,
        'next_endpoint':'User requested pause; no next experiment queued.',
        'current_goal_turn_classification':'PROGRESS_ALL_A60_ORGANIZED_USER_PAUSE_READY'})
    for root in [board,state]:
        root.update({'status':STATUS,'goal_status':'active','goal_status_record_scope':current['goal_status_record_scope'],
            'goal_pause_ready':True,'goal_pause_required_next':True,'pause_condition_met':True,
            'conditional_pause_request_fulfilled':True,'mainline_validated':False,'training_running':False,
            'candidate_has_pilot_support':summary['candidate_has_pilot_support'],
            'pilot_validation_status':'FIXED12_SCREEN_PASSED' if summary['candidate_has_pilot_support'] else 'FIXED12_SCREEN_FAILED',
            'A60_execution':current,'A60_terminal_result':evidence,'latest_update_utc':now,
            'latest_report':str(report),'latest_handoff':str(HANDOFF),'latest_preservation_receipt':str(RECEIPT),
            'post_A60_training_queue':[],'current_phase':str(P)})
    row = next(x for x in board['candidates'] if x.get('id') == 'A60_GEOMETRIC_PROGRESS_BOX_DECODER')
    row.update({'status':STATUS,'report':str(report),'latest_report':str(report),
        'formal_observed_calls':totals['started'],'formal_observed_updates':totals['updates'],
        'B_completed_epochs':12,'M_completed_epochs':12,'Q_completed_epochs':12,
        'available_detection_epochs':{a:[4,8,12] for a in ['B','M','Q']},
        'candidate_has_pilot_support':summary['candidate_has_pilot_support'],'mainline_validated':False,
        'efficacy_validated':False,'final_pilot_gate_passed':summary['candidate_has_pilot_support'],
        'final_result':evidence,'latest_metric_verification':str(P/'Q_fixed12_local_verification_v1.json'),
        'evidence_scope':evidence['scope']+' One short exploratory seed; no significance/convergence/unique-cause claim.',
        'last_live_observation':{'path':saved['receipt'],'all_original_processes_terminal':True},
        'post_A60_training_queue':[],'pause_condition_met':True})
    board['remaining_deliverable'] = 'Current batch completed and organized. Pause goal at explicit user request; no continuation queued.'
    state.update({'model_forwards':global_final['model_forwards'],'training_updates':global_final['all_optimizer_updates'],
        'detector_optimizer_updates_including_capacity':global_final['all_optimizer_updates'],
        'formal_optimizer_updates_observed':global_final['formal_optimizer_updates'],
        'formal_detector_training_updates':global_final['formal_optimizer_updates'],
        'formal_training_epochs':global_final['formal_epochs'],'counters_observed_lower_bounds_while_job_live':False,
        'counts_terminal':True,'training_counts_terminal':True,'current_native_job_terminal':True,
        'full_forward_cap_closed':True,
        'current_goal_turn':'PROGRESS_ALL_A60_ORGANIZED_USER_PAUSE_READY',
        'historical_saved_PR_counter_before_terminal_publication':state.get('CPU_saved_precision_recall_metric_aggregation_checks_A60'),
        'CPU_saved_precision_recall_metric_aggregation_checks_A60':6,
        'A60_saved_PR_verification_scope':
            {'terminal_fixed12_arms':['B','M','Q'],'metrics_per_terminal_arm':8,
             'verified_endpoint_records':['B4','B8','B12','M4','M12','Q12'],
             'count_scope':'Six existing endpoint verification reports; corrects the older counter lower bound, with no new inference or matching.',
             'intermediate_scope':'Read original 4/8 epoch trainer metrics during organization; the separate older B4/B8/M4 verification reports remain preserved.'}})
    BACKUP.mkdir()
    resolution = {}
    for path in ROOTS:
        dest = BACKUP/path.name
        shutil.copyfile(path,dest)
        for key in [str(path),*aliases[str(path)]]:
            resolution[key] = {'resolved_path':str(dest),**meta(dest)}
    create(HANDOFF,{'status':STATUS,'goal_status_before_pause_tool':'active','goal_pause_ready':True,
        'goal_is_not_complete':True,'mainline_validated':False,'recorded_client_utc':now,
        'user_words_verbatim':'跑完实验就先整理暂停吧','conditional_pause_request_fulfilled':True,
        'original_source_observation':saved['receipt'],'terminal_evidence':evidence,'global_final_counts':global_final,
        'original_supervisor':observation['supervisor'],'original_supervisor_exit':observation['supervisor_exit'],
        'owned_readonly_bridge_closed':close_proof,'post_A60_training_queue':[],
        'all_original_and_organized_assets_retained':True,'latest_preservation_receipt':str(RECEIPT),
        'remaining_action':'Call update_goal paused, report returned status and stop.',
        'resume_boundary':'No experiment pending. A later explicit resume must evaluate next work afresh; this conditional pause has been fulfilled.'})
    for path,value in zip(ROOTS,[board,state,current]):
        temp = path.with_name(path.name+'.organized_pause_tmp')
        create(temp,value)
        temp.replace(path)
        for alias in aliases[str(path)]:
            if not os.path.samefile(alias,path):
                alias_path = Path(alias)
                assert alias_path.resolve().is_relative_to(B.parents[1])
                link = alias_path.with_name(alias_path.name+'.organized_pause_link_tmp')
                os.link(path,link)
                link.replace(alias_path)
    files = dict(parent['files'])
    for path in ROOTS:
        actual = meta(path)
        for key in [str(path),*aliases[str(path)]]:
            files[key] = actual
            assert os.path.samefile(key,path) and meta(key) == actual
    for path in [*BACKUP.iterdir(),PARENT,HANDOFF,Path(__file__),Path(saved['receipt']),
                 *[P/(a+'_fixed12_local_verification_v1.json') for a in ['B','M','Q']],
                 P/'assemble_final_A60_v2.py',P/'read_observer_compact_v1.py',
                 P/'Q4_original_record_milestone_v1.json',P/'Q8_original_record_milestone_v1.json',
                 P/'OWNED_READONLY_BRIDGE_CLOSED_AFTER_A60_v1.json',D/'A60_Q_fixed12_export_sourcecheck_v1.json',
                 D/'A60_Q_fixed12_export_v1.json',D/'A60_Q_fixed12_download_v1.json']:
        files[str(path)] = meta(path)
    for rel,item in manifest.items():
        files[str(OUT/rel)] = item
    files[str(OUT/'06_汇总文件SHA256清单.json')] = meta(OUT/'06_汇总文件SHA256清单.json')
    create(RECEIPT,{'status':STATUS,'goal_status_before_pause_tool':'active','goal_pause_ready':True,
        'parent_receipt':str(PARENT),'parent_receipt_sha256':meta(PARENT)['sha256'],
        'parent_path_resolution':resolution,'aliases_updated':aliases,
        'unaffected_parent_files_rehashed_this_turn':False,'organized_package_manifest_all_rehashed':True,
        'files':files,'pause_condition_met':True,'post_A60_training_queue':[],
        'new_model_calls':0,'new_optimizer_updates':0,'training_protocol_changed':False})
    print(json.dumps({'status':STATUS,'goal_pause_ready':True,'candidate_has_pilot_support':summary['candidate_has_pilot_support'],
        'all_fixed12_metrics':summary['fixed12_metrics'],'global_final_counts':global_final,
        'report':str(report),'manifest_files_reverified':len(manifest)},ensure_ascii=True))

if __name__ == '__main__':
    main()
