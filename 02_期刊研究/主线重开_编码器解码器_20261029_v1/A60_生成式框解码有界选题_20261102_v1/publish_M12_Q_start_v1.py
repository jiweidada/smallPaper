"""Publish already verified B/M12 and an original Q live snapshot; no model calls."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

P = Path(__file__).resolve().parent
B = P.parent
D = B / 'A54_场景尺度约束候选_20261101_v1'
ROOTS = [B/'01_候选判别表.json', B/'15_当前状态与结果校验.json', P/'CURRENT_A60.json']
PARENT = P/'A60_M8_environment_pause_preservation_receipt_v1.json'
BACKUP = P/'before_M12_Q_start_publish_v1'
HANDOFF = P/'A60_turn_handoff_M12_Q_started_v1.json'
RECEIPT = P/'A60_M12_Q_start_preservation_receipt_v1.json'
STATUS = 'A60_BM_FIXED12_LOCALLY_VERIFIED_ORIGINAL_Q_RUNNING_FINISH_THEN_USER_PAUSE'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def meta(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}

def create(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def main():
    assert not RECEIPT.exists() and not HANDOFF.exists() and not BACKUP.exists()
    n = int(sys.argv[1])
    parsed = subprocess.run([sys.executable, str(P/'read_observer_receipt_v1.py'),
                             '--number', str(n), '--wait', '5'],
                            capture_output=True, text=True, check=True)
    saved = json.loads(parsed.stdout)
    o = saved['observation']
    assert o['source_bound'] and o['protocol_sha256'] == meta(P/'protocol_FROZEN_v1.json')['sha256']
    assert o['supervisor']['live'] and o['supervisor']['same_start']
    q = o['arms']['Q']
    assert q['process']['live'] and q['process']['same_start']
    assert q['process']['pid'] == 39733 and q['process']['actual_start_ticks'] == '33850461'
    evidence = {'status':'B_M_FIXED12_ORIGINAL_ARTIFACTS_LOCALLY_VERIFIED_Q_ORIGINAL_RUNNING',
                'observation':saved['receipt'], 'Q_process':q['process'],
                'Q_evaluations_available':[x['epoch'] for x in q['evaluations']],
                'verification_scope':'Fixed12 saved precision/recall reaggregation, original counts/input streams and model byte hashes; no rerun.'}
    for arm in ['B', 'M']:
        a = o['arms'][arm]
        vpath = P/(arm+'_fixed12_local_verification_v1.json')
        v = read(vpath)
        assert v['status'] == 'PASS_ORIGINAL_'+arm+'_FIXED12_ARTIFACTS_AND_LOCAL_MODEL_BACKUP'
        assert a['exit']['exit'] == 0 and a['completed_epochs'] == 12
        assert a['worker_reaped']['exit'] == 0 and a['worker_reaped']['wait_completed']
        assert v['metrics'] == a['result']['metrics']
        evidence[arm] = {'metrics':v['metrics'], 'latency':a['result']['latency'], 'verification':str(vpath),
                         'checkpoint':v['checkpoint_path'], 'checkpoint_sha256':v['checkpoint_sha256'],
                         'counts':v['original_counts']}
    fields = ['AP','AP50','AP75','APs','APs75','APm','APl','ARs']
    evidence['M_minus_B_pp'] = {k:evidence['M']['metrics'][k]-evidence['B']['metrics'][k] for k in fields}
    evidence['inference'] = 'M has no APs gain over B at fixed12 in this pilot; Q is still pending.'
    parent = read(PARENT)
    aliases = {}
    for path in ROOTS:
        keys = [k for k in parent['files'] if Path(k).name == path.name
                and Path(k).exists() and os.path.samefile(k, path)]
        assert keys, str(path)
        actual = meta(path)
        for key in keys:
            assert parent['files'][key] == actual, key
        aliases[str(path)] = keys
    board, state, current = [read(x) for x in ROOTS]
    now = datetime.now(timezone.utc).isoformat()
    protocol = read(P/'protocol_FROZEN_v1.json')
    forwards = sum(a['journal_counts']['started'] for a in o['arms'].values())
    updates = sum(a['journal_counts']['updates'] for a in o['arms'].values())
    epochs = sum(a['completed_epochs'] for a in o['arms'].values())
    before = protocol['global_before']
    report = P/'19_BM固定12轮对照与Q启动.txt'
    assert report.is_file()
    current.update({'status':STATUS, 'goal_status':'active', 'mainline_validated':False,
        'candidate_has_pilot_support':False, 'recorded_client_utc':now, 'recorded_utc':o['utc'],
        'snapshot_is_live_lower_bound':True, 'original_supervisor':o['supervisor'], 'arms':o['arms'],
        'source_binding_verified':True, 'observed_formal_model_forwards':forwards,
        'observed_formal_updates':updates, 'completed_formal_epochs':epochs,
        'global_observed_model_forwards':before['model_forwards']+forwards,
        'global_observed_all_optimizer_updates':before['all_optimizer_updates']+updates,
        'global_observed_formal_optimizer_updates':before['formal_optimizer_updates']+updates,
        'global_completed_formal_epochs':before['formal_epochs']+epochs,
        'currently_running_arm':'Q', 'remaining_authorized_queue':[], 'post_A60_training_queue':[],
        'pause_after_current_batch':str(P/'USER_FINISH_CURRENT_A60_THEN_PAUSE_v1.json'),
        'pause_condition_met':False, 'next_endpoint':'Original Q fixed12, then archive/verify/consolidate and pause.',
        'BM_fixed12_verified_evidence':evidence, 'M_AP_available':True, 'candidate_Q_tested':False,
        'fixed12_comparison_available':False, 'latest_report':str(report), 'latest_handoff':str(HANDOFF),
        'latest_preservation_receipt':str(RECEIPT), 'next_readonly_receipt':n+1,
        'final_assembly_script':str(P/'assemble_final_A60_v2.py'), 'read_only_bridge_session_id':3711,
        'timestamp_scope':'Client UTC observation receipt and monotonic elapsed; remote wall clock retained separately.',
        'current_goal_turn_classification':'PROGRESS_BM12_VERIFIED_Q_ORIGINAL_RUNNING'})
    for root in [board, state]:
        root.update({'status':STATUS,'goal_status':'active','mainline_validated':False,'training_running':True,
            'A60_execution':current,'A60_BM_fixed12_verified_evidence':evidence,'latest_update_utc':now,
            'latest_report':str(report),'latest_handoff':str(HANDOFF),'latest_preservation_receipt':str(RECEIPT),
            'pause_after_current_batch':str(P/'USER_FINISH_CURRENT_A60_THEN_PAUSE_v1.json'),
            'post_A60_training_queue':[],'current_phase':str(P)})
    row = next(x for x in board['candidates'] if x.get('id') == 'A60_GEOMETRIC_PROGRESS_BOX_DECODER')
    row.update({'status':STATUS,'report':str(report),'latest_report':str(report),
        'formal_observed_calls':forwards,'formal_observed_updates':updates,'B_completed_epochs':12,
        'M_completed_epochs':12,'Q_started':True,'Q_completed_epochs':q['completed_epochs'],
        'available_detection_epochs':{k:[x['epoch'] for x in v['evaluations']] for k,v in o['arms'].items()},
        'latest_metric_verification':str(P/'M_fixed12_local_verification_v1.json'),
        'evidence_scope':evidence['verification_scope']+' Q efficacy pending.',
        'BM_fixed12_verified_evidence':evidence,'mainline_validated':False,'efficacy_validated':False,
        'last_live_observation':{'path':saved['receipt'],'recorded_remote_utc':o['utc'],'Q_started':True},
        'post_A60_training_queue':[]})
    board['remaining_deliverable'] = 'Finish only original Q fixed12; collect all B/M/Q and then pause as explicitly requested.'
    state.update({'model_forwards':current['global_observed_model_forwards'],
        'training_updates':current['global_observed_all_optimizer_updates'],
        'detector_optimizer_updates_including_capacity':current['global_observed_all_optimizer_updates'],
        'formal_optimizer_updates_observed':current['global_observed_formal_optimizer_updates'],
        'formal_detector_training_updates':current['global_observed_formal_optimizer_updates'],
        'formal_training_epochs':current['global_completed_formal_epochs'],
        'counters_observed_lower_bounds_while_job_live':True,'counts_terminal':False,
        'current_goal_turn':'PROGRESS_BM12_VERIFIED_Q_ORIGINAL_RUNNING'})
    BACKUP.mkdir()
    resolution = {}
    for path in ROOTS:
        target = BACKUP/path.name
        shutil.copyfile(path, target)
        for key in [str(path), *aliases[str(path)]]:
            resolution[key] = {'resolved_path':str(target), **meta(target)}
    create(HANDOFF, {'status':STATUS,'goal_status':'active','mainline_validated':False,
        'recorded_client_utc':now,'source_observation':saved['receipt'],'BM_fixed12_verified_evidence':evidence,
        'original_supervisor':o['supervisor'],'original_Q':q['process'],'Q_progress':q['progress'],
        'bridge_session_id':3711,'bridge_script':str(D/'remote_A54.py'),'bridge_receipt_directory':str(D),
        'readonly_observer_source':str(P/'readonly_observer_reconnect_v2.py'),
        'next_readonly_receipt_hint':n+1,'numbering_rule':'Inspect max receipt number before a new request.',
        'latest_preservation_receipt':str(RECEIPT),'final_assembly_script':str(P/'assemble_final_A60_v2.py'),
        'user_instruction':str(P/'USER_FINISH_CURRENT_A60_THEN_PAUSE_v1.json'),
        'remaining_action':'Finish current Q, archive Q only, verify locally, assemble all B/M/Q, publish and pause; no extra experiments.'})
    for path, value in zip(ROOTS, [board,state,current]):
        temp = path.with_name(path.name+'.M12_Q_start_tmp')
        create(temp,value)
        temp.replace(path)
        for alias in aliases[str(path)]:
            if not os.path.samefile(alias,path):
                alias_path = Path(alias)
                assert alias_path.resolve().is_relative_to(B.parents[1])
                link = alias_path.with_name(alias_path.name+'.M12_Q_start_link_tmp')
                os.link(path,link)
                link.replace(alias_path)
    files = dict(parent['files'])
    for path in ROOTS:
        actual = meta(path)
        for key in [str(path), *aliases[str(path)]]:
            files[key] = actual
            assert os.path.samefile(key,path) and meta(key)==actual
    for path in [*BACKUP.iterdir(),PARENT,HANDOFF,Path(__file__),report,
                 P/'B_fixed12_local_verification_v1.json',P/'M_fixed12_local_verification_v1.json',
                 Path(saved['receipt']),P/'assemble_final_A60_v2.py',
                 D/'A60_M_fixed12_export_v1.json',D/'A60_M_fixed12_download_v1.json']:
        files[str(path)] = meta(path)
    create(RECEIPT,{'status':STATUS,'goal_status':'active','parent_receipt':str(PARENT),
        'parent_receipt_sha256':meta(PARENT)['sha256'],'parent_path_resolution':resolution,
        'aliases_updated':aliases,'unaffected_parent_files_rehashed_this_turn':False,
        'files':files,'pause_condition_met':False,'new_model_calls':0,'new_optimizer_updates':0,
        'training_protocol_changed':False})
    print(json.dumps({'status':STATUS,'global_forwards_lower_bound':current['global_observed_model_forwards'],
        'global_updates_lower_bound':current['global_observed_all_optimizer_updates'],
        'Q_completed_epochs':q['completed_epochs'],'M_fixed12_verified':True,'goal_status':'active'}))

if __name__ == '__main__':
    main()
