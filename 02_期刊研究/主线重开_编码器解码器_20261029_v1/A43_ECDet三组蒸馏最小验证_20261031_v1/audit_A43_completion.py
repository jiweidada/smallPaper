"""Audit every part of the bounded A43 objective using current local evidence."""
import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path
from datetime import datetime, timezone

D = Path(__file__).resolve().parent
BASE = D.parent
V = D / 'result_verification_v1'
N = D / 'next_candidate_A44_v1'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def near(a, b, tolerance=1e-10):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tolerance, (a, b)


def main():
    requirements = []
    def proven(name, evidence, explanation):
        requirements.append({'requirement': name, 'status': 'PROVED_WITH_SCOPED_EVIDENCE',
                             'evidence': [str(p) for p in evidence], 'scope': explanation})
    board = read(BASE / '01_候选判别表.json')
    state = read(BASE / '15_当前状态与结果校验.json')
    event = read(D / 'final_report_evidence.json')
    gate = read(V / 'final_practical_gate_verification.json')
    controls = read(V / 'B_M_Q_controls_verified.json')
    stats = read(V / 'statistical_interpretation_11_check.json')
    protocol = read(D / 'startup_evidence/execution_protocol_FROZEN.json')
    assert sha(D / 'startup_evidence/execution_protocol_FROZEN.json') == 'cb82214e481387ecf3510f6f7ff76d2161e5828136ba1571b81d5b4dc1365280'
    assert gate['status'] == controls['status'] == 'PASS'
    assert protocol['arms'] == ['B', 'M', 'Q'] and protocol['epochs_per_group'] == 12
    assert protocol['checkpoint_selection'] == 'fixed epoch12 EMA for every arm'

    # Verify previous immutable receipts against actual current or preserved bytes.
    history_resolution = {}
    snapshots = ['canonical_before_A43', 'state_before_B_terminal', 'state_before_A43_terminal']
    history_receipts = [BASE/'99_A42终点独立核验与去留回执.json',
                        BASE/'101_A43启动容量与冻结核验回执.json',
                        BASE/'103_A43原版终点独立核验回执.json']
    for receipt in history_receipts:
        obj = read(receipt)
        resolutions = {}
        for name, expected in obj['files_sha256'].items():
            target = Path(name)
            alternatives = [target] + [D/s/target.name for s in snapshots]
            actual = next((p for p in alternatives if p.exists() and sha(p) == expected), None)
            assert actual is not None, (receipt.name, name, expected)
            resolutions[name] = {'verified_at': str(actual), 'sha256': expected}
        history_resolution[str(receipt)] = resolutions
    old_gate = D.parent/'A42_ECDet三组最小验证_20261031_v1/result_verification_v1/final_practical_gate_verification.json'
    assert not read(old_gate)['pilot_pass']
    remap = read(D/'B_terminal_state_historical_remap.json')
    assert remap['source_receipt_sha256'] == sha(history_receipts[-1])
    proven('Preserve independent A42 negative and signed historical state', history_receipts+[old_gate,D/'B_terminal_state_historical_remap.json'],
           'Every signed input resolved to matching current/preserved bytes; A42 negative retained; old receipts not edited.')

    init = {}
    rows = {}
    all_hash_count = {}
    evidence_files = []
    for arm in ['B','M','Q']:
        root = D/'completed_results'/arm
        folder = root/('arm_'+arm)
        download = read(D/('completed_arm_'+arm+'_download_verification.json'))
        archive = D/('completed_arm_'+arm+'_bundle.zip')
        assert download['status'] == 'PASS'
        assert archive.stat().st_size == download['metadata']['bytes']
        assert sha(archive) == download['metadata']['sha256']
        with zipfile.ZipFile(archive) as z:
            manifest = json.loads(z.read('member_sha256.json'))
            assert len(manifest) == download['member_files_verified'] == 128
            for name, digest in manifest.items():
                assert sha(root/name) == digest, (arm,name)
        plan = read(root/'execution_protocol_FROZEN.json')
        assert plan == protocol
        for name, digest in plan['files_sha256'].items():
            if name == 'common_init_10class.pth':
                target = D/name
            elif name == 'teacher_B_epoch12.pth':
                target = D.parent/'A42_ECDet三组最小验证_20261031_v1/completed_results/B/arm_B/epoch12.pth'
            else:
                target = root/name
            assert sha(target) == digest, (arm,name)
        all_hash_count[arm] = len(plan['files_sha256'])
        init[arm] = read(folder/'initialization.json')
        row = read(folder/'result.json'); rows[arm] = row
        assert row['status'] == 'COMPLETED' and row['epochs'] == 12
        assert row['optimizer_updates'] == 9708 and row['full_model_forwards'] == 10159
        assert row['teacher_forwards'] == (0 if arm=='B' else 9708)
        assert read(folder/'exit_receipt.json')['exit_code'] == 0
        epochs = [json.loads(s) for s in (folder/'epochs.jsonl').read_text().splitlines()]
        assert [e['epoch'] for e in epochs] == list(range(1,13))
        assert all(e['optimizer_updates'] == e['epoch']*809 for e in epochs)
        assert all(math.isfinite(e[k]) for e in epochs for k in ['mean_native_loss','mean_distillation_loss'])
        cpu_path = V/(arm+'_epoch12_checkpoint_cpu_verification.json')
        cpu = read(cpu_path); assert cpu['status'] == 'PASS'
        assert cpu['checkpoint_sha256'] == sha(folder/'epoch12.pth')
        assert cpu['optimizer_steps_min'] == cpu['optimizer_steps_max'] == 9708
        assert not cpu['cuda_initialized']
        for role in ['raw','EMA']:
            native = cpu['raw_and_EMA_native_inference'][role]
            assert native['strict_native_load'] and native['all_parameters_finite']
            assert native['auxiliary_keys_absent'] and native['inference_parameters'] == 9803112
        if arm != 'B':
            assert row['teacher_frozen_unchanged']
            assert row['teacher_state_start_digest'] == row['teacher_state_end_digest']
            assert cpu['reconstructed_initial_auxiliary_matches_saved_digest']
            assert set(cpu['auxiliary_optimizer_steps'].values()) == {9708}
        dcheck = read(V/(arm+'_CPU_receipt_download_check.json'))
        assert dcheck['status']=='PASS' and dcheck['downloaded_CPU_object_matches_actual_remote_output']
        assert dcheck['local_CPU_receipt_sha256']==sha(cpu_path)
        met_path = V/(arm+'_epoch12_independent_metrics.json')
        met = read(met_path); assert met['status']=='PASS'
        assert met['all_precision_cells_max_abs_difference']==0
        assert met['all_recall_cells_max_abs_difference']==0
        assert met['gt_sha256']==sha(folder/'epoch12_EMA_GT.json')
        assert met['predictions_sha256']==sha(folder/'epoch12_EMA_predictions.json')
        ig = read(V/(arm+'_epoch12_independent_GT_ignore.json'))
        assert ig['status']=='PASS' and ig['raw_annotation_files_verified']==548
        assert ig['entire_evaluated_GT_reconstructed_from_raw_annotations']
        assert ig['evaluated_GT']==38602 and ig['ignored_valid_GT']==157
        assert ig['retained_predictions_inside_ignore_regions']==0
        for k in ['AP','APs','APs75']:
            near(met['metrics'][k],row['endpoint_metrics'][k])
            near(met['metrics'][k],gate['metrics'][arm][k])
        lat = row['latency']['samples_ms']
        assert len(lat)==30 and all(math.isfinite(t) and t>0 for t in lat)
        near(statistics.median(lat),gate['latency_median_ms'][arm])
        evidence_files += [archive,D/('completed_arm_'+arm+'_download_verification.json'),
                           cpu_path,V/(arm+'_CPU_receipt_download_check.json'),met_path,
                           V/(arm+'_epoch12_independent_metrics.npz'),
                           V/(arm+'_epoch12_independent_GT_ignore.json')]
    assert init['M']==init['Q']
    assert all(i['all_base_tensors_equal'] for i in init.values())
    assert len({i['common_init_sha256'] for i in init.values()})==1
    assert all(i['native_inference_parameters']==9803112 for i in init.values())
    assert init['M']['training_model_parameters']==init['Q']['training_model_parameters']==9812520
    proven('Freeze actual B/M/Q source, common fresh initialization and method', [D/'startup_evidence/execution_protocol_FROZEN.json',V/'B_M_Q_controls_verified.json'],
           f"Each128-member archive and extracted bytes checked; each arm rehashed{all_hash_count['B']} frozen inputs. Native base/initial tensors equal; M/Q same9408-param projection. Frozen sources preserve original decoder.")
    proven('Use fixed old A42 EMA teacher, not completed fresh A43 B', [D/'completed_results/M/arm_M/initialization.json',D/'completed_results/Q/arm_Q/result.json'],
           'Old teacher SHA pinned in protocol/startup; M/Q start digests equal, end digests unchanged; B teacher calls0.')
    capacity_path = D/'startup_evidence/capacity_preflight_result.json'
    cap = read(capacity_path)
    cpu0 = read(D/'startup_evidence/cpu_interface_validation.json')
    assert cap['status']==cpu0['status']=='PASS'
    assert cap['fixed_microbatch']==8 and cap['gradient_accumulation']==1
    assert cap['dense_GT_counts']==[902,512,462,431,370,351,348,346]
    for k in ['zero_output_influence_bitwise_equal','auxiliary_gradient_reaches_stem',
              'teacher_frozen_unchanged','M_Q_actual_teacher_stream_identical',
              'all_capacity_weights_discarded_before_formal_training','no_batch_search_or_fallback']:
        assert cap[k], k
    assert cap['new_student_forwards']==8 and cap['new_teacher_forwards']==4 and cap['optimizer_updates']==5
    assert max(a['peak_allocated_bytes'] for a in cap['arms'].values()) < 22*(1<<30)
    proven('Bounded zero-influence/gradient/frozen-teacher/native-export dense capacity', [capacity_path,D/'startup_evidence/cpu_interface_validation.json'],
           'One fixed densebatch8, no search/fallback; 8student/4teacher calls and5capacity updates discarded. Zero output effect and gradient into stem checked; peak<22GiB.')
    proven('Sequential each exactly12epochs, fixed12EMA, no resume/HPO rescue', [D/'terminal_summary/three_arm_result.json',D/'terminal_summary/supervisor_result.json',D/'completed_results/Q/arm_Q/epochs.jsonl'],
           'All three12/9708 updates/exit0; source and optimizer-state checks agree. No extra epoch or automatic new arm in frozen queue.')
    assert controls['formal_M_Q_teacher_pairing_verified']
    assert len(controls['epoch_pairing'])==12
    for flow in controls['epoch_pairing']:
        assert flow['sampler_order_same'] and flow['first_eight_actual_input_batches_same']
        assert flow['all_batch_teacher_predictions_assignments_and_both_targets_same_M_Q']
    # Directly compare recorded full streams; not only the verifier boolean.
    for e in range(1,13):
        flows={a:read(D/'completed_results'/a/('arm_'+a)/('epoch%02d_input_flow.json'%e)) for a in ['B','M','Q']}
        for a in ['B','M','Q']:
            assert flows[a]['epoch']==e and flows[a]['optimizer_updates']==e*809
            assert flows[a]['order_sha256']==flows['B']['order_sha256']
            assert flows[a]['first_eight_actual_input_sha256']==flows['B']['first_eight_actual_input_sha256']
        for k in ['teacher_all_batches_flow_sha256','first_eight_teacher_batch_sha256',
                  'teacher_matched_instances','teacher_unmatched_GT_due_to_query_limit']:
            assert flows['M'][k]==flows['Q'][k]
    proven('Same input records and actual teacher targets/assignments all12epochs', [V/'B_M_Q_controls_verified.json',D/'completed_results/Q/arm_Q/epoch12_input_flow.json'],
           'Every sampler order and first8 actual input batches; full rolling per-batch teacher outputs, assignments and both targets M/Q. Not every input pixel byte-compared.')
    proven('Finite training and strict native raw/EMA inference export', [V/(a+'_epoch12_checkpoint_cpu_verification.json') for a in ['B','M','Q']]+[V/'B_M_Q_controls_verified.json'],
           'Every saved epoch finite;216 sampled progress rows per arm plus frozen every-step fail-fast; every parameter finite in saved raw/EMA. Native9803112keys/aux removed; not a full archive of every intermediate gradient.')
    proven('Download/check checkpoint, protocol, source, prediction, GT, matrices and logs', [D/('completed_arm_'+a+'_download_verification.json') for a in ['B','M','Q']],
           'Archive SHA/bytes and every one of384 extracted member files rehashed; all91frozen inputs checked per arm.')
    proven('Independent AP/APs/APs75 matrix and raw GT/ignore reconstruction', [V/(a+'_epoch12_independent_metrics.json') for a in ['B','M','Q']]+[V/(a+'_epoch12_independent_GT_ignore.json') for a in ['B','M','Q']],
           'Standalone NumPy no training evaluator/torch/pycocotools:40400precision+400recall identical each;548rawannotations and all saved retained predictions checked. No regeneration of excluded predictions; COCO-style, not official VisDrone.')

    m = gate['metrics']
    qb = m['Q']['APs']-m['B']['APs']; qm = m['Q']['APs']-m['M']['APs']; ap = m['Q']['AP']-m['B']['AP']
    actual_gates = {'Q_minus_B_APs_at_least_0_5pp':qb>=.5,'Q_minus_M_APs_at_least_0_3pp':qm>=.3,'overall_AP_drop_no_more_than_0_2pp':ap>=-.2}
    assert actual_gates==gate['practical_gates']
    assert actual_gates=={'Q_minus_B_APs_at_least_0_5pp':False,'Q_minus_M_APs_at_least_0_3pp':False,'overall_AP_drop_no_more_than_0_2pp':True}
    assert not gate['pilot_pass'] and gate['decision']=='STOP_DIRECT_QUERY_SUPERVISION_CLAIM_NO_RESCUE_SWEEP'
    retired = next(r for r in board['candidates'] if r['id']=='A41-B')
    assert retired['status']=='RETIRED_AFTER_A43_FIXED12_PRACTICAL_GATES_FAILED'
    assert not retired['pilot_recommended'] and not retired['pilot_gate_pass']
    proven('Recompute frozen practical gates and stop failed direct-query claim', [V/'final_practical_gate_verification.json',BASE/'01_候选判别表.json'],
           'Q-B andQ-M APs both below0, so fail+.5/+.3pp gates; overallAP drop-.06455pp passes-.2 gate. No posthoc replacement endpoint or rescue sweep.')

    terminal = read(D/'terminal_summary_download_verification.json')
    proc = read(D/'terminal_summary/final_process_observation.json')
    assert terminal['status']=='PASS' and terminal['member_files_verified']==9
    assert all(p.get('missing') or not p.get('identity_matches') for p in proc['processes'].values())
    assert proc['nvidia_compute_processes']['exit_code']==0 and not proc['nvidia_compute_processes']['stdout'].strip()
    for a in ['B','M','Q']:
        assert (D/'terminal_summary'/('independent_checkpoint_'+a+'.json')).read_bytes()==(V/(a+'_epoch12_checkpoint_cpu_verification.json')).read_bytes()
    proven('True process termination and no queued extra direction', [D/'terminal_summary/final_process_observation.json',D/'terminal_summary/supervisor_result.json',D/'terminal_summary/queue_progress.json'],
           'Original PID/start-tick identities absent at collection; all exits0, compute process list empty. Frozen queue onlyB/M/Q; A44 allocation0.')
    counts = event['counts']
    assert counts['A43_formal_updates']==29124 and counts['A43_formal_student_forwards']==30477
    assert counts['A43_formal_teacher_forwards']==19416
    assert counts['A43_full_forwards_with_capacity']==49905 and counts['A43_updates_with_capacity']==29129
    assert counts['architecture_cumulative_full_forwards']==state['model_forwards']==83915
    assert counts['architecture_cumulative_updates_with_capacity']==state['training_updates']==58793
    assert counts['architecture_cumulative_formal_updates']==58774
    assert counts['architecture_cumulative_capacity_updates']==19
    assert counts['architecture_cumulative_epochs_sum_across_arms']==72
    assert state['cap']==49905 and state['counts_terminal'] and state['full_forward_cap_closed']
    proven('Close role-specific counters and per-arm budgets', [D/'final_report_evidence.json',D/'canonical_before_A43/15_当前状态与结果校验.json',BASE/'15_当前状态与结果校验.json'],
           'Previous counter snapshot plus actual formal/capacity roles; current49905full calls, cumulative83915;58793updates includes19capacity. Epoch72is across arms; no arm>12. CPU/NumPy/source work0new NN calls.')
    assert len(stats['checks'])==11 and stats['coverage']=='11/11 checked'
    assert not stats['significance_claim'] and stats['p_values_computed']==stats['confidence_intervals_computed']==0
    assert stats['one_training_seed'] and not stats['primary_endpoint_replaced']
    proven('Distinguish result/inference/hypothesis and statistical limits11/11', [V/'statistical_interpretation_11_check.json',V/'per_class_descriptive_scope.json',BASE/'104_A43三组终点去留与下一候选.txt'],
           'CAUTION: single seed12epochs, reused public validation, class heterogeneity only. Negative resource gate is not universal inefficacy, significance, semantic mechanism or novelty proof.')

    proposal = read(N/'A44_candidate_proposal.json')
    source = read(N/'source_read_scope_and_limits.json')
    add = read(N/'core_literature_matrix_addendum.json')
    assert proposal['training_allocation_now']==0 and not proposal['novelty_cleared'] and not proposal['mainline_validated']
    assert proposal['bounded_review']['search_query_count']==len(proposal['bounded_review']['queries'])==7
    assert proposal['bounded_review']['selected_method_papers']==3
    assert len(proposal['closest_comparison'])==3 and len(proposal['major_risks'])>=5
    assert proposal['minimal_validation_plan']['not_launched']
    assert proposal['draft_operator']['N0'] and proposal['draft_operator']['N1'] and proposal['draft_operator']['N2']
    assert source['native_level_ownership_is_fixed'] and source['native_offsets_and_attention_are_query_adaptive']
    assert not source['AdaMixer_immutable_blob_verified'] and not source['A44_training_started']
    assert len(add['rows'])==2 and add['existing_reference_reexamined']['id']=='BetterSampling'
    for value in source['native_files_sha256'].values():
        assert len(value)==64
    assert len(read(BASE/'14_架构核心文献矩阵.json')['rows'])==117
    for filename in ['search_01_raw.json','search_02_raw.json','primary_01_raw.json',
                     'primary_02_raw.json','primary_03_raw.json','primary_04_raw.json','primary_05_raw.json']:
        assert (N/filename).stat().st_size>0
    proven('Bounded primary literature/source follow-up and concrete affordable next candidate', [N/'A44_candidate_proposal.json',N/'A44_候选问题与最小验证路线.txt',N/'core_literature_matrix_addendum.json',N/'source_read_scope_and_limits.json'],
           'Seven queries/three selected methods plus native and author sampler source scopes. Explicit N0/N1/N2 operator, comparators, risks, no train,12epoch plan and stop rules. Generic dynamic-scale novelty excluded; no uniqueness claim; author immutable-source pin unavailable disclosed.')
    assert board['current_priority_candidate']==proposal['candidate_id']
    assert not board['mainline_validated'] and not state['mainline_validated']
    assert board['A44_proposal']['training_allocation']==0 and not event['other_direction_training_queued']
    assert not event['rescue_sweep_started'] and not event['A22_restarted'] and not event['old_DEIM_restarted'] and not event['compression_branch_reopened']
    proven('Canonical report/state and no false validated-mainline claim', [BASE/'01_候选判别表.json',BASE/'15_当前状态与结果校验.json',BASE/'104_A43三组终点去留与下一候选.txt'],
           'A43 retired; A44 untrained recommendation only. Local fulfilment status is scoped; app tool completion follows this audit. Historical receipts remain immutable.')

    report = BASE/'104_A43三组终点去留与下一候选.txt'
    text = report.read_text(encoding='utf-8')
    for a in ['B','M','Q']:
        for k in ['AP','APs','APs75']:
            assert format(m[a][k],'.10f') in text, (a,k)
    assert '11/11' in text and '非VisDrone官方榜单' in text and '新颖性风险高' in text
    evidence_files += [BASE/'01_候选判别表.json',BASE/'15_当前状态与结果校验.json',report,
                      D/'final_report_evidence.json',D/'B_terminal_state_historical_remap.json',
                      D/'state_before_A43_terminal/01_候选判别表.json',D/'state_before_A43_terminal/15_当前状态与结果校验.json',
                      D/'startup_evidence/execution_protocol_FROZEN.json',capacity_path,D/'startup_evidence/cpu_interface_validation.json',
                      D/'terminal_summary_bundle.zip',D/'terminal_summary_download_verification.json',
                      V/'final_practical_gate_verification.json',V/'B_M_Q_controls_verified.json',
                      V/'per_class_descriptive_scope.json',V/'statistical_interpretation_11_check.json',
                      D/'finalize_A43_report.py',Path(__file__)]
    evidence_files += history_receipts
    evidence_files += list(N.glob('*.json'))+list(N.glob('*.txt'))
    files = {str(p):sha(p) for p in sorted(set(evidence_files))}
    result = {'phase':'A43','event':'FULL_BOUNDED_OBJECTIVE_COMPLETION_AUDIT',
              'recorded_utc':datetime.now(timezone.utc).isoformat(),
              'status':'PASS','objective_requirements_met':True,'goal_tool_action_ready':'complete',
              'goal_completion_scope':event['goal_completion_scope'],
              'mainline_validated':False,'pilot_pass':False,'decision':gate['decision'],
              'metrics':m,'deltas_vs_B_pp':gate['deltas_vs_B_pp'],'Q_minus_M_pp':gate['Q_minus_M_pp'],
              'practical_gates':actual_gates,'counts':counts,'archive_members_each':128,
              'frozen_inputs_rehashed_each':all_hash_count,'terminal_summary_members':9,
              'requirements':requirements,'requirements_count':len(requirements),
              'historical_signed_input_resolution':history_resolution,
              'new_model_forwards_by_postprocessing':0,'new_optimizer_updates_by_postprocessing':0,
              'next_candidate':proposal['candidate_id'],'next_candidate_training_allocation':0,
              'files_sha256':files,
              'final_tool_receipt_outside_this_hashset':str(D/'goal_completion_tool_receipt.json')}
    out = BASE/'105_A43终点与目标完成审计回执.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','requirements':len(requirements),'hashed_files':len(files),
                      'frozen_files_each':all_hash_count,'next_training':0,'ready_for_goal_complete':True,
                      'receipt':str(out)},ensure_ascii=False))


if __name__=='__main__':
    main()

