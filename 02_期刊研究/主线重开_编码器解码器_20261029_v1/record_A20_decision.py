"""Audit actual A20 capacity evidence, reconcile all updates, retain active research goal."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil,sys,zipfile
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parent;O=R/'A20_VSSD宿主严格装载与容量预检_v1';D=O/'verified_remote_evidence'
C=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
REPORT='50_A20新编码器宿主实测与主线验证边界.txt';RECEIPT=R/'51_A20严格装载与容量实测核验回执.json'
def rd(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wr(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    assert not RECEIPT.exists(),'Already recorded: inspect, never rerun'
    utc=datetime.now(timezone.utc).isoformat();snap=R/'history_before_A20_freeze_v1'
    old=rd(R/'49_A19方法与宿主资产核验回执.json')
    for row in old['files']:
        p=snap/row['path'] if row['path'] in C else R/row['path']
        assert sha(p)==row['sha256'],row['path']
    expected=rd(O/'expected_remote_evidence.json');archive=O/'capacity_evidence.zip'
    assert archive.stat().st_size==expected['bytes'] and sha(archive)==expected['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(expected['members'])
        for n in z.namelist():assert z.read(n)==(D/n).read_bytes(),n
    terminal=rd(D/'terminal_runtime.json')
    for n,h in terminal['file_hashes'].items():assert sha(D/n if (D/n).exists() else O/n)==h,n
    assert terminal['compute_apps']==''
    p=rd(O/'capacity_protocol_FROZEN.json');strict=rd(O/'strict_host_load.json');result=rd(D/'capacity_result.json')
    assert (D/'capacity_protocol_FROZEN.json').read_bytes()==(O/'capacity_protocol_FROZEN.json').read_bytes()
    assert (D/'strict_host_load.json').read_bytes()==(O/'strict_host_load.json').read_bytes()
    for row in p['files']:assert sha(O/row['path'])==row['sha256'],row['path']
    assert result['protocol_sha256']==sha(O/'capacity_protocol_FROZEN.json')
    assert result['status']=='BOUNDED_CAPACITY_PASS_3_FORWARDS_2_DISPOSABLE_UPDATES'
    assert result['complete_model_forwards']==3 and result['capacity_only_optimizer_updates']==2 and result['formal_training_epochs']==0
    assert result['host_state_digest_before_GPU']==strict['host_state_digest']==p['expected_host_state_digest']
    assert strict['classification_state_keys']==347 and strict['backbone_shared_keys_equal_ema']==343 and strict['backbone_state_keys']==351
    assert strict['classification_strict_load'] and strict['mapped_backbone_strict_load'] and not strict['CUDA_initialized']
    events=[json.loads(s) for s in (D/'capacity_events.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    calls=[x for x in events if x['event']=='COMPLETE_FORWARD_ENTER'];updates=[x for x in events if x['event']=='DISPOSABLE_OPTIMIZER_UPDATE']
    assert [x['call_index'] for x in calls]==[1,2,3] and [x['update_index'] for x in updates]==[1,2]
    assert [x['stage'] for x in calls]==['eval_fp32_b1','train_amp_b1','train_amp_b2']
    assert events[-1]['event']=='CAPACITY_TERMINAL'
    assert [x['batch'] for x in result['stages']]==[1,1,2]
    for s in result['stages'][1:]:
        assert s['raw_loss_sanitizer_guard_count']==45 and s['scaler_scale_after']==128.0
        assert all(g['finite']>0 and g['nonzero']>0 for g in s['gradient_groups'].values())
        assert sum(g['finite'] for g in s['gradient_groups'].values())==762
        assert s['optimizer_state_numel']==2*(strict['host_parameters']-2)+762
    ann=rd(O/'new_host_annotation_resource_result.json')
    assert ann['new_model_forwards']==0 and ann['model_predictions_read'] is False
    assert ann['old_Dome_exclusion_or_gate_changed'] is False and ann['counts']['eligible_prefixes']==33
    assert sha(O/'new_host_annotation_resource_rules.json')==ann['rules_sha256']
    assert sha(O/'new_host_annotation_resource_preflight.py')==ann['script_sha256']
    assert sha(R/'a18_neighbour_annotation_preflight.py')==ann['unchanged_geometry_helper_source_sha256']
    assert rd(O/'next_scientific_protocol_DRAFT.json')['new_forward_budget_allocated']==0
    ops=R.parent/'前期调研_码率受限航拍小目标_20261028_v1/远端验证_20261029_v1/ssh_operation_receipts.jsonl'
    operations=[json.loads(s) for s in ops.read_text(encoding='utf-8').splitlines() if s.strip()]
    operations=[x for x in operations if str(x.get('id','')).startswith('A20_')]
    run=[x for x in operations if x['id']=='A20_execute_frozen_capacity_3_forward_max_2_disposable_update_max']
    assert len(run)==1 and run[0]['exit_code']==0
    assert json.loads(run[0]['stdout'])['result']==result
    wr(O/'remote_operation_receipts.json',{'operations':operations,'scope':'A20 isolated construction/assets/capacity only'})
    docs={n:rd(R/n) for n in C};board,search,matrix,state,source=[docs[n] for n in C]
    assert len(board['candidates'])==14 and len(matrix['rows'])==76
    assert state['model_forwards']==252 and state['cap']==255 and state['training_updates']==0
    priority='共享状态邻近实例干扰仍为未验证候选。新VSSD宿主严格装载及batch1/2容量已通过；先固定原生状态算子、网格/感受野控制、训练充分性和独立结局协议，再按有界预算建立诊断基线。当前无新模块或漏检因果证据。'
    ev={'utc':utc,'report':str(R/REPORT),'new_complete_model_forwards':3,'new_suffix_forwards':0,
        'actual_detector_optimizer_updates':2,'capacity_only_optimizer_updates':2,'formal_training_epochs':0,
        'new_diagnostic_fits':0,'core_literature_rows':76,'source_blobs_verified':8,
        'host_parameters':strict['host_parameters'],'strict_classification_and_backbone_mapping':True,
        'capacity_pass':True,'capacity_protocol_sha256':result['protocol_sha256'],
        'batch2_allocated_peak_MiB':result['stages'][2]['allocated_peak_MiB'],
        'batch2_reserved_peak_MiB':result['stages'][2]['reserved_peak_MiB'],
        'new_host_annotation_prefix_resource':33,'native_state_intervention_eligibility_verified':False,
        'failure_cause_verified':False,'distinct_algorithm_verified':False,'mainline_validated':False}
    for c in board['candidates']:
        if c['id']=='N_NONCAUSAL_STATE_INSTANCE_INTERFERENCE':
            c.update(status='NATIVE_HOST_STRICT_LOAD_AND_CAPACITY_PASS_MECHANISM_UNTESTED',
              A20_evidence=ev,next='Fix native state intervention/grid/RF/sham controls and baseline learning adequacy before trained-host diagnostic; no module training approved by a capacity pass.')
    board.update(A20_evidence=ev,latest_update_utc=utc,latest_report=str(R/REPORT),current_priority_question=priority,mainline_validated=False)
    search['A20_source_runtime_asset_verification']={**ev,'new_literature_search_queries':0,
       'author_source_and_PyPI_metadata':'See task bundle_manifest.json and *_pypi.json;8 sourceblobs verified,5 newly downloaded files',
       'old_method_read_scopes_unchanged':True}
    search['supplement_utc']=utc
    v=[x for x in matrix['rows'] if x['id']=='VSSD'];assert len(v)==1
    v[0]['A20_our_runtime_evidence']={**ev,'scope':'classification EMA strict load plus new diagnostic host capacity; not author detector/AP reproduction'}
    matrix.update(updated_utc=utc,latest_report=str(R/REPORT),latest_empirical_boundary='A20 specified diagnostic host3 calls/2 temporary updates; no trained detection outcome, no failure mechanism')
    source['A20_native_host_capacity']={**ev,'directory':str(O),'evidence_archive_sha256':expected['sha256'],
       'constructor_and_preprocessing':'Explicit classifier mapping/ImageNet backbone-only normalization; original native author arithmetic retained',
       'capacity_run_terminal':True,'GPU_compute_processes_after_run':[]}
    source['next_instance_ownership_question']={'question':priority,'protocol':str(O/'next_scientific_protocol_DRAFT.json'),
       'status':'DRAFT_NOT_FROZEN_NOT_EXECUTED','new_forward_budget_allocated':0,'new_user_permission_required':False}
    source['updated_utc']=utc
    hist=state.setdefault('authoritative_phase_history',[]);hist.append(state['latest_authoritative_phase'])
    state.update(recorded_utc=utc,model_forwards=255,cap=255,training_updates=2,
       formal_detector_training_updates=0,capacity_only_optimizer_updates=2,
       detector_optimizer_updates_including_capacity=2,formal_training_epochs=0,
       training_update_scope='Actual architecture detector optimizer updates2, both A20 disposable capacity-only; formal training updates0. Historical64 diagnostic linear fits/96 scalar heads separate; compression branch outside this ledger.',
       cap_scope='255complete detector calls throughA20:252historical+3capacity;255capCLOSED.72A15suffix calls separate. No automatic next phase.',
       current_goal_turn='progress_A20_strict_native_encoder_host_capacity_and_blinded_annotation_resource_verified',
       latest_report=str(R/REPORT),current_priority_question=priority,goal_status='ACTIVE',mainline_validated=False)
    state['latest_authoritative_phase']={**ev,'phase':'A20','classification':'progress','goal_status':'active',
       'status':'BOUNDED_CAPACITY_COMPLETE_SCIENTIFIC_MAINLINE_UNVALIDATED','complete_mainline':False,
       'full_model_forwards':255,'cumulative_authorized_protocol_cap':255,'full_forward_cap_closed':True,
       'historical_suffix_forwards':72,'bounded_phase_complete':True,'next':priority}
    presult=R/'history_before_A20_result_update_v1';presult.mkdir(exist_ok=False)
    for n in C:shutil.copy2(R/n,presult/n)
    for n,x in docs.items():wr(R/n,x)
    task_names=['a20_host.py','a20_capacity.py','prepare_bundle.py','prepare_capacity_inputs.py','freeze_capacity_protocol.py',
       'bundle_manifest.json','bundle_download_upload_manifest.json','cpu_preparation_bundle.zip',
       'strict_host_load.json','capacity_inputs.zip','capacity_input_package_receipt.json','capacity_inputs/input_manifest.json',
       'capacity_protocol_FROZEN.json','capacity_run_control.zip','capacity_run_control_receipt.json',
       'expected_remote_evidence.json','capacity_evidence.zip','remote_operation_receipts.json',
       'new_host_annotation_resource_rules.json','new_host_annotation_resource_preflight.py','new_host_annotation_resource_result.json',
       'new_host_annotation_inputs.json','new_host_per_image_resource.json','new_host_eligible_image_pairs.json',
       'new_host_prefix_candidates.json','next_scientific_protocol_DRAFT.json','timm_pypi.json','einops_pypi.json','yacs_pypi.json']
    names=C+[REPORT,'record_A20_decision.py']+[str((O/n).relative_to(R)).replace('\\','/') for n in task_names]
    names += [str(p.relative_to(R)).replace('\\','/') for p in sorted(D.iterdir()) if p.is_file()]
    receipt={'utc':utc,'status':'A20_STRICT_NATIVE_HOST_CAPACITY_AND_ANNOTATION_RESOURCE_VERIFIED_GOAL_ACTIVE',
       **ev,'goal_status':'ACTIVE','cumulative_complete_model_forwards':255,'complete_forward_cap':255,
       'complete_forward_cap_closed':True,'actual_cumulative_detector_optimizer_updates':2,
       'formal_detector_training_updates':0,'capacity_weights_saved':False,'GPU_processes_after_run':[],
       'historical_suffix_forwards':72,'historical_CPU_diagnostic_fits':64,'candidate_rows':14,
       'evidence_archive_sha256':expected['sha256'],'evidence_archive_members':11,
       'prior_A19_hashes_verified':len(old['files']),'prior_canonical_snapshot':str(snap),
       'files':[{'path':n,'sha256':sha(R/n),'bytes':(R/n).stat().st_size} for n in names],
       'next':'Native-state intervention and baseline-learning protocol; no new protocol/forward/training allocation yet',
       'limits':'Capacity and annotation resource only; no AP, stable throughput, training duration, confirmed mechanism or distinct innovation'}
    wr(RECEIPT,receipt)
    print(json.dumps({k:receipt[k] for k in ['status','goal_status','mainline_validated','cumulative_complete_model_forwards',
      'actual_cumulative_detector_optimizer_updates','formal_detector_training_updates','capacity_pass','new_host_annotation_prefix_resource']},ensure_ascii=False))
if __name__=='__main__':main()
