import datetime,hashlib,json,pathlib,shutil
D=pathlib.Path(__file__).resolve().parent;B=D.parent
REPORT=B/'92_A42三组最小验证_已启动与固定判据.txt'
RECEIPT=B/'93_A42三组启动与单卡实测核验回执.json'
def rd(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
def wr(p,x):pathlib.Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb')as f:
        for c in iter(lambda:f.read(1<<20),b''):h.update(c)
    return h.hexdigest()
assert not RECEIPT.exists()
log=B.parent/'前期调研_码率受限航拍小目标_20261028_v1/远端验证_20261029_v1/ssh_operation_receipts.jsonl'
rows=[]
for line in log.read_text(encoding='utf-8').splitlines():
    try:x=json.loads(line)
    except json.JSONDecodeError:continue
    if x.get('id','').startswith('A42_')and x.get('action'):rows.append(x)
wr(D/'ssh_selected_receipts.json',{'rows':rows})
observations=[r for r in rows if r['id']=='A42_live_observation_before_local_publish_v1']
assert len(observations)==1
obs=json.loads(observations[0]['stdout'])
assert obs['worker_identity_matches_live_proc']and obs['queue_status']=='RUNNING'and not obs['stderr_tail']
wr(D/'live_observation_before_publish.json',obs)
evidence=D/'startup_evidence'
plan=rd(evidence/'execution_protocol_FROZEN.json')
cap=rd(evidence/'capacity_preflight_result.json');batch=rd(evidence/'batch_selection_result.json')
assert sha(D/'common_init_10class.pth')==plan['common_init_sha256']
for f in ['runtime_common.py','detail_reader.py','evaluate.py','train_arm.py','supervise_three_arms.py']:
    assert sha(D/f)==plan['files_sha256'][f]

old=rd(B/'91_A41暂定主线选择与证据核验回执.json')
for p,h in old['files_sha256'].items():assert sha(p)==h,p
names=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
snap=D/'canonical_before_A42';snap.mkdir(exist_ok=False);remap={}
for name in names:
    target=snap/name;shutil.copy2(B/name,target);remap[str(B/name)]=str(target)
wr(D/'historical_A41_canonical_remap.json',{'scope':'A41 receipt untouched. Five evolving canonical files archived here to retain historical hashes.','map':remap})

now=datetime.datetime.now(datetime.timezone.utc).isoformat();progress=obs['progress']
n=progress['optimizer_updates'];f=progress['full_model_forwards']
phase={'phase':'A42','recorded_utc':now,'status':'THREE_ARM_PILOT_RUNNING_ORIGINAL_ARM','current_user_task_status':'RUNNING','previous_research_selection_goal_status':'complete','mainline_selected_for_pilot':True,'mainline_validated':False,'empirical_algorithm_improvement_established':False,'comparative_AP_available':False,'recommended_candidate':'A41-A','report':str(REPORT),'receipt':str(RECEIPT),'execution_protocol':str(evidence/'execution_protocol_FROZEN.json'),'remote_root':'/root/c10_region_restore_20261029_v1/architecture_probe_v1/a42_ecdet_three_arm_20261031_v1','source_pin':'d8752b75c9beaecfcdd0e5cd207bae3b6946ae23','active_arm':obs['active_arm'],'arms_training_started':['B'],'arms_queued':['S','L'],'microbatch':8,'gradient_accumulation':1,'epochs_max_each':12,'seed':20261031,'capacity_check':'PASS','capacity_peak_allocated_bytes':[v['peak_allocated_bytes']for v in batch['rows']],'zero_residual_output_bitwise_equal':True,'L_raw_class_head_input_invariant_checked':True,'extra_parameters':120192,'new_capacity_only_optimizer_updates':12,'new_capacity_full_forwards':138,'observed_new_formal_updates_lower_bound':n,'observed_new_formal_full_forwards_lower_bound':f,'observed_completed_epochs_B':progress['completed_epochs'],'full_model_forwards_cumulative_lower_bound':3395+138+f,'detector_updates_including_capacity_cumulative_lower_bound':528+12+n,'all_counters_scope':'A42 live lower bounds at saved observation, not terminal totals; original D0/DA outside architecture counter scope.','observation':str(D/'live_observation_before_publish.json'),'source_host_is_compact_ViT':True,'CPU_probe_fits_new':0,'new_core_papers':0,'A22_restarted':False,'old_DEIM_restarted':False,'compression_branch_reopened':False,'automatic_new_direction_training':False,'failure_action':'Stop localization-only claim if fixed twelve-epoch practical comparison fails; no module/layer/threshold rescue sweep. Stop remaining queue on execution error; no automatic retry.','next':'Finish sequential B/S/L and fixed epoch12 EMA comparison; server supervisor writes three_arm_result.json and Chinese summary.','tool_goal_scope':'A41 selection goal remains complete; this is the user explicitly requested experimental follow-up task, not an inferred new goal.'}

board=rd(B/names[0]);board.update(status='A42_PROVISIONAL_MAINLINE_IN_BOUNDED_PILOT',latest_update_utc=now,latest_report=str(REPORT))
board['A42_evidence']=phase;board['pilot_validation_status']='RUNNING_B_S_L_QUEUED';board['remaining_deliverable']=phase['next']
board['next_decision_protocol']={'status':'FROZEN_RUNNING','file':phase['execution_protocol']}
board['bounded_diagnostic_protocol']=phase['execution_protocol']
for c in board['candidates']:
    if c['id']=='A41-A':c.update(status='PILOT_RUNNING_ORIGINAL_ARM',next=phase['execution_protocol'],efficacy_validated=False)
wr(B/names[0],board)

state=rd(B/names[3]);state['A42_evidence']=phase
state.update(recorded_utc=now,latest_report=str(REPORT),latest_authoritative_phase=phase,current_user_task_phase=phase,current_user_task_status='RUNNING_EXPERIMENTS',mainline_validated=False,model_forwards=3395+138+f,training_updates=528+12+n,formal_detector_training_updates=526+n,capacity_only_optimizer_updates=14,detector_optimizer_updates_including_capacity=528+12+n,formal_training_epochs=progress['completed_epochs'],current_native_job_terminal=False,counts_terminal=False,full_forward_cap_closed=False,counters_observed_lower_bounds_while_job_live=True)
state['cap']=30615;state['cap_scope']='A42 only: at most 3*(12*809 + 3*137 + 40) formal full forwards plus138 capacity calls. All older phase caps remain closed.'
state['goal_status_scope']='A41 research-selection goal complete; new user-authorized experiment task currently running.'
state['training_update_scope']='Cumulative detector optimizer updates include historical and capacity updates; live A42 count is a saved observed lower bound.'
wr(B/names[3],state)

audit=rd(B/names[4]);audit['A42_native_baseline_and_detail_branch_execution']={'source_pin':phase['source_pin'],'official_80class_strict_load':str(evidence/'strict_official_80class_load.json'),'isolated_patch':str(evidence/'isolated_source_patch_receipt.json'),'compatibility':'future annotation on ecvit for Python3.9; optional unused TensorBoard import; raw-loss fail-fast guard before native nan_to_num','formulas_in_finite_B_case':'unchanged','zero_reader_control':'bitwise equality checked on inspected input','same_native_losses':True,'same_FDR_and_LQE_coupling':True,'native_published_recipe_reproduction':False,'working_code_remote_hash_matches_local':5,'training_protocol':phase['execution_protocol'],'report':str(REPORT)}
wr(B/names[4],audit)

for p,h in old['files_sha256'].items():assert sha(remap.get(p,p))==h,p
wr(D/'startup_state.json',phase)
files=[p for p in D.iterdir()if p.is_file()]+[p for p in evidence.rglob('*')if p.is_file()]+[p for p in snap.iterdir()if p.is_file()]+[B/n for n in names]+[REPORT]
receipt=dict(phase)
receipt.update(previous_A41_evidence_files_verified=len(old['files_sha256']),historical_canonical_files_remapped=5,downloaded_startup_members_verified=17,common_initialization_backup_sha256=sha(D/'common_init_10class.pth'),files_sha256={str(p):sha(p)for p in files})
wr(RECEIPT,receipt)
for p,h in receipt['files_sha256'].items():assert sha(p)==h,p
print(json.dumps({'status':'PASS','current_task':'RUNNING_EXPERIMENTS','B_completed_epochs':progress['completed_epochs'],'B_updates_observed':n,'new_capacity_updates':12,'new_capacity_forwards':138,'hashes':len(receipt['files_sha256']),'comparative_AP_available':False}))
