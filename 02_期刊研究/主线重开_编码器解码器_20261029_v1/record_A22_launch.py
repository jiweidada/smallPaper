"""Verify an actually live fixed12 pilot and record partial counters without claiming completion."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,sys,zipfile
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parent;O=R/'A22_十二轮诊断基线执行_v1';D=O/'verified_startup_evidence'
C=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
REPORT='54_A22十二轮诊断基线启动与执行状态.txt';RECEIPT=R/'55_A22训练启动与首批更新核验回执.json'
def rd(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wr(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    assert not RECEIPT.exists(),'Already recorded; inspect current job, never relaunch'
    utc=datetime.now(timezone.utc).isoformat();snap=R/'history_before_A22_freeze_v1'
    previous=rd(R/'53_A21共享状态路径与研究设计核验回执.json')
    for row in previous['files']:assert sha((snap if row['path'] in C else R)/row['path'])==row['sha256'],row['path']
    manifest=rd(O/'execution_manifest_FROZEN.json')
    assert manifest['new_training_complete_forward_cap']==38832 and manifest['new_optimizer_attempt_cap']==9708
    for row in manifest['files']:assert sha(O/row['path'])==row['sha256'],row['path']
    for row in manifest['dependencies']:assert sha(Path(row['local_source']))==row['sha256'],row['local_source']
    assert manifest['evaluation_complete_forwards_allocated']==0
    e=rd(O/'expected_startup_evidence.json');p=O/'startup_evidence.zip'
    assert sha(p)==e['sha256'] and p.stat().st_size==e['bytes']
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None and set(z.namelist())==set(e['members'])
        for n in z.namelist():assert z.read(n)==(D/n).read_bytes(),n
    assert (D/'execution_manifest_FROZEN.json').read_bytes()==(O/'execution_manifest_FROZEN.json').read_bytes()
    for n,h in rd(D/'live_execution_source_hashes.json').items():assert sha(O/n)==h,n
    assert rd(D/'training_initial_host_receipt.json')['host_state_digest']==manifest['initial_state_digest']
    assert rd(D/'training_start_data_integrity.json')['aggregate_sha256']==rd(O/'train_asset_manifest.json')['aggregate_sha256']
    steps=[json.loads(s) for s in (D/'first8_optimizer_step_events.jsonl').read_text().splitlines()]
    assert [x['actual_updates'] for x in steps if x['event']=='OPTIMIZER_STEP_COMPLETE']==list(range(1,9))
    changes=rd(D/'initial_to_first_update_parameter_changes.json');assert all(changes['groups_changed'].values())
    source_ops=R.parent/'前期调研_码率受限航拍小目标_20261028_v1/远端验证_20261029_v1/ssh_operation_receipts.jsonl'
    operations=[json.loads(s) for s in source_ops.read_text(encoding='utf-8').splitlines() if s.strip()]
    operations=[x for x in operations if str(x.get('id','')).startswith('A22_')]
    launch=next(x for x in operations if x['id']=='A22_verify_allocation_and_launch_exactly_one_pilot12_supervisor')
    assert launch['exit_code']==0
    launch= json.loads(launch['stdout'])['launch']
    assert launch['manifest_sha256']==sha(O/'execution_manifest_FROZEN.json')
    envop=next(x for x in operations if x['id']=='A22_verify_live_training_assertion_and_runtime_environment')
    assert envop['exit_code']==0;env= json.loads(envop['stdout']);assert env['assertions_not_disabled_by_environment']
    wr(O/'live_assertion_environment_check.json',env)
    wr(O/'remote_operation_receipts.json',{'operations':operations,'scope':'A22 source/asset/preflight/launch/live reads, no post-training evaluation'})
    live=rd(O/'readonly_snapshot_before_startup_record.json');progress=live['progress']
    assert live['supervisor']['live'] and live['supervisor']['identity_matches'] and live['child']['live'] and live['child']['identity_matches']
    assert live['supervisor']['pid']==launch['pid']==13761 and live['supervisor']['start_ticks']==launch['start_ticks']==99433280
    assert live['child']['pid']==13762 and live['child']['start_ticks']==99433288
    assert live['training_result'] is None and live['supervisor_result'] is None and progress['status']=='RUNNING'
    assert progress['validation_calls']==0 and progress['actual_updates']>=8
    forward_observed=max(live['latest_forward_entry']['index'],progress['forward_entries'],
       live['latest_optimizer_event'].get('forward_entries',0),live['latest_training_event'].get('forward_entries',0))
    updates_observed=max(progress['actual_updates'],live['latest_optimizer_event'].get('actual_updates',0),
       live['latest_training_event'].get('actual_updates',0))
    assert forward_observed<=38832 and updates_observed<=9708
    counts={'snapshot_window_utc':live['utc'],'counter_scope':'Observed lower bounds from separate live streams, not terminal counts',
       'new_training_forward_entries_observed':forward_observed,'new_training_completed_forwards_progress':progress['forward_completed'],
       'new_formal_updates_observed':updates_observed,'new_optimizer_attempts_progress':progress['optimizer_attempts'],
       'AMP_skips_progress':progress['AMP_skips'],'epochs_completed':progress['epochs_completed'],'current_epoch':progress['epoch'],
       'cumulative_forward_entries_observed':260+forward_observed,'cumulative_completed_updates_observed':2+updates_observed}
    wr(O/'verified_running_counter_snapshot.json',counts)
    report=f'''固定12轮诊断基线已真实启动，当前仍在运行。主线仍为待检验候选，尚无训练后验证集AP、共享状态有害交互结论或新算法增量。启动与训练进度不代替选题成立。

1. 核查策略与检索记录
本轮没有新增文献检索，核心矩阵保持77条、候选14个。继续落实A21已预先约定的问题、单一P2状态位置、27主/6敏感性样本、11组对照与停止门槛。新增核对原生coco_dataset.py和postprocessor.py的[W,H]坐标语义，检查precision tensor的AP@500实现。最近方法和完整阅读缺口保持原记录，未凭代码运行排清创新性。

2. 数据与原始资产
训练使用官方VisDrone2019-DET-train全部6471张原图及原始标注，有效GT343204。远端启动时再次逐份核对本地SHA256，聚合hash={rd(O/'train_asset_manifest.json')['aggregate_sha256']}。没有使用旧YOLO labels.cache或压缩分支子集。
548张val原图与标注按本地原件上传，共38759个有效GT。CPU已核对全部548图/标注hash。val资产ZIP为81536695字节，SHA256=f17872f7ad5dbf636239206aefffdbc414a79997cefea656d1739162b78ee4f2。此时只检查像素/标注与合成统计例子，未对真实val图执行检测器或查看其预测。

3. 算法、初始化及12轮配方
VSSD固定c10de2f8b717a04459b09cbd651de2e768f302c8，Dome接口固定2dde3bc1946a3e9fad9abd0612b59fc39bd6b861。分类EMA严格装载后映射到4路骨干，新颈部与解码器按seed20261030初始化。实际训练初始state_digest={manifest['initial_state_digest']}，与A20/A21一致，未复用两次容量更新的临时权重。
模型总参数38708395，可训练38708393，优化器含762个可训练参数张量/4组。800square、RGB/255、骨干单独ImageNet归一化；训练仅按图/epoch哈希做0.5水平翻转，框cx同步变为1-cx。physical batch2、累积4个微批，末组按实际7张图加权；不声称等价于原生batch8的BN/DN/损失统计。
AdamW foreachFalse，骨干/头学习率2e-5/2e-4，bias/1D不衰减，其余weight_decay0.000065。初始GradScaler128、clip_norm0.1；第1次真实更新用0.001预热因子，第1000次达到1，跳步不推进预热。训练query保持250-500，未来诊断固定500。raw第12轮为固定端点，不选EMA或val最佳点。此显式诊断配方不冒充作者160轮精度复现。
服务器快照CPU配额相当于8核、共享内存16GiB，使用4个spawn加载进程，主OMP/MKL4。单RTX3090。源文件和科学设计在训练前固定，执行manifest SHA256={sha(O/'execution_manifest_FROZEN.json')}。

4. 实现验证及原始证据
CPU检查已通过：严格宿主装载、优化器分组、水平翻转框几何、不同query的一对一配对指标、零效果bootstrap拒绝、拒绝仅因背景更差造成的假增益、状态变化范数控制、AP@500保留第151名真阳性、空检测AP、原生长方形图像[W,H]缩放，以及33个已固定像素处理的逐字节重现。CPU检查0次完整检测器前向、0次优化器更新；原生postprocessor另有1次合成例子调用。
训练、固定端点评估、状态对照及统计代码均已实现并固定。后续实际评估另须绑定固定checkpoint并分配额度；运行中的训练监督器不会自动执行评估。
训练进程PID13762/start_ticks99433288，监督进程PID13761/start_ticks99433280。两个进程的实际/proc身份和命令行已核对；assert未被PYTHONOPTIMIZE环境关闭。NVIDIA可能报告宿主命名空间PID，任务身份以这两个容器内PID/start_ticks/命令行为准。
首个真实更新后，对learned parameters分组哈希，骨干、颈部、解码器全部改变；不是仅BN统计变化。前8次优化器进入/成功返回事件与前12批输入hash已归档，未为该检查增加模型调用。

5. 当前快照及能支持的判断
快照窗口服务器UTC={live['utc']}。进度文件确认第{progress['epoch']}轮、已完成{progress['forward_completed']}个训练微批、{progress['actual_updates']}次真实更新，AMP跳步{progress['AMP_skips']}；该轮共3236微批。独立日志最新模型入口达到{forward_observed}次，不把入口或活跃快照当成全部完成。
最近记录的累积组allocated峰值{progress['allocated_peak_MiB']:.3f}MiB、reserved峰值{progress['reserved_peak_MiB']:.3f}MiB；这是最近组的记录，不是12轮全程峰值。该组原始loss与梯度检查通过，未由不同批次的loss差推断收敛或精度改善。
当前已完成整轮数{progress['epochs_completed']}，验证集模型调用0。计数来自多个活跃日志流，是已观察到的下界；训练会继续，后续以同一进程的新快照更新。主机与服务器墙钟不同步，先后关系依据控制包hash、上传/启动回执和进程身份确认，不比较两端UTC字符串冒充预先约定的证据。

6. 预算、后续判据与风险
仅分配本组12轮：最多38832个完整训练调用、9708次优化器尝试/真实更新上限、48小时墙钟。架构累计调用总上限39092，包含此前260次。孤立AMP溢出按预设规则记录/跳步，连续3或累计32次停止；原始loss/参数非有限、OOM、数据/源码不匹配或墙钟到限保留并退出。监督器仅运行这一子进程，没有自动恢复、加轮、启动候选模块或旧队列。
12轮完成后按预定门槛查学习充分性，再决定是否执行11组S/M对照。后续计划的548原图评估与额外330对照调用当前分配0次，需按既定设计绑定实际固定12 checkpoint与源hash。新用户许可不是待办，待办是运行到端点、核对结果与分配精确执行额度。
候选依然未验证：局部混合与全局状态的因果归属、像素/状态反事实分布偏移、27分组的效能、12轮学习是否充分，以及EVMDet/PIDE-Net完整方法重合均有边界。即使初筛通过，也只能决定是否值得进一步研究；不能把GT供体/掩膜当可部署算法，不能把代理IoU当AP增益。编码器、交互和解码器的总体选题保持开放。

7. 可核验链接与文件
https://github.com/YuHengsss/VSSD/tree/c10de2f8b717a04459b09cbd651de2e768f302c8
https://github.com/RicePasteM/Dome-DETR/tree/2dde3bc1946a3e9fad9abd0612b59fc39bd6b861
科学设计和最近文献范围见52_A21共享状态路径隔离与诊断研究设计.txt及其scientific_design_PRESPECIFIED.json。
A22目录保存8份执行源码、冻结manifest、CPU检查、全量数据清单、启动ZIP与进程快照。启动证据ZIP SHA256={e['sha256']}，16成员；当前汇总为55_A22训练启动与首批更新核验回执.json。
目标继续ACTIVE；本报告只确认训练启动与部分真实更新，不是12轮完成、因果结果或可发表主线通过声明。
'''
    (R/REPORT).write_text(report,encoding='utf-8')
    docs={n:rd(R/n) for n in C};board,search,matrix,state,source=[docs[n] for n in C]
    assert len(board['candidates'])==14 and len(matrix['rows'])==77
    assert state['model_forwards']==260 and state['cap']==39092 and state['training_updates']==2
    evidence={'utc':utc,'report':str(R/REPORT),'status':'FIXED12_DIAGNOSTIC_BASELINE_RUNNING',
       'supervisor_PID':13761,'supervisor_start_ticks':99433280,'training_PID':13762,'training_start_ticks':99433288,
       'job_identity_verified_live':True,'count_snapshot':counts,'new_training_full_cap':38832,'new_actual_update_cap':9708,
       'epoch_cap':12,'wall_cap_hours':48,'no_validation_during_training':True,'actual_eval_calls':0,
       'execution_manifest_sha256':sha(O/'execution_manifest_FROZEN.json'),'startup_archive_sha256':e['sha256'],
       'all_parameter_groups_changed_after_first_update':True,'CPU_preflight_detector_forwards':0,
       'mainline_validated':False,'trained_mechanism_effect_estimated':False,'algorithm_increment_trained':False}
    priority='固定12轮诊断基线已运行；监测原训练PID13762/start99433288及监督PID13761/start99433280，禁止凭超时/锁文件重启。固定端点后核对学习门槛并按预定S/M对照分配评估额度。当前无机制或新算法结论；总体编码器/解码器选题仍开放。'
    for cand in board['candidates']:
        if cand['id']=='N_NONCAUSAL_STATE_INSTANCE_INTERFERENCE':cand.update(status='DIAGNOSTIC_BASELINE_TRAINING_RUNNING_MECHANISM_UNTESTED',A22_evidence=evidence,next=priority)
    board.update(A22_evidence=evidence,latest_update_utc=utc,latest_report=str(R/REPORT),current_priority_question=priority,mainline_validated=False)
    search['A22_code_and_data_verification']={**evidence,'new_literature_queries':0,'core_matrix_rows_unchanged':77,
       'primary_source_checks':['native coco_dataset.py orig_sizeW/H','native postprocessor flatTopQ','installed COCOeval precision@500']}
    search['supplement_utc']=utc
    matrix.update(updated_utc=utc,latest_report=str(R/REPORT),latest_empirical_boundary='A22 fixed12 baseline is live, partial updates only; no trained validation/causal effect/AP improvement')
    for row in matrix['rows']:
        if row['id']=='VSSD':row['A22_runtime_scope']=evidence
    source['A22_fixed12_diagnostic_baseline']={**evidence,'directory':str(O),'scientific_design':str(O/'scientific_design_PRESPECIFIED.json'),
       'evaluation_code':str(O/'evaluate_fixed_endpoint.py'),'evaluation_execution_allocated':False}
    source['next_instance_ownership_question']={'question':priority,'read_only_inspector':'/root/c10_region_restore_20261029_v1/architecture_probe_v1/a22_vssd_dome_pilot12_20261030_v1/inspect_job.py',
       'training_live':True,'no_restart_on_observation_timeout':True,'new_user_permission_required':False}
    source['updated_utc']=utc
    state.setdefault('authoritative_phase_history',[]).append(state['latest_authoritative_phase'])
    state.update(recorded_utc=utc,model_forwards=260+forward_observed,training_updates=2+updates_observed,
       formal_detector_training_updates=updates_observed,capacity_only_optimizer_updates=2,
       detector_optimizer_updates_including_capacity=2+updates_observed,formal_training_epochs=progress['epochs_completed'],
       training_counts_terminal=False,counters_observed_lower_bounds_while_job_live=True,
       current_goal_turn='progress_A22_fixed12_raw_diagnostic_baseline_live_with_verified_real_updates',
       latest_report=str(R/REPORT),current_priority_question=priority,goal_status='ACTIVE',mainline_validated=False,
       training_update_scope='2 historical A20 capacity-only updates plus observed completed A22 formal pilot updates; live lower bounds, not terminal totals.64 diagnostic fits/72suffix calls separate.',
       cap_scope='260closed historical calls +38832allocated fixed12 training calls;39092total capOPEN. Endpoint/assay allocated0; no automatic resume/extension.')
    state['latest_authoritative_phase']={**evidence,'phase':'A22','classification':'progress','goal_status':'active',
       'full_model_forwards':260+forward_observed,'cumulative_authorized_protocol_cap':39092,
       'full_forward_cap_closed':False,'actual_cumulative_detector_optimizer_updates':2+updates_observed,
       'formal_detector_training_updates':updates_observed,'counts_terminal':False,'counts_are_observed_lower_bounds':True,
       'bounded_phase_complete':False,'launch_validation_complete':True,'complete_mainline':False,'next':priority}
    before=R/'history_before_A22_running_record_v1';before.mkdir(exist_ok=False)
    for n in C:shutil.copy2(R/n,before/n)
    for n,x in docs.items():wr(R/n,x)
    names=C+[REPORT,'record_A22_launch.py']+[str(p.relative_to(R)).replace('\\','/') for p in sorted(O.iterdir()) if p.is_file()]
    names +=[str(p.relative_to(R)).replace('\\','/') for p in sorted(D.iterdir()) if p.is_file()]
    receipt={**evidence,'utc':utc,'status':'A22_FIXED12_BASELINE_LAUNCH_AND_REAL_UPDATES_VERIFIED_GOAL_ACTIVE',
       'goal_status':'ACTIVE','cumulative_complete_forward_entries_observed':260+forward_observed,
       'complete_forward_cap':39092,'complete_forward_cap_closed':False,
       'actual_cumulative_updates_observed':2+updates_observed,'formal_updates_observed':updates_observed,
       'counts_terminal':False,'core_literature_rows':77,'candidate_rows':14,'prior_A21_hashes_verified':len(previous['files']),
       'historical_suffix_forwards':72,'historical_CPU_diagnostic_fits':64,'CPU_postprocessor_fixture_calls':1,
       'files':[{'path':n,'sha256':sha(R/n),'bytes':(R/n).stat().st_size} for n in names],
       'next':'Read exact original live PID/start_ticks and its terminal receipt; do not relaunch or treat an observation timeout as termination',
       'limits':'Still running; no fixed12 checkpoint, trained AP, mechanism estimate, new increment or validated final research mainline'}
    wr(RECEIPT,receipt)
    print(json.dumps({k:receipt[k] for k in ['status','goal_status','mainline_validated','cumulative_complete_forward_entries_observed','formal_updates_observed','counts_terminal','training_PID','supervisor_PID']},ensure_ascii=False))
if __name__=='__main__':main()
