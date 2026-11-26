from pathlib import Path
import argparse,datetime,hashlib,json,shutil

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
A47=BASE/'A47_微小框匹配稳定性选题与验证_20261101_v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def save(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def verify_resolved(data):
    count=0
    if isinstance(data,dict):
        if 'preserved_at' in data and 'sha256' in data:
            assert sha(data['preserved_at'])==data['sha256'],data['preserved_at']
            return 1
        for value in data.values():count+=verify_resolved(value)
    return count

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('status_receipt');args=parser.parse_args()
    remote=json.loads((ROOT/args.status_receipt).read_text())
    assert remote['status']=='EXECUTED' and remote['exit']==0
    observation=json.loads(remote['stdout']);files=observation['files']
    capacity=files['capacity_result.json'];baseline=files['arm_B/progress.json']
    assert capacity['status']=='PASS' and capacity['optimizer_updates']==9 and capacity['full_model_forwards']==12
    assert baseline['status'] in ['TRAINING','VALIDATING','COMPLETED']
    receipt_path=A47/'06_A47结果与历史保留校验回执.json'
    old=json.loads(receipt_path.read_text(encoding='utf-8'))
    before=ROOT/'before_A48';before.mkdir(exist_ok=False)
    remapped={}
    for name in ['01_候选判别表.json','15_当前状态与结果校验.json']:
        path=BASE/name
        assert sha(path)==old['current_files_sha256'][str(path)]
        copy=before/name;shutil.copyfile(path,copy)
        assert sha(copy)==sha(path)
        remapped[str(path)]={'preserved_at':str(copy),'sha256':sha(copy)}
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    question='每个GT框等量参与编码器训练归一化统计，能否比普通BN和框像素加权提高VisDrone小目标检测？'
    evidence={'id':'A48','status':'PILOT_TRAINING_NOT_VALIDATED','dataset':'VisDrone2019-DET train6471/val548',
              'host':'ECDet-S','question':question,'increment':'training-only per-instance box-weighted encoder BN moments',
              'previous_BN_boundary':'20261027 fixed raw/EMA state replacement stopped; no old state swapping here',
              'protocol':str(ROOT/'protocol_FROZEN.json'),'report':str(ROOT/'01_候选主线与最小验证说明.txt'),
              'capacity':capacity,'baseline_observed_progress':baseline,'observed_unix':observation['observed_unix'],
              'fixed_endpoint':'B/M/Q each12, fixed12EMA; no rescue sweep or additional queue',
              'mainline_confirmed':False,'evidence_level':'credible label-area imbalance and explicit comparator, efficacy untested',
              'source_snapshot':str(ROOT/args.status_receipt)}
    board=json.loads((before/'01_候选判别表.json').read_text(encoding='utf-8'))
    board.update(date='2026-11-01',status='A48_PILOT_IN_PROGRESS_MAINLINE_UNCONFIRMED',goal_status='active',
          goal_status_scope='Broad user goal: continue until an evidence-supported mainline is found; another negative diagnostic is not completion.',
          mainline_validated=False,mainline_selected_for_pilot=True,selection_stage_complete=False,
          current_priority_candidate='A48 实例等量编码器归一化统计',current_priority_question=question,
          current_host_candidate='ECDet-S',latest_report=evidence['report'],latest_update_utc=utc,
          remaining_deliverable='固定12轮B/M/Q终点与输入一致性；未达标则换实质不同方向，当前不宣称新颖性成立。',A48_execution=evidence)
    board.setdefault('candidates',[]).append({'id':'A48','name':'实例等量编码器归一化统计','status':'TRAINING_PILOT','mainline_validated':False})
    save(BASE/'01_候选判别表.json',board)
    state=json.loads((before/'15_当前状态与结果校验.json').read_text(encoding='utf-8'))
    new_forward=12+baseline['full_model_forwards'];new_updates=9+baseline['optimizer_updates']
    state.update(recorded_utc=utc,goal_status='active',mainline_validated=False,diagnostics_complete=False,
          latest_authoritative_phase='A48_INSTANCE_BALANCED_MOMENT_TRAINING_PILOT',
          latest_report=evidence['report'],current_priority_question=question,current_priority_candidate=board['current_priority_candidate'],
          mainline_selected_for_pilot=True,model_forwards=83963+new_forward,training_updates=58793+new_updates,
          formal_detector_training_updates=58774+baseline['optimizer_updates'],capacity_only_optimizer_updates=28,
          detector_optimizer_updates_including_capacity=58793+new_updates,
          formal_training_epochs=72+baseline.get('completed_epochs',baseline.get('epochs',0)),
          training_counts_terminal=False,counters_observed_lower_bounds_while_job_live=True,counts_terminal=False,
          full_forward_cap_closed=False,cap=114452,
          cap_scope='A48 adds at most12 capacity and30477 formal full-model forwards; older closed phases preserved. No numerical goal-token budget.',
          A48_execution=evidence,current_goal_turn={'classification':'progress','goal_status':'active','mainline_confirmed':False,
                         'not_complete_just_because_one_candidate_tested':True},
          goal_status_scope=board['goal_status_scope'])
    state.setdefault('authoritative_phase_history',[]).append({'utc':utc,'phase':'A48','status':'PILOT_TRAINING','mainline_confirmed':False})
    save(BASE/'15_当前状态与结果校验.json',state)
    resolved={}
    for name,expected in old['current_files_sha256'].items():
        item=remapped.get(name,{'preserved_at':name,'sha256':expected})
        assert item['sha256']==expected and sha(item['preserved_at'])==expected,name
        resolved[name]=item
    older_count=verify_resolved(old['previous_A46_all49_inputs_resolved'])+verify_resolved(old['older_input_resolution'])
    assert len(resolved)==105 and older_count==122
    save(ROOT/'04_A48本次启动观察.json',observation)
    lines=['2026-11-01｜A48新候选最小训练验证已启动，主线尚未确认。',
      '问题：'+question,'数据/宿主：VisDrone2019-DET / ECDet-S。',
      '三组：B普通BN；M框像素加权；Q每实例等量。各12轮、相同初始化/seed/数据流，固定12EMA。',
      '数据准备：6471训练图+标注、548验证图+标注全部SHA通过，与旧A43清单一致；现有环境复用。',
      '容量：3组各3次真实更新及1次无GT原BN推理，合计12前向/9更新；与正式训练分记。',
      '现场B观察：'+json.dumps(baseline,ensure_ascii=False),
      'M/Q尚未开始；由已冻结监督器依次执行，失败不重试；无额外方向训练队列。',
      '本次新增已观察计数下界：全模型前向%d，优化更新%d。'%(new_forward,new_updates),
      '旧A47的105项与更早122项均经保存路径核验，旧裁决/源码/权重未覆盖。',
      '尚无A48新验证AP。数量/面积差异、代码检查、容量通过和开训都不是有效性/新颖性证明。',
      '目标保持active；不得仅因完成一次阴性诊断标记整个找主线目标完成。']
    (ROOT/'02_A48训练启动与实时状态.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    signed=[ROOT/'protocol_FROZEN.json',ROOT/'01_候选主线与最小验证说明.txt',ROOT/'02_A48训练启动与实时状态.txt',
            ROOT/'04_A48本次启动观察.json',ROOT/'literature_method_matrix.json',ROOT/'search_raw_records_v1.json',
            ROOT/'training_assets_upload.json',ROOT/'verified_launch_response.json',
            BASE/'01_候选判别表.json',BASE/'15_当前状态与结果校验.json']
    signed += [ROOT/name for name in json.loads((ROOT/'protocol_FROZEN.json').read_text())['files_sha256']]
    result={'status':'PASS','goal_status':'active','broader_mainline_found':False,'phase':'A48_TRAINING_STARTED',
          'previous_A47_receipt':str(receipt_path),'previous_A47_receipt_sha256':sha(receipt_path),
          'previous_A47_all105_inputs_resolved':resolved,'older_A46_A45_A107_A105_preserved_count':older_count,
          'older_input_resolution':{'A46':old['previous_A46_all49_inputs_resolved'],'older':old['older_input_resolution']},
          'new_observed_forward_lower_bound':new_forward,'new_observed_update_lower_bound':new_updates,
          'current_files_sha256':{str(p):sha(p) for p in signed},'observation_unix':observation['observed_unix']}
    save(ROOT/'05_A48启动与历史保留校验回执.json',result)
    print(json.dumps({'status':'PASS','previous_current105':105,'earlier_preserved':older_count,
              'goal_status':'active','mainline_confirmed':False,'new_forward_lower_bound':new_forward,
              'new_updates_lower_bound':new_updates,'baseline':baseline}))
