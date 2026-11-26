from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

if __name__=='__main__':
    verified=load(ROOT/'B_completed_local_verification.json');assert verified['status']=='PASS'
    obs=load(ROOT/'observer_05.json');assert obs['status']=='EXECUTED' and obs['exit']==0
    observed=json.loads(obs['stdout']);b=observed['arms']['B'];m=observed['arms']['M']
    assert b['result']['status']=='COMPLETED' and not b['identity']['present']
    assert m['identity']['present'] and m['identity']['same_start_ticks'] and m['identity']['owned_command']
    init=load(ROOT/'M_startup/initialization.json')
    assert init['all_initial_tensors_identical'] and init['model_parameters']==9803112 and init['seed']==20261101
    assert init['common_init_sha256']==b['result']['initialization_sha256']
    flow=[]
    for epoch in [1,2]:
        base=load(ROOT/'completed_B_v1/arm_B'/('epoch%02d_input_flow.json'%epoch))
        control=load(ROOT/'M_startup'/('epoch%02d_input_flow.json'%epoch))
        for key in ['order_sha256','first_eight_actual_input_sha256','first_eight_pre_forward_rng_sha256','all_batches_targets_sha256']:
            assert base[key]==control[key],(epoch,key)
        calls=control['normalization_calls']
        assert len(calls)==4 and all(v['weighted_training_calls']==epoch*809-1 for v in calls.values())
        flow.append({'epoch':epoch,'sampler_first8_actual_inputs_first8_rng_all_targets_match':True,
                     'four_sites_weighted_training_calls_each':epoch*809-1})
    metrics=verified['endpoint_metrics']
    lines=['2026-11-01｜A48原版B12完成，本轮目标继续active，主线仍未确认。',
      'B原版：12轮/9708次正式更新，10159次全模型前向（训练9708+三次验证411+延迟测量40）。',
      '固定第12轮EMA：AP %.6f，APs %.6f，small AP75 %.6f，ARs %.6f。'%(metrics['AP'],metrics['APs'],metrics['APs75'],metrics['ARs']),
      '口径：548val、38602评估GT、maxDets300、class0区域过滤、原图框面积；COCO-style，非官方VisDrone成绩。',
      'B训练进程1392已正常退出0；监督器1305继续，M训练进程1927/start_ticks18996958已启动并存活。',
      '已下载完整B结果包312786148字节，SHA124c8cb888328bc7ee6811377494037a5197da8438c0f98b036c945594863178。',
      '38个文件逐项字节数/SHA及解包CRC核验通过，包括第4/8/12轮权重、预测、GT、精度召回数组与日志；未删除远端原件。',
      '本地CPU检查：raw/EMA与公共初始化键/shape/dtype一致，学习状态确实改变；四个BN出口计数均9708且affine非零改变；482份AdamW参数状态step均9708。',
      '这些是文件/模型状态完整性核验，未重新执行GPU预测，不声称独立复现AP；新增模型前向/优化为0。',
      'M初始化与B相同；前两轮顺序、前8个实际输入批、前向前CPU/CUDA RNG与全部目标流SHA一致。四个加权BN出口实际调用分别808/1617次，与第一步lambda0回退普通BN一致。',
      '最新M快照：'+json.dumps(m['progress'],ensure_ascii=False),
      'Q尚未开始。M/Q仍按既定12轮终点执行；没有新候选AP差值，不提前判有效，也不更换lambda/位置/seed。',
      '主要门槛保持：Q-B APs>=0.5pp，Q-M APs>=0.3pp，Q-B AP>=-0.2pp；未过停止本配方并继续换实质不同方向。',
      '最新可更新进度入口CURRENT_A48.json；引用值须回到对应不可变原始SSH观察，下一轮须重新查询。']
    (ROOT/'12_B12终点与M启动验收.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    names=['12_B12终点与M启动验收.txt','B_completed_local_verification.json','B_completed_bundle_download.json',
           'B_bundle_ready.json','M_startup/initialization.json','M_startup/epoch01_input_flow.json',
           'M_startup/epoch02_input_flow.json','observer_04.json','observer_05.json','11_计时与随机流核查_不改训练.txt']
    receipt={'status':'PASS','goal_status':'active','mainline_confirmed':False,'previous_turn_classification':'progress_and_verified_wait',
       'this_turn_classification':'progress_and_verified_wait','B_completed_exit0':True,'B_archive_payload_files_verified':38,
       'B_fixed12_metrics':metrics,'M_two_epoch_flow_checks':flow,'M_live_identity':m['identity'],
       'Q_status':'not_started','new_separate_detector_operations':0,'new_optimizer_jobs':0,
       'observed_stage_counters_lower_bound':observed['observed_counters_lower_bounds'],
       'protocol_sha256':sha(ROOT/'protocol_FROZEN.json'),'files_sha256':{f:sha(ROOT/f) for f in names}}
    (ROOT/'13_B终点与继续验证回执.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'status':'PASS','goal_status':'active','B12_AP':metrics['AP'],'B12_APs':metrics['APs'],
              'M_actual_epoch_flow_checks':flow,'M_progress':m['progress'],'mainline_confirmed':False}))
