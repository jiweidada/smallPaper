"""Publish the observational B4 result and scoped, zero-forward cache audits."""
from pathlib import Path
from datetime import datetime,timezone
import json
import hashlib
import shutil
import os

P=Path(__file__).resolve().parent
B=P.parent
R54=B/'A54_场景尺度约束候选_20261101_v1'
def load(p):return json.loads(p.read_text('utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(4<<20),b''):h.update(x)
    return h.hexdigest()
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def replace(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(t,p)


def main():
    live_files=sorted(R54.glob('A57_B4_final_live_*_rpc.json'));assert len(live_files)==1
    rpc=load(live_files[0]);assert rpc['exit']==0;live=json.loads(rpc['stdout'])
    assert live['supervisor']['status']=='RUNNING' and live['supervisor']['active_arm']=='B'
    assert all(r['owned_live'] for r in live['owned_jobs']) and live['frozen_inputs_unchanged']
    assert {(r['pid'],r['start_ticks']) for r in live['owned_jobs']}=={(17829,26463967),(17830,26464019)}
    assert 'supervisor_exit' not in live and not any('failure.json' in a for a in live['arms'].values())
    assert set(live['arms'])=={'B'}
    metrics=load(P/'B_epoch04_remote_v1/arm_B/epoch04_RAW_metrics.json')
    cached=load(P/'B_epoch04_cached_audit_v1.json');weight=load(P/'B_epoch04_checkpoint_local_verification.json')
    assert cached['status']=='PASS_CACHED_METRIC_AND_FORMAT_AUDIT'
    assert weight['status']=='PASS_CHECKPOINT_BYTES_ENVELOPE_FINITE_AND_DYNAMIC_PE_CACHE'
    assert cached['GT_semantically_identical'] and weight['all_floating_tensors_finite'] and weight['selected_weights_updated']
    assert abs(metrics['AP']-cached['metrics_recomputed_from_saved_precision']['AP'])<1e-12
    assert live['arms']['B']['metrics_by_epoch.json']['4']==metrics
    polls=sorted(R54.glob('A57_wait05_poll_*_rpc.json'));assert len(polls)==93
    seen=[]
    for f in polls:
        r=load(f);assert r['exit']==0;s=json.loads(r['stdout'])
        assert s['status']=='RUNNING' and s['active_arm']=='B' and all(j['owned_live'] for j in s['jobs'])
        seen.append(s)
    save(P/'B4_bounded_wait_receipts.json',{'previous_goal_turn_classification':'verified_wait','polls':len(polls),
        'first_remote_utc':seen[0]['utc'],'last_remote_utc':seen[-1]['utc'],'first_AP_at_epoch':4,
        'files':{str(f):sha(f) for f in polls},'extra_model_forwards':0,'extra_optimizer_updates':0})
    prior=load(P/'A57_B_epoch01_preservation_receipt.json')
    mutable=[B/'01_候选判别表.json',B/'15_当前状态与结果校验.json',P/'CURRENT_A57.json']
    before=P/'before_B_epoch04_update';before.mkdir(exist_ok=False);mapping={}
    for f in mutable:
        row=prior['files'][str(f)];assert sha(f)==row['sha256'] and f.stat().st_size==row['bytes']
        target=before/f.name;shutil.copy2(f,target);mapping[str(f)]={'snapshot':str(target),'sha256':sha(target)}
    save(P/'prior_B_epoch01_resolution.json',{'reason':'Current status advances after first B4 AP; old first-epoch receipt resolves through these exact immutable snapshots.','files':mapping})
    save(P/'B_epoch04_latest_live_snapshot.json',live)
    j=live['arms']['B']['journal'];progress=live['arms']['B']['progress.json']
    totals={'model_forwards':226281+3+j['entries'],'cap':285885,'formal_detector_training_updates':194686+j['updates'],
        'capacity_only_optimizer_updates':76,'training_updates':194762+j['updates'],
        'detector_optimizer_updates_including_capacity':194762+j['updates'],'formal_training_epochs':240+j['epochs_completed']}
    report=P/'08_B4低AP与已核查范围.txt'
    lines=[
        'A57 B4首次AP：基线检测表现很低；Q未正式训练，不能判定算法增量',
        '',
        '当前方案：VisDrone2019-DET + pinned MambaSOD作者源码兼容版。B为源码MQI；M为普通全网格SSM编码后采样；Q为完整上下文状态上的query条件C读出。',
        '当前宿主仅有ImageNet Vim预训练，没有检测器预训练；本次是统一12轮短日程，不是原文300轮复现。最近方法和历史差异仍见01_候选问题与最近方法对照.txt。',
        '',
        '已证实事实：第4轮raw观察值（0–100尺度）',
        f'AP {metrics["AP"]:.12f}；AP50 {metrics["AP50"]:.12f}；AP75 {metrics["AP75"]:.12f}。',
        f'APs {metrics["APs"]:.12f}；APs75 {metrics["APs75"]:.12f}；ARs {metrics["ARs"]:.12f}。',
        '这些是极低的检测指标，不能用loss有限或权重变化称为已有可用性能，更不能称Q已有提升。M/Q尚未进入正式训练；此前仅有丢弃权重的容量步骤。',
        'B4训练6472次整模型前向、3236次优化器尝试及返回更新；验证137次前向，548图、38602有效GT、157个有效GT被ignore过滤。事件前缀合计6609次进入/完成，已本地重计数。',
        '',
        '已核查的范围',
        '缓存precision/recall数组重算8项指标与日志一致；本次没有重新运行模型或COCO evaluator。GT与冻结参考逐字段一致，图像ID/类别范围、框正面积、有限数值检查通过。',
        '矩形图像的宽高缩放在开训前已核验；目前代码仍按orig_size=[宽,高]转换。缓存框归一化尺寸处于合理数值范围，但这不能证明模型定位已学好。',
        'B4 raw权重317767876字节，SHA256 51861becf41c7b40528a99b8927e149fb5bd0f44432172e599f3ac71aec04a8b；原件已下载本地并校验。',
        '2011个状态张量浮点值均有限；骨干patch投影、分类头、框头、query和参考框都发生实际变化。不存在“只记更新次数但所选权重完全未动”的情况；这仍不是检测效果证明。',
        '唯一张量形状变化是encoder.pos_encoding.pe从100×100×256扩为160×160×256，源自作者代码按P2尺寸生成的确定性缓存。与冻结源码重新生成的缓存完全相同，前100×100区域也与初始化一致；不是新增可学习参数。',
        '冷加载此权重需先将位置编码缓存构建为160×160，再strict加载；无需通过检测前向扩容，不据此恢复或重试训练。操作说明见B_epoch04_checkpoint_local_verification.json。',
        '',
        '描述性诊断，不能当AP或一对一召回',
        '已选top300缓存预测的GT覆盖上界：IoU≥0.5时忽略类别16.1598%、同类别10.7585%；IoU≥0.75时分别1.5621%和1.0880%。同一个预测可以覆盖多个GT，因此这些数不是COCO recall，也不用于改变门槛。',
        '这批预测仍有定位/类别及排序问题，不能由覆盖上界直接归因于其中某一个环节。',
        '',
        '源码事实与尚未证实的原因',
        '作者原loss_labels采用focal.mean(1).sum()/num_GT。每组400 query时，相同focal值相对query求和口径有1/400系数；分类权重2等效于sum口径0.005，框L1/GIoU权重为2/5。当前配方忠实保留这一公式，辅助头的求和也与原train循环相同。',
        '这种相对尺度可能影响短训共享特征学习，但没有测量实际分项梯度或反事实性能，不能断言它是低AP唯一原因、必然是错误，或把改一个系数包装成论文创新。AdamW的归一化也使损失系数不能直接解释为模型更新幅度比例。',
        '检查没有覆盖完整旧版内核与当前宿主的端到端数值等价、充分收敛或原论文完整复现。因此只能说已核查范围未发现GT/缩放/汇总/选定权重更新问题，不能宣称所有接入问题已被排除。',
        '',
        '继续判定，协议未变',
        f'最近现场：{live["utc"]}（远端UTC）；原B处于第{progress["epoch"]}轮，事件前缀{j["entries"]}次模型进入、{j["completions"]}次完成、{j["updates"]}次更新，已完成{j["epochs_completed"]}轮。为运行中的快照下界。',
        '原监督器17829/start26463967和worker17830/start26464019均现场核验存活，冻结源码及公共初始化哈希匹配。',
        '第4/8轮只观察；B12仍需AP≥8且APs≥3才启动M/Q。若B12不足，取消未启动M/Q，记为当前宿主/短训配方不足，不能记为Q训练阴性。',
        '若B通过，Q12同时需Q−B APs≥+0.5pp、Q−M APs≥+0.3pp、Q−B AP≥−0.2pp；未通过则停止本配方。当前不修改损失、增加epoch/seed或替换模块挽救，不重试。',
        '当前判断：原版短训学习水平是首要不确定性。A57增量尚无正式性能检验，论文主线仍未成立；目标保持active。',
    ]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    evidence=load(before/'CURRENT_A57.json')
    evidence.update(status='A57_B4_VERY_LOW_AP_ORIGINAL_B_CONTINUES',report=str(report),observed_utc=live['utc'],
        formal_model_forwards=j['entries'],formal_model_forward_completions=j['completions'],formal_detector_optimizer_updates=j['updates'],formal_epochs=j['epochs_completed'],
        live_snapshot=str(P/'B_epoch04_latest_live_snapshot.json'),new_A57_AP_metrics={'B_epoch4_RAW':metrics},
        B4_primary_gate_assessed=False,Q_formal_training_started=False,M_formal_training_started=False,
        B4_exact_validation_prefix={'model_entries':6609,'model_completions':6609,'optimizer_updates':3236,'epochs':4},
        cache_metric_audit=str(P/'B_epoch04_cached_audit_v1.json'),checkpoint_verification=str(P/'B_epoch04_checkpoint_local_verification.json'),
        static_loss_scale_audit=str(P/'B_epoch04_loss_scale_static_audit_v1.json'),
        low_AP_cause='UNRESOLVED; inherited relative classification loss scaling is a hypothesis, not established causality',
        next='Observe original B8/12; frozen B12 learning gate controls M/Q. No mid-run fix, automatic retry or rescue.')
    replace(P/'CURRENT_A57.json',evidence)
    now=datetime.now(timezone.utc).isoformat()
    for n in ['01_候选判别表.json','15_当前状态与结果校验.json']:
        d=load(B/n);d.update(status=evidence['status'],A57_execution=evidence,latest_report=str(report),latest_update_utc=now,
            goal_status='active',mainline_validated=False,candidate_selected_for_pilot=True,mainline_selected_for_pilot=False,
            pilot_validation_status='B4 very low AP; B continues to fixed12 learning gate; M/Q formal training not started')
        if n.startswith('01_'):
            for row in d['candidates']:
                if row['id']=='A57_FULL_GRID_SPATIAL_STATE_READOUT':row.update(status=evidence['status'],report=str(report),latest_live_snapshot=evidence['live_snapshot'],B4_observation=metrics,Q_formal_training_started=False)
        else:
            d.update(totals);d.update(recorded_utc=now,current_phase='A57_B_TRAINING_AFTER_VERY_LOW_B4_AP',
                counts_terminal=False,training_counts_terminal=False,current_native_job_terminal=False,full_forward_cap_closed=False,
                counters_observed_lower_bounds_while_job_live=True,previous_goal_turn_classification='verified_wait: original B progressed; no AP previously',
                current_goal_turn='progress: first B4 AP obtained, cached metrics/GT/coordinates/updated checkpoint audited; baseline learning very poor, Q untested; original fixed12 continues',
                current_user_task_status='B4 AP0.0378878/APs0.0273510; source-faithful short baseline insufficient so far, wait B8/12; no Q efficacy claim')
            d.setdefault('authoritative_phase_history',[]).append({'phase':'A57','utc':now,'classification':'progress','goal_status':'active','mainline_validated':False,
                'B4_metrics':metrics,'observed_lower_bound_counts':totals,'report':str(report),'source_bound_protocol_unchanged':True})
        replace(B/n,d)
    save(P/'goal_progress_A57_B4_verified.json',{'classification':'progress','previous_goal_turn_classification':'verified_wait',
        'goal_status':'active','mainline_validated':False,'B4_metrics':metrics,'observed_cumulative_lower_bounds':totals,
        'exact_B4_validation_prefix':evidence['B4_exact_validation_prefix'],'live_snapshot':evidence['live_snapshot'],
        'extra_model_forwards_by_audit':0,'extra_optimizer_updates_by_audit':0,'extra_COCO_evaluate_calls':0,
        'CPU_positional_buffer_constructions':1,'report':str(report),'primary_gate_or_recipe_changed':False})
    for n,row in prior['files'].items():
        f=Path(mapping[n]['snapshot']) if n in mapping else Path(n)
        assert sha(f)==row['sha256'] and f.stat().st_size==row['bytes'],n
    save(P/'prior_B_epoch01_receipt_reverified.json',{'status':'PASS','files_verified':len(prior['files']),'resolution':str(P/'prior_B_epoch01_resolution.json')})
    files=[f for f in P.rglob('*') if f.is_file() and '__pycache__' not in f.parts]+mutable[:2]
    save(P/'A57_B4_preservation_receipt.json',{'status':'B4_LOW_AP_AND_SCOPED_AUDITS_PRESERVED','goal_status':'active','mainline_validated':False,
        'observed_lower_bound_counts':totals,'files':{str(f):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(set(files))},
        'extra_model_forwards_by_publication':0,'extra_optimizer_updates_by_publication':0})
    print(json.dumps({'status':evidence['status'],'B4_AP':metrics['AP'],'B4_APs':metrics['APs'],'counts':totals,'report':str(report),'goal_status':'active'},ensure_ascii=False))


if __name__=='__main__':main()
