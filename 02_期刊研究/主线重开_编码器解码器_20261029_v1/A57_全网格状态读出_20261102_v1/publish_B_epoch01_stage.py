"""Metadata-only publication of the first completed epoch and CPU flow audit."""
from pathlib import Path
from datetime import datetime,timezone
import json
import hashlib
import shutil
import os

P=Path(__file__).resolve().parent
B=P.parent
R54=B/'A54_场景尺度约束候选_20261101_v1'
def load(f):return json.loads(f.read_text('utf-8'))
def sha(f):
    h=hashlib.sha256()
    with f.open('rb') as q:
        for c in iter(lambda:q.read(4<<20),b''):h.update(c)
    return h.hexdigest()
def save(f,z):
    with f.open('x',encoding='utf-8') as q:json.dump(z,q,indent=2,ensure_ascii=False,allow_nan=False)
def replace(f,z):
    t=f.with_suffix(f.suffix+'.tmp');t.write_text(json.dumps(z,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(t,f)


def main():
    rpc=load(R54/'A57_B_epoch01_post_audit_live_rpc.json');assert rpc['exit']==0
    live=json.loads(rpc['stdout'])
    assert live['status']=='RUNNING' and live['active_arm']=='B' and all(r['owned_live'] for r in live['owned_jobs'])
    assert live['failure'] is None and live['AP_metrics'] is None and live['frozen_inputs_unchanged']
    assert {(r['pid'],r['start_ticks']) for r in live['owned_jobs']}=={(17829,26463967),(17830,26464019)}
    local=load(P/'B_epoch01_local_verification.json');assert local['status']=='PASS_LOCAL_BYTES_ORDER_AND_EVENT_PREFIX_RECOUNT'
    audit=load(P/'B_epoch01_remote_v1/B_epoch01_dataflow_audit_v1.json')
    assert audit['status']=='PASS_FIRST_EPOCH_ORDER_TARGET_PIXEL_AND_COUNT_AUDIT'
    assert audit['whole_model_forwards']==audit['optimizer_updates']==0 and audit['CPU_only']
    first=load(P/'B_epoch01_remote_v1/arm_B/epoch01_train.json')
    j=live['journal'];assert j['epochs_completed']==1 and j['updates']>=809 and j['entries']>=1618
    prior=load(P/'A57_formal_start_preservation_receipt.json')
    mutable=[B/'01_候选判别表.json',B/'15_当前状态与结果校验.json',P/'CURRENT_A57.json']
    before=P/'before_B_epoch01_update';before.mkdir(exist_ok=False);mapping={}
    for f in mutable:
        row=prior['files'][str(f)];assert sha(f)==row['sha256'] and f.stat().st_size==row['bytes']
        target=before/f.name;shutil.copy2(f,target);mapping[str(f)]={'snapshot':str(target),'sha256':sha(target)}
    save(P/'prior_formal_start_resolution.json',{'reason':'Mutable canonical files advance after verified completed epoch; old formal-start receipt resolves through these exact snapshots.','files':mapping})
    report=P/'07_B首轮数据流核验与继续判定.txt'
    gib=first['counts']['peak_allocated_bytes']/(1<<30)
    lines=[
        'A57 B首轮已完成；原固定12轮进程继续，尚无AP',
        '事实来源：B_epoch01_remote_v1中的原始flow/train/event前缀、CPU审计，以及B_epoch01_local_verification.json。',
        '现场UTC（远端）：'+live['utc']+'；本机与远端时钟分开记录，跨主机不直接相减。',
        '',
        '完成的真实训练与输入核验',
        'B首轮完整结束：6471图、343204目标、1618次模型前向、809次优化器尝试和809次返回完成更新，无跳步。事件前缀4855行已下载、逐文件SHA和事件次数复核。',
        '全部图像顺序与独立seed20261101的randperm一致，无漏图/重复索引；按原始标注重建整轮目标张量，流摘要与训练记录一致。',
        '前8个microbatch、32张图像的处理后像素和目标联合摘要重放一致。这不是全数据集像素逐字节重放。M/Q的跨组完整流一致性仍待各组实际运行。',
        f'首轮训练耗时{first["training_seconds"]:.2f}秒，microbatch平均loss {first["microbatch_mean_loss"]:.6f}，峰值allocated {gib:.3f}GiB。均不能当AP、最终吞吐基准或Q组成本。',
        'CPU审计约17.4秒，额外模型前向0、优化器更新0；原GPU训练继续，代码/日程/预算没有修改。',
        '',
        '原进程仍在运行',
        f'现场事件前缀：B进入{j["entries"]}次前向、完成{j["completions"]}次、完成{j["updates"]}次更新，已完成{j["epochs_completed"]}轮，正在第2轮。数字是快照下界。',
        '监督器17829/start_ticks26463967、worker17830/start_ticks26464019均按现场PID身份核验存活；冻结源码和公共初始化SHA保持匹配。',
        'M/Q尚未启动，等待B12的AP≥8且APs≥3最低学习门槛。目前没有A57 AP，不把loss或输入核验当主线成立。',
        '',
        '下一次科学判定',
        '第4/8轮只观察，第12轮raw权重固定判定；B不足则取消M/Q并标注短训宿主不足。',
        'B通过后Q同时须达到：Q−B APs≥+0.5pp、Q−M APs≥+0.3pp、Q−B AP≥−0.2pp；未通过则淘汰本配方，保留资料，不自动增加seed、轮数或模块。',
        'M/Q共享状态推进、上下文门控和空间读取结构，C分别来自context/query。总注册参数相同不表示每个参数更新相同；详见06_M与Q读取边界核查.txt。',
        '完整问题、最近工作与历史差异见01_候选问题与最近方法对照.txt；同名文献消歧与全文未取得状态见05_MambaSOD同名消歧与全文补查.txt。',
        '',
        '当前结论：候选方案的数据和运行口径得到进一步核验；性能、论文主线和新颖性均尚未证实。目标保持active。',
    ]
    with report.open('x',encoding='utf-8') as q:q.write('\n'.join(lines)+'\n')
    save(P/'B_epoch01_post_audit_live_snapshot.json',live)
    evidence=load(before/'CURRENT_A57.json')
    evidence.update(status='A57_B_EPOCH01_VERIFIED_FIXED12_RUNNING',formal_model_forwards=j['entries'],
        formal_model_forward_completions=j['completions'],formal_detector_optimizer_updates=j['updates'],formal_epochs=j['epochs_completed'],
        observed_utc=live['utc'],live_snapshot=str(P/'B_epoch01_post_audit_live_snapshot.json'),report=str(report),
        first_epoch_dataflow_verified=True,first_epoch_dataflow_verification=str(P/'B_epoch01_local_verification.json'),
        extra_forward_calls_by_dataflow_audit=0,extra_optimizer_updates_by_dataflow_audit=0,
        next='Observe original B epoch4/8/12; M/Q only if fixed12 baseline learning gate passes. No repeat launch, capacity or rescue.')
    replace(P/'CURRENT_A57.json',evidence)
    totals={'model_forwards':226281+3+j['entries'],'cap':285885,'formal_detector_training_updates':194686+j['updates'],
        'capacity_only_optimizer_updates':76,'training_updates':194762+j['updates'],
        'detector_optimizer_updates_including_capacity':194762+j['updates'],'formal_training_epochs':240+j['epochs_completed']}
    now=datetime.now(timezone.utc).isoformat()
    for n in ['01_候选判别表.json','15_当前状态与结果校验.json']:
        d=load(B/n);d.update(status=evidence['status'],A57_execution=evidence,latest_report=str(report),latest_update_utc=now,
            goal_status='active',mainline_validated=False,candidate_selected_for_pilot=True,mainline_selected_for_pilot=False)
        if n.startswith('01_'):
            for row in d['candidates']:
                if row['id']=='A57_FULL_GRID_SPATIAL_STATE_READOUT':row.update(status=evidence['status'],report=str(report),latest_live_snapshot=evidence['live_snapshot'])
        else:
            d.update(totals);d.update(recorded_utc=now,current_phase='A57_B_EPOCH2_RUNNING_FIRST_EPOCH_FLOW_VERIFIED',
                counts_terminal=False,training_counts_terminal=False,current_native_job_terminal=False,
                full_forward_cap_closed=False,counters_observed_lower_bounds_while_job_live=True,
                previous_goal_turn={'classification':'verified_wait','phase':'A57 B first epoch','evidence':str(P/'goal_progress_A57_verified_wait_01.json')},
                previous_goal_turn_classification='verified_wait: owned B worker advanced with frozen source and no AP',
                current_goal_turn='progress: first full training epoch, independent CPU order/target/pixel audit and exact event-prefix recount completed; original B fixed12 continues',
                current_user_task_status='A57 B2 running, epoch1 dataflow verified; first AP at4, fixed12 decision; mainline unresolved')
            d.setdefault('authoritative_phase_history',[]).append({'phase':'A57','utc':now,'classification':'progress',
                'goal_status':'active','status':evidence['status'],'snapshot':evidence['live_snapshot'],
                'observed_lower_bounds':totals,'first_epoch_input_verified':True,'mainline_validated':False,'new_AP':None})
        replace(B/n,d)
    save(P/'goal_progress_A57_B_epoch01_verified.json',{'classification':'progress','previous_classification':'verified_wait',
        'goal_status':'active','mainline_validated':False,'epoch1_verified':True,'live_snapshot':evidence['live_snapshot'],
        'observed_lower_bounds':totals,'new_detector_forwards_by_audit':0,'new_optimizer_updates_by_audit':0,
        'research_boundary_audit':str(P/'A57_static_MQ_boundary_audit_v1.json'),'report':str(report)})
    for n,row in prior['files'].items():
        f=Path(mapping[n]['snapshot']) if n in mapping else Path(n)
        assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256'],n
    save(P/'prior_formal_start_receipt_reverified.json',{'status':'PASS','files_verified':len(prior['files']),'resolution':str(P/'prior_formal_start_resolution.json')})
    files=[f for f in P.rglob('*') if f.is_file() and '__pycache__' not in f.parts]+mutable[:2]
    save(P/'A57_B_epoch01_preservation_receipt.json',{'status':'FIRST_EPOCH_DATAFLOW_AND_CONTINUING_ORIGINAL_B_PRESERVED',
        'goal_status':'active','mainline_validated':False,'observed_lower_bound_counts':totals,
        'files':{str(f):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(set(files))},
        'extra_detector_forwards_by_publication':0,'extra_optimizer_updates_by_publication':0})
    print(json.dumps({'status':evidence['status'],'observed_counts':totals,'report':str(report),'goal_status':'active'},ensure_ascii=False))


if __name__=='__main__':main()
