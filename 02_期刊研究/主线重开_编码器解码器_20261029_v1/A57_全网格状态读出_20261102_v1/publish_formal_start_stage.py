"""Publish verified live-prefix evidence; does not access a GPU or restart jobs."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import os
import shutil
import zipfile

P=Path(__file__).resolve().parent
B=P.parent
ROOT=B.parent.parent
R54=B/'A54_场景尺度约束候选_20261101_v1'


def load(p):return json.loads(p.read_text('utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def replace(p,v):
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(temp,p)


def main():
    rpc=load(R54/'A57_formal_start_snapshot_export_rpc.json');assert rpc['exit']==0
    response=json.loads(rpc['stdout']);m=response['export']
    source=R54/'A57_downloads/formal_start_snapshot_v1.zip';assert sha(source)==m['sha256']
    archive=P/source.name
    assert not archive.exists();shutil.copy2(source,archive)
    dest=P/'formal_start_remote_snapshot_v1';dest.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())==set(m['files'])
        for n,row in m['files'].items():
            f=(dest/n).resolve();assert f.is_relative_to(dest.resolve());data=z.read(n)
            assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
            f.parent.mkdir(parents=True,exist_ok=True)
            with f.open('xb') as q:q.write(data)
    snap=load(dest/'formal_live_snapshot_v1.json')
    assert snap==response['snapshot'] and snap['status']=='PASS_OWNED_BASELINE_RUNNING_WITH_REAL_UPDATES'
    assert all(row['owned_live'] for row in snap['owned_jobs'])
    assert snap['frozen_sources_and_initialization_match']
    assert snap['protocol_sha256']==sha(P/'protocol_FROZEN_v1.json')
    events=[json.loads(s) for s in (dest/'arm_B/events.jsonl').read_bytes().splitlines(keepends=True) if s.endswith(b'\n')]
    counts=snap['B_journal_counts']
    assert max(r['model_counter']['entries'] for r in events)==counts['entries']
    assert max(r['model_counter']['completions'] for r in events)==counts['completions']
    assert sum(r['event']=='model_forward_entered' for r in events)==counts['entries']
    assert sum(r['event']=='model_forward_completed' for r in events)==counts['completions']
    assert sum(r['event']=='optimizer_step_returned' for r in events)==counts['optimizer_updates_returned']
    assert sum(r['event']=='optimizer_step_attempt' for r in events)==counts['optimizer_step_attempts']
    assert counts['optimizer_updates_returned']>0
    save(P/'formal_start_local_verification.json',{'status':'PASS_EXACT_BYTES_AND_JOURNAL_RECOUNT',
        'snapshot_utc':snap['observation_end_utc'],'archive_sha256':m['sha256'],'files_verified':len(m['files']),
        'observed_counts':counts,'scope':'Live journal prefix; not a completed arm or AP result',
        'new_detector_forwards_by_verification':0,'new_optimizer_updates_by_verification':0})
    receipt=load(P/'A57_capacity_stage_preservation_receipt.json')
    mutable=[B/'01_候选判别表.json',B/'15_当前状态与结果校验.json',P/'CURRENT_A57.json']
    before=P/'before_A57_formal_start';before.mkdir(exist_ok=False)
    resolution={}
    for f in mutable:
        old=receipt['files'][str(f)];assert sha(f)==old['sha256'] and f.stat().st_size==old['bytes']
        target=before/f.name;shutil.copy2(f,target)
        resolution[str(f)]={'snapshot':str(target),'sha256':sha(target)}
    save(P/'prior_capacity_stage_resolution.json',{'reason':'Mutable current status advances after real B updates; old capacity receipt resolves through these exact snapshots.',
        'receipt':str(P/'A57_capacity_stage_preservation_receipt.json'),'files':resolution})
    report=P/'03_正式启动与评价口径.txt'
    progress=snap['progress'];gib=progress.get('peak_allocated_bytes',0)/(1<<30)
    text=[
        'A57正式启动记录；主线尚未验证',
        '记录依据：'+snap['observation_end_utc']+' 的远端只读快照，已下载并逐文件SHA及事件行数复核。',
        '',
        '当前问题与三组',
        'VisDrone2019-DET + pinned MambaSOD作者源码兼容版，实际72,697,131参数；不称论文完整复现。',
        '问题：保留同一完整网格状态和同一空间参考后，query条件状态读出是否比普通上下文SSM编码后采样更有小目标价值？',
        'B原源码MQI；M全网格上下文C输出后双线性采样；Q同一上下文状态、query生成C并在同一参考处读取。没有新增P2、query、标签或参数。',
        '空间压缩造成小目标损失仍是待验证假设；原码池化结构及算子等价核验不能直接证明任务瓶颈。',
        '',
        '开训前已完成',
        '恰好3次真实eval-group1前向（B/M/Q各一次，batch4）；0优化器更新、0 AP评价。均输出400query、10类和4维框，数值有限。',
        '与既有COCO风格GT逐字段相同：548图、38602有效GT、157有效GT被class0区域union IoA过滤。',
        '原始数据orig_size=[宽,高]；评价接入草稿的相反解释已在开训前更正，1920×1080及600×1200矩形坐标检查通过。',
        '真实预测框与独立NumPy缩放最大差约0.000117像素。后处理为sigmoid query×class top300，无NMS或重评分。',
        '公共初始化文件及所有状态tensor已核验，三组从同一413张量ImageNet Vim预训练映射及同一随机检测器初始化开始；不用容量测试后的权重。',
        '完整源码、算子、GT和COCO依赖哈希已写入 protocol_FROZEN_v1.json，在任何A57 AP之前冻结。',
        '',
        '真实运行证据',
        f'B组第1/12轮；快照事件前缀：整模型进入{counts["entries"]}次、完成{counts["completions"]}次、返回完成优化器更新{counts["optimizer_updates_returned"]}次。',
        '一个前向尚在运行属于快照时刻状态；不能把进入次数都称为已完成。原进程继续，以上均是快照下界。',
        f'较早的每50批进度写入记录：50个microbatch、25次更新，loss={progress.get("last_loss"):.6f}，梯度范数={progress.get("last_gradient_norm_before_clip"):.6f}，峰值allocated={gib:.3f}GiB。',
        '监督器PID17829/start_ticks26463967；B worker PID17830/start_ticks26464019，现场两者身份匹配且存活。',
        'NVIDIA所列PID来自驱动命名空间，不直接当作容器worker PID。训练归属按容器PID/start_ticks、命令和连续事件记录核对。',
        'M、Q未启动，等待B固定12轮最低学习检查；目前没有A57 AP指标或候选有效性结论。',
        '',
        '固定判定与退出',
        'B/M/Q最多各12轮，batch4累积2次，同一seed、epoch顺序、翻转、原生损失与评价。每轮1618前向/809更新。4、8、12轮验证，固定12轮raw判定，无EMA或最好轮次。',
        'B12需AP≥8且APs≥3，否则取消未启动M/Q，记为该短训宿主不足，不能称Q训练阴性。',
        'B通过后，Q12须同时达到：Q−B APs≥+0.5pp、Q−M APs≥+0.3pp、Q−B AP≥−0.2pp。未达标淘汰该具体配方，不追加轮数、seed、模块或阈值挽救。',
        '总72小时、单组32小时硬上限；失败取消未启动组，保留原始材料，后续队列为空。容量粗估约60小时训练，不是完成时间保证。',
        '每组上限19867整模型前向、9708更新；三组上限59601前向、29124更新，不含此前15次容量更新和3次接入前向。',
        '',
        '研究边界',
        '这是候选主线进入短训验证，未证明新颖性、稳定提升、充分收敛或期刊可投。单seed、重复使用公开val、原文全文未取得、README参数差异及跨数据集/部署成本仍须以后补证。',
        'A56具体配方退役裁决保留，不重启；零时间读取、query-C和交错/双向扫描各自均不宣称首次。',
        '',
        '继续核查：只读原PID及start_ticks、supervisor_state.json、arm_B/progress.json和events.jsonl。不要重复运行launch、capacity或本publish脚本。',
    ]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
    old=load(before/'CURRENT_A57.json')
    evidence=dict(old,status='A57_B_RUNNING_FIXED12_RAW',formal_training_launched=True,owned_jobs_live=True,
        counts_terminal=False,formal_model_forwards=counts['entries'],formal_model_forward_completions=counts['completions'],
        formal_detector_optimizer_updates=counts['optimizer_updates_returned'],formal_epochs=counts['epochs_completed'],
        eval_integration_model_forwards=3,eval_integration_optimizer_updates=0,
        formal_queue={'active':'B','conditional_pending':['M','Q'],'condition':'B fixed12 AP>=8 and APs>=3','post_training_queue':None},
        source_bound_protocol=str(P/'protocol_FROZEN_v1.json'),protocol_sha256=sha(P/'protocol_FROZEN_v1.json'),
        live_snapshot=str(dest/'formal_live_snapshot_v1.json'),observed_utc=snap['observation_end_utc'],
        counters_observed_lower_bounds_while_job_live=True,report=str(report),new_A57_AP_metrics=None,
        next='Observe original fixed12 B process; M/Q only if B minimum-learning gate passes. No repeat launch or rescue.')
    replace(P/'CURRENT_A57.json',evidence)
    totals={'model_forwards':226281+3+counts['entries'],'cap':285885,
        'formal_detector_training_updates':194686+counts['optimizer_updates_returned'],
        'capacity_only_optimizer_updates':76,'training_updates':194762+counts['optimizer_updates_returned'],
        'detector_optimizer_updates_including_capacity':194762+counts['optimizer_updates_returned'],
        'formal_training_epochs':240+counts['epochs_completed']}
    now=datetime.now(timezone.utc).isoformat()
    for n in ['01_候选判别表.json','15_当前状态与结果校验.json']:
        d=load(B/n)
        d.update(status=evidence['status'],goal_status='active',mainline_validated=False,
            candidate_selected_for_pilot=True,mainline_selected_for_pilot=False,
            mainline_selected_for_pilot_scope='A57 B12 running; candidate screen selected, paper mainline not validated.',
            latest_report=str(report),latest_update_utc=now,A57_execution=evidence,
            current_host_candidate='Pinned MambaSOD author-code default non-ablation, common compatibility fixes; not full-paper replication',
            next_decision_protocol=str(P/'protocol_FROZEN_v1.json'))
        if n.startswith('01_'):
            for row in d['candidates']:
                if row['id']=='A57_FULL_GRID_SPATIAL_STATE_READOUT':
                    row.update(status=evidence['status'],source_bound_protocol=evidence['source_bound_protocol'],
                               latest_live_snapshot=evidence['live_snapshot'],report=str(report))
        else:
            d.update(totals)
            d.update(recorded_utc=now,current_phase='A57_B_TRAINING_FIXED12_RAW',
                counts_terminal=False,training_counts_terminal=False,current_native_job_terminal=False,
                full_forward_cap_closed=False,counters_observed_lower_bounds_while_job_live=True,
                training_update_scope='Through A57 capacity76 plus observed original B optimizer-returned journal prefix; pending arms not counted.',
                full_model_forwards_scope='Through A57 capacity226281 +3 terminal eval integration + observed B forward entries; lower bound while live. cap285885 is frozen all-arm maximum.',
                current_user_task_status='A57 fresh B12 training running; M/Q conditional on fixed12 learning gate; mainline unresolved',
                previous_goal_turn_classification='progress: A57 native operator/init/capacity and pre-AP design frozen',
                current_goal_turn='progress: exact3eval integration and GT/decode/source binding passed; fresh B12 actually running with real optimizer returns')
            d.setdefault('authoritative_phase_history',[]).append({'phase':'A57','utc':now,'status':evidence['status'],
                'classification':'progress','goal_status':'active','mainline_validated':False,
                'counts_observed_lower_bounds':totals,'source_bound_protocol':evidence['source_bound_protocol'],
                'snapshot':evidence['live_snapshot'],'new_A57_AP_metrics':None})
        replace(B/n,d)
    catalogue=ROOT/'文献总库_20261101_v1/03_索引与核对'
    assert len(load(catalogue/'文献总库.json')['records'])==423
    supplement=catalogue/'后续补链_A57开训与评价口径.json'
    save(supplement,{'phase':'A57','base_catalogue_records_unchanged':423,'new_fulltext_PDFs':0,
        'human_read_attestations_added':False,'mainline_validated':False,
        'literature_scope':str(P/'01_候选问题与最近方法对照.txt'),'execution_report':str(report),
        'source_bound_protocol':str(P/'protocol_FROZEN_v1.json'),'live_snapshot':evidence['live_snapshot']})
    # Prove the older preservation receipt still resolves to its exact original bytes.
    for n,row in receipt['files'].items():
        f=Path(resolution[n]['snapshot']) if n in resolution else Path(n)
        assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256'],n
    save(P/'prior_capacity_receipt_reverified.json',{'status':'PASS_WITH_EXPLICIT_MUTABLE_FILE_SNAPSHOTS',
        'files_verified':len(receipt['files']),'resolution':str(P/'prior_capacity_stage_resolution.json')})
    paths=[f for f in P.rglob('*') if f.is_file() and '__pycache__' not in f.parts]
    paths += [B/'01_候选判别表.json',B/'15_当前状态与结果校验.json',supplement]
    save(P/'A57_formal_start_preservation_receipt.json',{'status':'A57_SOURCE_BOUND_B_RUNNING_PRESERVED',
        'goal_status':'active','observed_counts':totals,'live_prefix_scope':counts,
        'files':{str(f):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(set(paths))},
        'new_detector_forwards_by_publication':0,'new_optimizer_updates_by_publication':0})
    print(json.dumps({'status':evidence['status'],'observed_B_updates':counts['optimizer_updates_returned'],
        'observed_B_forward_entries':counts['entries'],'canonical_observed_totals':totals,
        'report':str(report),'goal_status':'active'},ensure_ascii=False))


if __name__=='__main__':main()
