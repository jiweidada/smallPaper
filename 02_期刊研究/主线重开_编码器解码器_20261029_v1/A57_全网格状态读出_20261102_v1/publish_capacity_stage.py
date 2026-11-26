"""Publish verified A57 engineering evidence without claiming a research success."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import shutil

P=Path(__file__).resolve().parent
B=P.parent
ROOT=B.parent.parent


def load(p):return json.loads(p.read_text('utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for s in iter(lambda:f.read(4<<20),b''):h.update(s)
    return h.hexdigest()


def save(p,d,mode='x'):
    with p.open(mode,encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    prior_receipt=load(B/'A56_尺度内对比读取_20261102_v1/A56_terminal_preservation_receipt.json')
    assert all(sha(Path(p))==r['sha256'] for p,r in prior_receipt['files'].items())
    audit=load(P/'capacity_terminal_input_audit.json')
    assert audit['status']=='PASS' and all(not r['owned_live'] for r in audit['owned_jobs'])
    assert audit['whole_detector_forwards']==audit['capacity_only_updates']==15
    assert load(P/'common_initialization_local_verification.json')['status']=='PASS_LOCAL_INITIALIZATION_TENSORS'
    c1=load(P/'capacity_result.json')
    c4=load(P/'capacity_batch4_result.json')
    design=load(P/'pilot_design_FROZEN_v1.json')
    assert c1['counts']['whole_model_forward_completions']==6
    assert c4['counts']['whole_model_forward_completions']==9
    assert c1['counts']['capacity_only_optimizer_updates']+c4['counts']['capacity_only_optimizer_updates']==15
    assert c4['rough_three_arm_training_hours']<=72
    assert design['formal_training_launched'] is False
    for prefix in ['capacity_','capacity_batch4_']:
        plan=load(P/(prefix+'scope_FROZEN_v1.json'))
        assert all(sha(P/n)==h for n,h in plan['files_sha256'].items())
        result=load(P/(prefix+'result.json'))
        for r in result['records']:
            assert r['initial_state_exact'] and r['weights_discarded']
        for step in range(len(result['records'][0]['steps'])):
            assert len({json.dumps(r['steps'][step]['RNG_before_forward'],sort_keys=True) for r in result['records']})==1
    report=P/'02_真实容量结果与12轮设计.txt'
    lines=[
        'A57：真实容量结果与下一次最小验证设计',
        '2026-11-02。源码、初始化、算子和容量阶段已完成；正式12轮训练尚未启动。总目标active，主线和新颖性均未验证。',
        '',
        '当前已收敛的候选问题：在完整context状态和相同reference位置下，query条件状态读出能否比普通SSM编码后采样提供额外的小目标检测价值？',
        '数据：VisDrone2019-DET。模型：固定MambaSOD作者源码版与单独保存的共同兼容修复，实际72697131参数。不是已复现原文7490万参数配置或发表数字。',
        'B保留原MQI压缩/对齐读出；M完整context输出后采样；Q完整context上的query-C状态读出。M/Q空间位置、状态推进及参数拓扑匹配。具体先例、差异与范围见01文件。',
        '',
        '实际验证事实',
        'CPU算子v1的argsort绑定错误保留，v2仅修复接口。累计21次进入、2次反向、0次组件更新；CUDA组件4次前向、2次反向。它们不是检测器训练或任务收益证明。',
        '官方Vim Tiny初始化在本地下载115013118字节，LFS SHA通过后上传；413个预训练tensor映射，位置编码保留中间CLS后由14×14插到40×40。',
        '公共初始化完整下载并在本地核对2011个tensor及digest，33个BN计数均为0，浮点tensor有限；B/M/Q使用同一初值，未使用容量更新后的权重。',
        'microbatch1：三组各2次完整前向/更新，均有限，Q峰值约5.64GiB，但粗估三组12轮85.40小时。该设置不作为正式训练配方。',
        '在任何新AP或正式epoch之前，做了一次预先固定的batch4利用率核查：三组各3批，含第一seed批、902-GT所在批、另一最大GT总数批。没有改变模型/组数/query数/算法或72小时预算；没有继续搜索其他batch。',
        '',
        'batch4结果（训练态、group6、6层decoder同样checkpoint，BF16前向/FP32原criterion，全部参数可学习）',
        '组 | 峰值allocated GiB | 两个热后压力批中较慢步秒数 | 12轮训练粗估小时',
    ]
    for r in c4['records']:
        peak=max(s['peak_allocated_bytes'] for s in r['steps'])/(1<<30)
        seconds=max(s['seconds_forward_loss_backward_step'] for s in r['steps'][1:])
        hours=c4['rough_training_only_hours_per12_from_single_postwarmup_stress_step'][r['arm']]
        lines.append(f'{r["arm"]} | {peak:.3f} | {seconds:.3f} | {hours:.2f}')
    lines += [
        '三组约60.00小时，低于预定72小时的训练粗估预算。此估计不含全部数据加载、每四轮验证、存盘和传输；只来自两个热后压力批的较慢值，不能当稳定吞吐量基准、完成时间保证或300轮成本证明。',
        'Q比简单M需要更多显存与计算；若AP只有微小收益，未必值得论文投入，不能宣称本方法更快或轻量。',
        '容量两阶段共15次整模型前向、15次临时优化器更新；正式训练0轮，AP评价0次。6+9次均完整完成，loss/梯度有限，输入与前向前RNG核对通过；所有临时权重丢弃。',
        '两份监督器与worker均退出0、无超时，按PID/start_ticks核实不存活。源码冻结SHA与公共初始化SHA在终点仍匹配，后续队列为空。',
        '',
        '下一次最小验证设计已在任何新AP之前冻结',
        'B→M→Q各最多12轮。batch4累积2个microbatch，每轮809次更新，最后一次7张图；同初始权重、seed20261101、图像次序、flip、损失与COCO风格ignore口径。',
        '固定第12轮raw权重为主终点，4/8轮只观察；不选最好轮次。统一12轮短日程、全参数训练；这不是原文300轮日程复现。',
        'B12先满足AP≥8、APs≥3的最低学习检查，否则取消未启动的M/Q，记为该短训host不足，不称Q训练阴性。',
        '若B满足学习检查，Q12同时需：Q−B APs≥+0.5pp、Q−M APs≥+0.3pp、Q−B AP≥−0.2pp。门槛是资源决策，非统计显著性或期刊验收。',
        '设计含总72小时、单组32小时上限；失败取消未启动组。无追加seed、层/点/增益/阈值搜索或续轮挽救；不重启A22/A56。',
        '还需把训练/评价实现绑定进可执行协议，完成预分配的3次eval-group1/后处理接入检查及COCO GT一致性核对，检查存储和进程后才启动。当前没有待用户确认项；这是剩余实现工作。',
        '',
        '研究边界',
        '原文/代码支持的是结构事实和最近先例；“空间压缩限制细小目标证据”仍是待验证假设。零时间读出代数、query-C、双向/交错扫描均已有基础，不单独声称新颖。',
        '单seed短训、公用val多次开发、原文主文未读、实际参数量差异，以及后续充分收敛和跨数据集成本均保留风险。容量通过不是找到可发表创新。',
        '',
        '文件：pilot_design_FROZEN_v1.json、capacity_result.json、capacity_batch4_result.json、capacity_terminal_input_audit.json、common_initialization_local_verification.json。',
    ]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    before=P/'before_A57'
    before.mkdir(exist_ok=False)
    names=['01_候选判别表.json','15_当前状态与结果校验.json']
    for n in names:shutil.copy2(B/n,before/n)
    save(P/'prior_canonical_resolution.json',{'reason':'A56 terminal receipt resolves its old canonical bytes through before_A57 after mutable board advances.',
              'files':{str(B/n):{'snapshot':str(before/n),'sha256':sha(before/n)} for n in names}})
    evidence={'phase':'A57','status':'A57_CAPACITY_VERIFIED_DESIGN_FROZEN_IMPLEMENTATION_PENDING',
              'goal_status':'active','mainline_validated':False,'novelty_cleared':False,
              'candidate_selected_for_pilot':True,'formal_training_launched':False,'formal_queue':None,
              'owned_jobs_live':False,'capacity_terminal':True,'counts_terminal':True,
              'capacity_model_forwards':15,'capacity_detector_optimizer_updates':15,
              'formal_model_forwards':0,'formal_detector_optimizer_updates':0,'formal_epochs':0,
              'CPU_operator_entries_including_failed_API_call':21,'CPU_synthetic_backward_calls':2,
              'GPU_operator_forwards':4,'GPU_synthetic_backward_calls':2,'synthetic_optimizer_updates':0,
              'capacity_decoder_layer_entries_including_checkpoint_recomputation':180,
              'model_parameters':72697131,'common_initialization_sha256':design['initialization_sha256'],
              'pilot_design':str(P/'pilot_design_FROZEN_v1.json'),'report':str(report),
              'rough_three12_training_hours':c4['rough_three_arm_training_hours'],
              'new_A57_AP_metrics':None,'prior_A56_reopened':False,
              'next':'Implement and source-bind the frozen design, exactly3eval integration forwards, identicalGT check, then launch fresh B12; M/Q depend on B12 learning gate.'}
    save(P/'CURRENT_A57.json',evidence)
    for n in names:
        d=load(B/n)
        d.update({'status':evidence['status'],'goal_status':'active','mainline_validated':False,
                  'current_priority_candidate':'A57_FULL_GRID_SPATIAL_STATE_READOUT',
                  'candidate_selected_for_pilot':True,'mainline_selected_for_pilot':False,
                  'mainline_selected_for_pilot_scope':'A57 bounded candidate/design selected; no confirmed paper mainline or AP result.',
                  'current_priority_question':design['question'],'latest_report':str(report),'A57_execution':evidence})
        if n.startswith('01_'):
            assert not any(r['id']=='A57_FULL_GRID_SPATIAL_STATE_READOUT' for r in d['candidates'])
            d['candidates'].append({'id':'A57_FULL_GRID_SPATIAL_STATE_READOUT','name':'全网格状态的空间query条件读出',
              'status':evidence['status'],'selected_for_pilot':True,'mainline_validated':False,'novelty_cleared':False,
              'old_references':['D008','D025','A22/N','A29/P','A42','A43','A44','A51','A54','A56'],
              'distinct_increment':'Read a context-only full-grid selective state at shared geometric query references rather than align context length to query count; compare context-generated versus query-generatedC, not bare SSM replacement.',
              'new_basis':'Pinned native source, bounded nearest-method comparison, exact operator/gradient validation, matched real source-native capacity; no AP or task-causal claim yet.',
              'strong_control':'Full-grid context-conditioned SSM output then identical geometric sampling',
              'pilot_design':str(P/'pilot_design_FROZEN_v1.json'),'report':str(report)})
        else:
            prior=load(before/n)
            assert prior['model_forwards']==226266 and prior['training_updates']==194747
            d.update({'phase':'A57','current_phase':'A57_CAPACITY_TERMINAL_DESIGN_PENDING_EXECUTION',
              'latest_authoritative_phase':'A57','model_forwards':226281,'cap':226281,
              'formal_detector_training_updates':194686,'capacity_only_optimizer_updates':76,
              'training_updates':194762,'detector_optimizer_updates_including_capacity':194762,
              'formal_training_epochs':240,'counts_terminal':True,'training_counts_terminal':True,
              'current_native_job_terminal':True,'full_forward_cap_closed':True,
              'counters_observed_lower_bounds_while_job_live':False,
              'full_model_forwards_scope':'Exact226266 throughA56 plusA57 capacity6+9=226281; synthetic operator calls excluded; no new formal training.',
              'current_user_task_status':'A57 candidate ready for frozen-design implementation; mainline unresolved',
              'previous_goal_turn_classification':'progress: A56 endpoint verified and retired, MambaSOD source access corrected',
              'current_goal_turn':'progress: source/operator/native initialization and24GB capacity verified; bounded next design frozen before anyA57 AP'})
        save(B/n,d,mode='w')
    supplement=ROOT/'文献总库_20261101_v1/03_索引与核对/后续补链_A57状态读出候选与验证.json'
    assert len(load(ROOT/'文献总库_20261101_v1/03_索引与核对/文献总库.json')['records'])==423
    save(supplement,{'phase':'A57','base_catalogue_records_unchanged':423,'new_fulltext_PDFs':0,
                    'literature_and_scope':str(P/'01_候选问题与最近方法对照.txt'),
                    'research_report':str(report),'design':str(P/'pilot_design_FROZEN_v1.json'),
                    'mainline_validated':False,'human_read_attestations_added':False})
    paths=[p for p in P.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.json','.txt','.log','.pth','.zip']]
    paths += [B/n for n in names]+[supplement]
    paths=sorted(set(paths))
    receipt={'status':'A57_CAPACITY_AND_DESIGN_PRESERVED','goal_status':'active',
             'counts':evidence,'files':{str(p):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths},
             'new_detector_forwards_by_publication':0,'new_optimizer_updates_by_publication':0}
    save(P/'A57_capacity_stage_preservation_receipt.json',receipt)
    print(json.dumps({'status':evidence['status'],'files_hashed':len(paths),'model_forwards_total':226281,
                      'A57_capacity_updates':15,'new_formal_training_epochs':0,'report':str(report)},ensure_ascii=False))


if __name__=='__main__':main()
