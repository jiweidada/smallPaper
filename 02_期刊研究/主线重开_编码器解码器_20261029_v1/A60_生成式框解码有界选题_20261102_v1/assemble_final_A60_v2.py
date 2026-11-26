"""Consolidate original completed A60 artifacts, without model execution."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import shutil

P=Path(__file__).resolve().parent
OUT=P/'本轮A60实验汇总_20261103_v1'
POLICY=P/'USER_FINISH_CURRENT_A60_THEN_PAUSE_v1.json'
FIELDS=['AP','AP50','AP75','APs','APs75','APm','APl','ARs']

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):d.update(b)
    return d.hexdigest()
def write_json(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    assert not OUT.exists()
    policy=read(POLICY)
    assert policy['user_words_verbatim']=='跑完实验就先整理暂停吧'
    assert not policy['new_candidates_after_current_batch_authorized']
    protocol=read(P/'protocol_FROZEN_v1.json')
    assert protocol['epochs_per_arm']==12 and protocol['evaluation_epochs']==[4,8,12]
    expected={'started':78514,'completed':78514,'updates':9708,'backwards':77652,
              'epochs':12,'train_forwards':77652,'eval_forwards':822,'latency_forwards':40}
    metrics,results,verifications,curves,weights,assets={},{},{},[],[],[]
    for arm in ['B','M','Q']:
        vf=P/('B_fixed12_local_verification_v1.json' if arm=='B' else arm+'_fixed12_local_verification_v1.json')
        v=read(vf)
        assert v['status']=='PASS_ORIGINAL_'+arm+'_FIXED12_ARTIFACTS_AND_LOCAL_MODEL_BACKUP'
        raw=P/('fixed12_'+arm+'_remote_v1')/arm
        result=read(raw/'result.json')
        assert result['status']=='COMPLETED_FIXED12_RAW' and result['arm']==arm
        assert result['counts']==v['original_counts']==expected and result['training_epochs']==12
        assert result['metrics']==v['metrics']==read(raw/'epoch12_raw_metrics.json')
        assert result['protocol_sha256']==sha(P/'protocol_FROZEN_v1.json')
        assert result['source_sha256']==protocol['source_sha256'] and result['no_best_epoch_selection']
        cp=Path(v['checkpoint_path']);assert cp.is_file() and cp.stat().st_size>400_000_000
        assert sha(cp)==v['checkpoint_sha256']==result['fixed12_checkpoint_sha256']
        weights.append({'arm':arm,'path':str(cp),'bytes':cp.stat().st_size,'sha256':v['checkpoint_sha256'],
                        'model_state':'fixed epoch12 raw, no EMA/best selection'})
        metrics[arm]=result['metrics'];results[arm]=result;verifications[arm]=str(vf)
        for epoch in range(1,13):
            row=read(raw/f'epoch{epoch:02d}.json')
            assert row['epoch']==epoch and row['updates']==809*epoch and math.isfinite(row['mean_loss'])
            curves.append({'arm':arm,'epoch':epoch,'mean_training_loss':row['mean_loss'],
                           'train_seconds':row['train_seconds'],'metrics':row.get('metrics')})
        assets.extend([{'kind':'verified_raw_results','arm':arm,'path':str(raw)},
                       {'kind':'local_verification','arm':arm,'path':str(vf),'sha256':sha(vf)}])
    qroot=P/'fixed12_Q_remote_v1'
    judgment=read(qroot/'fixed12_judgment_v1.json');supervisor=read(qroot/'supervisor_exit_v1.json')
    assert supervisor['status']=='ALL_THREE_FIXED12_COMPLETED' and supervisor['exit']==0
    assert supervisor['completed_arms']==['B','M','Q'] and supervisor['remaining_queue']==[]
    qv=read(Path(verifications['Q']))
    assert qv['judgment']==judgment and judgment['metrics']==metrics and judgment['full_input_stream_audit_passed']
    delta={'Q_minus_B_APs':metrics['Q']['APs']-metrics['B']['APs'],
           'Q_minus_M_APs':metrics['Q']['APs']-metrics['M']['APs'],
           'Q_minus_B_AP':metrics['Q']['AP']-metrics['B']['AP'],
           'Q_over_M_median_latency':results['Q']['latency']['median_ms']/results['M']['latency']['median_ms']}
    assert all(math.isclose(delta[k],judgment['delta'][k],rel_tol=0,abs_tol=1e-12) for k in delta)
    gates={'APs_vs_B':delta['Q_minus_B_APs']>=.5,'APs_vs_M':delta['Q_minus_M_APs']>=.3,
           'AP_preservation':delta['Q_minus_B_AP']>=-.2,'latency':delta['Q_over_M_median_latency']<=1.10}
    assert gates==judgment['gates'] and all(gates.values())==judgment['candidate_has_pilot_support']
    assert not judgment['mainline_validated']
    evaluated=[x for x in curves if x['metrics'] is not None]
    assert [(x['arm'],x['epoch']) for x in evaluated]==[(a,e) for a in ['B','M','Q'] for e in [4,8,12]]
    paths={**protocol['source_sha256']}
    for name,expected_sha in paths.items():assert sha(P/name)==expected_sha
    common_init=P/'CPU_constructor_stage_v1/common_init_A60_v1.pth'
    assert common_init.is_file() and sha(common_init)==protocol['initialization_sha256']
    dataset_manifest=P/'dataset_manifest_v1.json'
    assert sha(dataset_manifest)==protocol['dataset_manifest_sha256']
    environment=read(P/'environment_snapshot_v2.json')
    wheel=Path(environment['isolated_pycocotools']['local_wheel'])
    assert sha(wheel)==environment['isolated_pycocotools']['wheel_sha256']
    copy_bytes=sum(x['bytes'] for x in weights)+common_init.stat().st_size
    copy_bytes+=sum(x.stat().st_size for a in ['B','M','Q'] for x in (P/('fixed12_'+a+'_remote_v1')).rglob('*') if x.is_file())
    assert shutil.disk_usage(P).free>copy_bytes+(2<<30)
    summary={'status':'ALL_ORIGINAL_A60_FIXED12_VERIFIED_AND_ORGANIZED',
        'recorded_client_utc':datetime.now(timezone.utc).isoformat(),
        'user_instruction':policy,'goal_pause_requested':True,'goal_pause_ready':True,
        'thread_goal_status_before_pause_tool':'active','mainline_validated':False,
        'candidate_has_pilot_support':all(gates.values()),'protocol':protocol,
        'fixed12_metrics':metrics,'fixed12_delta':delta,'fixed12_gates':gates,
        'all_12_epoch_training_records_and_4_8_12_metrics':curves,
        'all_metric_differences_Q_minus_B_pp':{k:metrics['Q'][k]-metrics['B'][k] for k in FIELDS},
        'all_metric_differences_Q_minus_M_pp':{k:metrics['Q'][k]-metrics['M'][k] for k in FIELDS},
        'resources':{a:{'parameters':protocol['trainable_parameters'],'elapsed_seconds':results[a]['elapsed_seconds'],
            'peak_allocated_bytes':results[a]['peak_allocated_bytes'],'peak_reserved_bytes':results[a]['peak_reserved_bytes'],
            'latency':results[a]['latency'],'formal_counts':results[a]['counts']} for a in ['B','M','Q']},
        'formal_counts_total':{k:sum(results[a]['counts'][k] for a in ['B','M','Q']) for k in expected},
        'raw_verification_reports':verifications,'local_model_backups':weights,
        'common_initialization':{'source':str(common_init),'sha256':protocol['initialization_sha256']},
        'environment_snapshot':environment,
        'dataset_manifest':{'source':str(dataset_manifest),'sha256':protocol['dataset_manifest_sha256']},
        'original_supervisor_exit':supervisor,'original_fixed12_judgment':judgment,
        'new_model_forwards_by_organization':0,'new_optimizer_updates_by_organization':0,
        'new_COCO_evaluations_by_organization':0,'checkpoint_loads_by_organization':0,
        'limits':['Single short exploratory seed, not convergence or statistical significance.',
          'COCO-style internal metrics, not official VisDrone challenge scores.',
          'Controlled R50-FPN/DynamicHead implementation, not full author FlowDet reproduction.',
          'CEOSOD full method remains unread; no global novelty clearance.',
          'Q-M also changes within-image time-label synchrony/distribution; no unique geometric causal claim.',
          'Latency excludes image decoding and postprocessing; no onboard deployment-speed claim.']}
    OUT.mkdir();code=OUT/'03_代码与冻结协议';code.mkdir()
    models=OUT/'04_最终模型与共同初始化';models.mkdir()
    raw_out=OUT/'05_原始结果与核验';raw_out.mkdir()
    for item in weights:
        dest=models/(item['arm']+'_epoch12_raw.pth')
        shutil.copy2(Path(item['path']),dest)
        assert dest.stat().st_size==item['bytes'] and sha(dest)==item['sha256']
        item['organized_copy_path']=str(dest)
        print(json.dumps({'copied_model':item['arm'],'bytes':item['bytes']}),flush=True)
    init_dest=models/'common_init_A60_v1.pth';shutil.copy2(common_init,init_dest)
    assert sha(init_dest)==protocol['initialization_sha256']
    summary['common_initialization']['organized_copy_path']=str(init_dest)
    for arm in ['B','M','Q']:
        source=P/('fixed12_'+arm+'_remote_v1');dest=raw_out/arm
        shutil.copytree(source,dest)
        original_manifest=read(source/'manifest.json')
        for file in dest.rglob('*'):
            if not file.is_file():continue
            rel=file.relative_to(dest).as_posix()
            expected=original_manifest[rel] if rel!='manifest.json' else {'bytes':(source/rel).stat().st_size,'sha256':sha(source/rel)}
            assert file.stat().st_size==expected['bytes'] and sha(file)==expected['sha256']
        shutil.copy2(Path(verifications[arm]),raw_out/Path(verifications[arm]).name)
        assets.append({'kind':'organized_raw_archive_copy','arm':arm,'path':str(dest),'own_result':str(dest/arm/'result.json')})
    wheels=code/'wheels';wheels.mkdir();shutil.copy2(wheel,wheels/wheel.name)
    assert sha(wheels/wheel.name)==environment['isolated_pycocotools']['wheel_sha256']
    for source in [dataset_manifest,P/'environment_snapshot_v2.json']:
        shutil.copy2(source,code/source.name)
    shutil.copy2(P/'CPU_constructor_stage_v1/result.json',code/'common_initialization_provenance.json')
    env_note='环境版本和原训练路径均已记录；没有打包整个conda环境，也没有在新环境运行复现。\n源码保留原实验的绝对路径；恢复到新实例时需按实际位置配置runtime中的TRAIN/VAL及评价pydeps路径。\n共同初始化和三个固定12轮模型在04目录，原始评价和核验记录在05目录。\n训练图像未重复复制进本汇总包；数据身份及原路径见dataset_manifest_v1.json。\n'
    with (code/'环境与路径说明.txt').open('x',encoding='utf-8') as f:f.write(env_note)
    for name in paths:shutil.copy2(P/name,code/name)
    for name in ['protocol_FROZEN_v1.json',POLICY.name,'head_source_derivation_v1.json','criterion_source_derivation_v2.json']:
        src=P/name
        if src.is_file():shutil.copy2(src,code/name)
    write_json(OUT/'01_三组完整指标与判据.json',summary)
    assets.extend([{'kind':'model_checkpoint',**x} for x in weights])
    assets.extend([{'kind':'research_report','path':str(P/'03_候选问题_最近方法_最小验证.txt')},
                   {'kind':'nearest_prior_scope','path':str(P/'16_生成式检测近作与创新边界.txt')},
                   {'kind':'CEOSOD_public_scope','path':str(P/'02d_CEOSOD公开材料与条件归因边界_v1.json')},
                   {'kind':'terminal_archive_contract','path':str(P/'MQ_fixed12_archive_scope_FROZEN_v1.json')}])
    write_json(OUT/'02_文件索引与本地权重位置.json',{'weights_copied_again':True,'common_initialization_copied':True,'source_checkpoints_retained':True,'assets':assets,
        'scope':'Existing original assets retained in place; frozen source and protocol copied into this summary folder.'})
    def table(rows):
        out=['组别/轮数 | '+' | '.join(FIELDS)]
        for arm,epoch,values in rows:out.append(f'{arm}/{epoch} | '+' | '.join(f'{values[k]:.6f}' for k in FIELDS))
        return '\n'.join(out)
    verdict='三组固定12轮短测通过预定筛选门槛。用户要求在本批整理后暂停，未追加确认实验。' if all(gates.values()) else '候选未同时达到预定筛选门槛。本批结束，按用户要求暂停，保留全部结果。'
    lines=['## Material Passport','Origin Skill: academic-research-suite / experiment-agent',
        'Verification Status: ANALYZED；固定12轮保存数组/事件/哈希核验，4/8轮取原训练JSON记录；未重跑训练或独立模型推理。',
        'Version: A60_final_fixed12_v2','','本轮A60实验结论与暂停安排','',verdict,'',
        '一、研究问题与三组定义',
        'VisDrone航拍小目标生成式框解码：相同尺寸偏采样扰动框下，几何进度条件能否超过物理时间条件？',
        'B：线性框插值+物理时间；M：尺度相关框插值+物理时间alpha；Q：与M相同扰动框+归一化几何进度u。',
        '模型为受控R50-FPN+六阶段DiffusionDet DynamicHead，部署只调用一次完整decoder。候选是训练条件改动。','',
        '二、固定12轮终点（百分制；未选best；raw模型）',
        table([(a,12,metrics[a]) for a in ['B','M','Q']]),'',
        '三、全部预定4/8/12轮检测结果',table([(x['arm'],x['epoch'],x['metrics']) for x in evaluated]),'',
        '四、固定判据（差值为绝对百分点）',
        f'Q-B APs={delta["Q_minus_B_APs"]:+.6f}pp，要求>=+0.5pp，结果={gates["APs_vs_B"]}。',
        f'Q-M APs={delta["Q_minus_M_APs"]:+.6f}pp，要求>=+0.3pp，结果={gates["APs_vs_M"]}。',
        f'Q-B AP={delta["Q_minus_B_AP"]:+.6f}pp，要求>=-0.2pp，结果={gates["AP_preservation"]}。',
        f'Q/M中位模型时延比={delta["Q_over_M_median_latency"]:.6f}，要求<=1.10，结果={gates["latency"]}。','',
        '五、数据、训练、资源与比较口径',
        'VisDrone2019-DET，6471训练/548验证，10类，640x640拉伸，共同seed20261102。',
        'microbatch1、累积8；每轮809更新，最后7张按实际组大小平均。三组各12轮/9708更新。',
        '内部COCO-style、maxDets300、原图裁剪GT面积small<=1024、class0 union-IoA过滤；不是官方VisDrone挑战指标。',
        '三组初始化及12轮原训练raw-stream摘要一致；M/Q扰动框摘要一致、条件摘要不同；原始评价GT/先验哈希相同。',
        '逐图独立重放/重新COCO匹配没有运行；全部指标及资源明细见01文件。']
    for a in ['B','M','Q']:
        z=results[a];lines.append(f'{a}：参数{protocol["trainable_parameters"]}，原任务耗时{z["elapsed_seconds"]/3600:.4f}小时，峰值allocated={z["peak_allocated_bytes"]/(1<<30):.4f}GiB，模型中位前向={z["latency"]["median_ms"]:.6f}ms。')
    lines+=['时延是同卡batch1 eager全模型，BF16编码器/FP32头，排除图像解码与后处理；不代表机载速度。','',
        '六、结论边界与待核问题',*summary['limits'],
        '原文/代码阅读范围、最近方法逐项差异与推断在02索引指向的调研文件中保留。几何暴露诊断不等于漏检根因。','',
        '七、文件与暂停',
        '01文件含三组全部12轮loss记录、4/8/12轮检测指标、终点差值/门槛与资源；loss不作为跨组检测效果结论。',
        '02文件索引原始资产和整理副本；03目录保存冻结代码、协议、数据清单、环境版本和评价wheel。',
        '04目录保存B/M/Q固定12轮raw模型及共同初始化，均与原件SHA相同；05目录保存三组原始记录与核验。',
        '原源码、旧失败、原始日志、指标、权重均保留原位置；本次整理未新增模型前向或优化器更新。',
        '用户指令：'+policy['user_words_verbatim'],
        '本批后续训练队列为空。整理完成后按用户要求暂停线程研究目标；未宣称主线已经完整验证或论文创新已经成立。']
    with (OUT/'00_本轮结论与暂停说明.txt').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
    manifest={str(q.relative_to(OUT)).replace('\\','/'):{'bytes':q.stat().st_size,'sha256':sha(q)}
              for q in OUT.rglob('*') if q.is_file()}
    write_json(OUT/'06_汇总文件SHA256清单.json',manifest)
    print(json.dumps({'status':summary['status'],'candidate_has_pilot_support':all(gates.values()),
        'pause_ready':True,'output':str(OUT),'files':len(manifest)+1,'local_models_verified':3},ensure_ascii=False))

if __name__=='__main__':main()
