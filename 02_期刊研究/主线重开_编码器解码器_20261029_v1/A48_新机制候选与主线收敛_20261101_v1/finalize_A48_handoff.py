from pathlib import Path
import datetime,hashlib,json

ROOT=Path(__file__).resolve().parent
LIB=ROOT.parents[1]/'文献_相关原始论文_20261004'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':
    previous=json.loads((ROOT/'05_A48启动与历史保留校验回执.json').read_text(encoding='utf-8'))
    for name,h in previous['current_files_sha256'].items():assert sha(name)==h,name
    response=json.loads((ROOT/'remote_status_06.json').read_text(encoding='utf-8'))
    assert response['status']=='EXECUTED' and response['exit']==0
    observation=json.loads(response['stdout']);p=observation['files']['arm_B/progress.json']
    assert observation['frozen_file_checks']['all_pass']
    assert all(v.get('same_start_ticks') and v.get('task_root_in_command') for v in observation['identities'].values())
    counters={'new_model_forward_lower_bound':12+p['full_model_forwards'],
              'new_training_update_lower_bound_including_capacity':9+p['optimizer_updates'],
              'new_formal_training_update_lower_bound':p['optimizer_updates'],'new_capacity_updates':9,
              'cumulative_model_forward_lower_bound':83963+12+p['full_model_forwards'],
              'cumulative_optimizer_update_lower_bound':58793+9+p['optimizer_updates'],
              'cumulative_formal_training_update_lower_bound':58774+p['optimizer_updates'],
              'cumulative_capacity_updates':28,'completed_formal_epoch_sum_lower_bound':72+p['completed_epochs']}
    ledger={'user_date':'2026-11-01','goal_status':'active','mainline_confirmed':False,
       'objective':'继续寻找有证据、有方法区别、可在单卡24GB数天内判别的小论文主线；候选失败后继续换实质不同机制。',
       'skill':'academic-research-suite: settled targeted non-systematic research and experiment-agent code validation; one assistant, no delegation',
       'current_candidate':'A48 训练期每个GT框等量参与编码器BN统计','host':'ECDet-S','dataset':'VisDrone2019-DET',
       'protocol':str(ROOT/'protocol_FROZEN.json'),'protocol_sha256':sha(ROOT/'protocol_FROZEN.json'),
       'remote_root':'/root/uav_a48_20261101_v1','python':'/opt/conda/envs/mamba/bin/python',
       'remote_endpoint':'root@172.17.120.160:30172','password_persisted':False,
       'report':str(ROOT/'01_候选主线与最小验证说明.txt'),'supplement':str(ROOT/'06_补充近邻核查_阅读边界.txt'),
       'baseline_observed_progress':p,'other_formal_arms':'M/Q not started at this observation, queued after B',
       'observed_server_unix':observation['observed_unix'],'counter_status':'lower bounds from saved observation; not current live query',
       'counters':counters,'old_preservation_receipt':str(ROOT/'05_A48启动与历史保留校验回执.json'),
       'next_actions':['Fresh SSH read current supervisor/arm progress and stderr; do not rerun launcher/capacity/trainer.',
          'Keep frozen B/M/Q each12, same init/seed/order/actual inputs/RNG. Check every completed arm result and fixed12EMA.',
          'Compare Q-B APs>=0.5pp, Q-M APs>=0.3pp, Q-B overallAP>=-0.2pp. These are practical pilot gates, not significance/novelty proof.',
          'On any numerical/infrastructure failure stop and preserve. No automatic retries, lambda/site/seed rescue or longer follow-up queue.',
          'On endpoint failure retire this recipe and continue a genuinely different mechanism. A negative candidate alone does not complete broad goal.',
          'On positive pilot assess closest methods incl. weighted BN, CateNorm, CEASC, FilteredBN and missing LearnableBN full method; modest task-specific extension not novel BN formula.',
          'Before changing canonical01/15, preserve their A48 versions referenced by05 receipt and remap signed historical inputs. Do not modify frozen stage sources or old data/weights.'],
       'source_reading_limits':'Some PDFs unavailable; read scopes and source failures explicit. No full novelty search or completed manuscript claim.'}
    save(ROOT/'00_A48目标与运行账本.json',ledger)
    note=['2026-11-01｜A48最新保存观察；目标active，主线尚未确认。',
      '候选问题：按GT实例均衡训练归一化统计，能否比普通BN及框像素加权提高小目标检测？',
      'VisDrone + ECDet-S；三组各12轮，固定第12轮EMA；不追加新方向/调参训练队列。',
      '原版B：已完成%d轮，当前第%d轮第%d/809批，%d次正式更新。'%(p['completed_epochs'],p['epoch'],p['microbatch'],p['optimizer_updates']),
      'M/Q：正式训练尚未开始，按已冻结监督器依次执行；三组容量检查已通过。',
      '本快照B loss=%s，peak allocated=%.3fGiB；进程PID/start_ticks符合启动记录，stderr为空。'%(p['loss'],p['peak_allocated_bytes']/1024**3),
      '13项冻结输入SHA核查通过；训练/验证全部原始图像和标注哈希已通过。',
      '当前无候选同轮AP差值，不能判断有效。下一轮只看真实完成轮次，最终按12轮端点判别。',
      '计数：'+json.dumps(counters,ensure_ascii=False),
      '保存观察对应远端unix='+str(observation['observed_unix'])+'，不替代后续实时查询。',
      '报告：01_候选主线与最小验证说明.txt；追加近邻：06_补充近邻核查_阅读边界.txt；续接：00_A48目标与运行账本.json。']
    (ROOT/'07_最新观察与继续入口.txt').write_text('\n'.join(note)+'\n',encoding='utf-8')
    registry=[]
    for name,scope in [
       ('A48_WBN_Yasuda_2020.pdf','PDF §4.1 and §4.2 discussion; not full'),
       ('A48_CateNorm_2022.pdf','PDF §3, equations1-5 and train/test flow; not full'),
       ('65_FlowDet_CVPRW2026_LoViF.pdf','PDF §3.1–3.4, prior and pairing ablations; not full'),
       ('SOD_2026_PR_TOLF.pdf','abstract/introduction only this stage'),
       ('A48_Filtered_Batch_Normalization_ICPR2020.pdf','PDF archived; methods read via same-version arXiv HTML §III eq4-13, not full PDF')]:
        path=LIB/name
        if path.is_file():registry.append({'path':str(path),'sha256':sha(path),'reading_scope':scope})
    save(LIB/'A48_research_reading_registry_v1.json',{'user_date':'2026-11-01','records':registry,
          'failed_downloads':str(ROOT/'source_acquisition_v1.json'),'additional_downloads':str(ROOT/'source_acquisition_supplement_v1.json')})
    files=['00_A48目标与运行账本.json','07_最新观察与继续入口.txt','06_补充近邻核查_阅读边界.txt',
           'remote_status_06.json','preparation_result.json','capacity_result.json','B_initialization_observed.json',
           'source_acquisition_v1.json','source_acquisition_supplement_v1.json','search_raw_records_supplement_v1.json']
    result={'status':'PASS','goal_status':'active','mainline_confirmed':False,'frozen_protocol_unchanged':True,
          'previous_receipt_sha256':sha(ROOT/'05_A48启动与历史保留校验回执.json'),
          'previous_signed_current_files_reverified':len(previous['current_files_sha256']),
          'older_preserved_inputs_previously_verified':227,'counters':counters,
          'files_sha256':{f:sha(ROOT/f) for f in files}}
    save(ROOT/'08_A48本轮交付校验.json',result)
    print(json.dumps({'status':'PASS','goal_status':'active','mainline_confirmed':False,'baseline_progress':p,'counters':counters}))
