"""Write acceptance only after each resumed arm saved a fresh complete epoch."""
import datetime
import hashlib
import json
import pathlib
import sys
import tomllib
from zoneinfo import ZoneInfo

here=pathlib.Path(__file__).resolve().parent
root=here.parents[2]
destination=root/'英文小论文/data/06_第二轮_20261014/恢复执行_20261016_v5'
inputs=[pathlib.Path(v) for v in sys.argv[1:]]
assert len(inputs)==2
checks=[json.loads(v.read_text(encoding='utf-8')) for v in inputs]
assert {d['account'] for d in checks}=={'A','B'}
now=datetime.datetime.now(datetime.timezone.utc)
receipt={'verified_utc':now.isoformat(),'verified_los_angeles':now.astimezone(ZoneInfo('America/Los_Angeles')).isoformat(),
         'repair_version':'execution_v5','all_four_jobs_resumed_and_separately_saved':True,
         'frozen_model_and_recipe_unchanged':True,'shared_v4_artifacts_archived':True,
         'first_complete_epoch_results_are_intermediate':True,'accounts':{},'source_sha256':{}}
lines=['保存路径修复与续跑验收', '验收时间（America/Los_Angeles）：'+receipt['verified_los_angeles'],
       '四个续训均已载入原完整状态，完成至少一个新的完整评估并在各自目录保存last.pth。',
       'v3模型源码、配置、初始化和96轮配方不变；v4共写现场保留在v5/before_repair。',
       '原已完成7项不重跑；同卡最多2个训练。', '']
for data,path in zip(checks,inputs):
    account=data['account']
    assert data['all_first_saved_epochs_verified']
    assert data['protected_source_files_verified_after_restart']==(656 if account=='A' else 677)
    launch=json.loads((destination/('launch_'+account+'.json')).read_text(encoding='utf-8'))
    identity=data['controller_identity']
    assert identity and identity['start_ticks']==launch['identity']['start_ticks'] and identity['state'] not in ['T','Z']
    assert len(data['main_trainers'])==2
    for trainer in data['main_trainers']:
        assert trainer['ppid']==identity['pid']
        assert trainer['environment']['OMP_NUM_THREADS']=='4' and trainer['environment']['MKL_NUM_THREADS']=='4'
    assert data['queue_state']['state']=='parallel_training' and not data['queue_state'].get('errors')
    receipt['accounts'][account]=data
    receipt['source_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for job in data['jobs']:
        assert job['first_saved_epoch_verified'] and job['last_checkpoint_verified']
        lines.append(account+' | '+job['job']+' | 已完整评估'+str(job['evaluated_epochs'])+'/96轮'+
                     ' | last_epoch='+str(job['last_checkpoint_epoch_zero_based'])+
                     ' | 独立目录：'+job['native_output_directory'])
config=tomllib.loads(pathlib.Path('C:/Users/jiwei/.codex/automations/automation/automation.toml').read_text(encoding='utf-8'))
assert config['status']=='PAUSED'
receipt['half_hour_monitor_status']='PAUSED'
lines.extend(['','B端rank16_mean_full96_seed2在原队列中等空位，从公共初始化开始，仍按原96轮配方。',
              '各账号训练结束后原队列自动执行既定独立val/testdev评估、D分析，A的平移测试及B的SOEP成本测量。',
              '训练过程中每轮保存；如资源再次终止，按任务各自目录中的完整检查点恢复。',
              '没有全局RNG快照，不声称逐位等同于从未中断的训练。',
              '半小时聊天自动监测保持PAUSED，未重新启用。',
              '本验收只证明恢复和独立保存有效，不是96轮最终结果。'])
with (destination/'修复与续跑验收.json').open('x',encoding='utf-8') as handle:
    json.dump(receipt,handle,ensure_ascii=False,indent=2)
with (destination/'修复与续跑说明.txt').open('x',encoding='utf-8') as handle:
    handle.write('\n'.join(lines)+'\n')
handoff=root/'英文小论文/data/06_第二轮_20261014/每半小时跟进_v4/监测交接_20261015.txt'
with handoff.open('a',encoding='utf-8') as handle:
    handle.write('\n最新恢复验收（'+receipt['verified_los_angeles']+'）：两端已切换execution_v5，四臂首次新完整评估与独立last保存均验证通过。旧v4共写现场已归档保留；详见恢复执行_20261016_v5/修复与续跑验收.json。半小时自动监测保持PAUSED，不自行启用。\n')
state_path=handoff.parent/'notification_state.json'
state=json.loads(state_path.read_text(encoding='utf-8'))
state.update(repair_status='verified_v5_separate_outputs',repair_authorization_received=True,
             repair_verification='../恢复执行_20261016_v5/修复与续跑验收.json',platform_status='connected_gpu_allocated',
             current_execution='execution_v5',automation_status='PAUSED')
temporary=state_path.with_suffix('.tmp');temporary.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(state_path)
print('\n'.join(lines))
