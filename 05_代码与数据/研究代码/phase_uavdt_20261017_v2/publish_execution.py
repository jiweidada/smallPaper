"""Archive prior planning state, then publish actual acceptance and launch evidence."""
import datetime,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]/'data/07_第二数据集_20261017';E=BASE/'执行_v2'
now=datetime.datetime.now(datetime.timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%S_%fZ')
proof=json.loads((E/'dual_final_acceptance.json').read_text())
snap=json.loads((BASE/'每半小时跟进_v2/latest_snapshot.json').read_text(encoding='utf-8'))
launch_path=next(E.glob('*_six_formal_jobs_launch.json'));raw=json.loads(launch_path.read_text(encoding='utf-8'))
launches={a:json.loads(r['stdout']) for a,r in raw['results'].items()};assert all(r['returncode']==0 for r in raw['results'].values())
capacity_raw=json.loads(sorted(E.glob('*_compact_preflight_status.json'))[-1].read_text(encoding='utf-8'))
capacity={a:json.loads(r['stdout'])['capacity'] for a,r in capacity_raw['results'].items()}
manifest_path=BASE/'UAVDT_96轮待执行队列_20261017_v2.json';policy_path=BASE/'第二数据集选择状态.json'
plan_path=HERE.parents[1]/'data/00_实验计划与证据索引/UAVDT第二数据集96轮实验计划_20261017_v2.txt'
for path in [manifest_path,policy_path,plan_path]:shutil.copy2(path,E/(stamp+'_before_release_'+path.name))
manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
manifest.update(state='running',queue_is_executable=True,deployed=True,new_training_started=True,remaining_technical_gates=[],blocked_reason=None,
                entry_command='各账号唯一控制器：<python> -u <root>/code/run_uavdt_queue.py；已启动，禁止重复执行。',updated_utc=now.isoformat(),
                acceptance=str(E/'dual_final_acceptance.json'),launch_evidence=str(launch_path),monitor_entry=str(HERE/'monitor_training.py'))
manifest['initialization']['dataset_common_sha256']=proof['A']['initialization_sha256']
manifest['historical_cost_projection_before_capacity_test']=manifest.pop('cost_projection')
manifest['cost_projection']={'measured_short_real_batches':True,'concurrent_jobs_per_card':3,
 'account_step_only_training_wall_hours':{a:p['per_arm_step_projection_train_hours'] for a,p in proof.items()},
 'caveat':'40 timed regular batches per worker; excludes full augmentation variation, validation, final independent evaluations and analysis. Not a guaranteed completion time. Concurrent job-hours are not physical GPU-card-hours.'}
for a,config in manifest['accounts'].items():
 config.update(maximum_concurrent_trainings=3,planned_launch_batches=[['c','c_mean','baseline']],controller=launches[a],acceptance=proof[a])
 for job in config['jobs']:
  live=next(j for j in snap['accounts'][a]['jobs'] if j['job']==job['job_id'])
  assert live['process']['alive'] and live['process']['identity_matches'] and live['output_binding_passed']
  job.update(state=live['status']['state'],pid=live['process']['pid'],start_ticks=live['process']['start_ticks'])
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
policy=json.loads(policy_path.read_text(encoding='utf-8'))
policy.update(state='UAVDT_six_jobs_running',allow_automatic_training_start=True,UAVDT_training_adapter_ready=True,preparation_in_progress=False,
 technical_release_complete=True,updated_utc=now.isoformat(),new_training_started=True,launched_jobs_total=6,
 max_concurrent_trainings_per_account={'A':3,'B':3},release_evidence=str(launch_path),acceptance_evidence=str(E/'dual_final_acceptance.json'),
 recommendation='UAVDT selected and six jobs launched under explicit user authorization.',
 monitor_entry=str(HERE/'monitor_training.py'),repeat_launch_prohibited=True)
policy_path.write_text(json.dumps(policy,ensure_ascii=False,indent=2),encoding='utf-8')
text=plan_path.read_text(encoding='utf-8')
text=text.replace('任务已纳入计划，但目前不是可执行训练队列；UAVDT 训练入口与接续控制器尚未实现验收。','执行状态更新：两端全部技术预检通过，已一次性启动六个96轮任务。每卡并发3个，各自独立保存目录。原准备状态已归档至执行_v2，下面仍保留预声明验收条件。')
text=text.replace('计划中的新实验目录（尚未据此宣称已部署）：','已部署并启动的新实验目录：')
text=text.replace('训练忽略区域的处理仍需统计正例重叠后明确并测试，不能仅设置 iscrowd 就宣称已生效。','训练忽略区域已固定为增强前均值填充，并保护全部正例框内像素；验证/测试原图不擦除。此为本项目一致应用的训练适配，不能宣称复现作者训练配方。')
text+='\n十、实测验收与开训更新（'+now.isoformat()+'）\n两端40735帧SHA、划分、GT/忽略框、640前向、C/C-mean各17440新增参数、初始预测一致、10项协议测试、4项原生stage2集成、完整保存/恢复及跨任务路径拒绝测试全部通过。\n数据公共初始化SHA256：'+proof['A']['initialization_sha256']+'\n'
for a in ['A','B']:
 text+=f'{a}：1/2/3并发均通过，选择3；控制器PID{launches[a]["pid"]}，各job独立目录。\n'
text+='短批次3并发投影训练主体约39–40小时/卡；此数不含完整增强变化、每轮验证和最终评估/分析，须以完整轮次更新。前述245任务小时/5.1天为并发实测前历史估计，不再作为本轮实时工期。\n启动证据：'+str(launch_path)+'\n只读监测入口：'+str(HERE/'monitor_training.py')+'\n'
plan_path.write_text(text,encoding='utf-8')
record={'checked_utc':now.isoformat(),'acceptance':proof,'launch':launches,'capacity':capacity,'snapshot':snap['source'],
 'scope':'Actual 1/2/3 real-batch tests only; three is maximum tested safe, not a hardware concurrency maximum.',
 'GPU_type':'RTX3090 24GiB','six_formal_jobs_started':True}
with (E/'双卡并发实测与六任务启动验收.json').open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
lines=['双卡并发实测与六任务启动验收',now.isoformat(),'配置：UAVDT，batch4，640输入，AMP，每进程4线程。每worker含高密度真实GT的Mosaic/Mixup压力、8步预热及40步计时；不写正式训练权重。','账号 | 并发 | GPU总显存峰值MiB | GPU最少空闲MiB | 合计图像/秒']
for a,c in capacity.items():
 for row in c['cases']:
  lines.append(f'{a} | {row["concurrency"]} | {row["peak_GPU_used_MiB"]} | {row["minimum_GPU_free_MiB"]} | {row["aggregate_images_per_second"]:.2f}')
lines+=['两卡均选择3并发：已测安全且总吞吐最高；没有测试4及以上，不宣称3为硬件极限。','A运行seed0，B运行seed1；每端基线、C、C-mean各96轮，六任务都已启动。','训练完成后既定控制器自动执行best_stg2绑定、独立验证/官方测试及尺寸/定位分析；异常不自动重跑。','数据新公共初始化SHA：'+proof['A']['initialization_sha256'],'LA检查时间：'+snap['checked_America_Los_Angeles'],'保存目录和线程环境已核验，当前尚无完整验证结果。','完整原始证据：'+str(launch_path),'只读监测：'+str(HERE/'monitor_training.py')]
(E/'双卡并发实测与六任务启动验收.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'state':'published','formal_jobs':6,'maximum_concurrent_per_card':3,'report':str(E/'双卡并发实测与六任务启动验收.txt')},ensure_ascii=False))
