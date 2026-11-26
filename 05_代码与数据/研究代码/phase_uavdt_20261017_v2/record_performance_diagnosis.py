"""Publish bounded performance diagnosis with raw sources and restored runtime settings."""
import datetime,hashlib,json
from pathlib import Path
from zoneinfo import ZoneInfo
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]/'data/07_第二数据集_20261017';E=BASE/'执行_v2'
sources=[]
def latest(tag):
 path=sorted(E.glob('*_'+tag+'.json'))[-1];sources.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 raw=json.loads(path.read_text(encoding='utf-8'));assert all(r.get('returncode')==0 for r in raw['results'].values())
 return {a:json.loads(r['stdout']) for a,r in raw['results'].items()}
system=latest('slow_training_CPU_IO_GPU_diagnostic');native=latest('GPU_locality_native_loop_overhead')
integrity=latest('performance_diagnostic_final_integrity');trial=latest('CPU_locality_trial_corrected_measurement');restored=latest('CPU_locality_trial_complete')
for a in ['A','B']:
 assert not integrity[a]['changed_frozen_files'] and not trial[a]['restore_receipt']['errors']
 assert restored[a]['trial']['result']['state']=='complete_and_original_affinities_restored'
 assert all(j['main_affinity']==list(range(128)) for j in restored[a]['current_jobs'])
 assert all(j['args_fields']['use_amp'] and not j['args_fields']['use_amp_fallback_fp32'] and not j['warning_error_fallback_lines'] for j in integrity[a]['jobs'])
now=datetime.datetime.now(datetime.timezone.utc)
record={'checked_utc':now.isoformat(),'checked_America_Los_Angeles':now.astimezone(ZoneInfo('America/Los_Angeles')).isoformat(),
 'sources':sources,'training_step_count':{'UAVDT':4747,'prior_VisDrone':1617,'ratio':4747/1617},
 'regular_loop_projection':{'seconds_per_step':.4,'training_minutes_per_epoch':4747*.4/60,'validation_approx_minutes_per_epoch':1.2,'full96_wall_hours':(4747*.4+72)*96/3600},
 'confirmed':['Two distinct GPU UUIDs, one24GiB3090 per account.','Fixed640, batch4, AMP, two dataloader workers and four computation threads for every arm.',
 'Data waiting median approximately0.0002s/step; cgroup quota8CPU and low CPU-pressure during sample.',
 'Temperature49–51C and no thermal/power throttle active in recorded samples.',
 'Each step enters five CUDA Profile contexts, each synchronizing at entry and exit even with TIME_DEBUG=False. Finite checks and scalar metric copies add more synchronization points.',
 'The prior short capacity test used a simplified update loop and timed only epoch0 regular batches; memory stress used just three dense augmented updates. It did not measure full-epoch native-loop throughput.',
 'All recorded first-batch SHAs agree within seed across the three arms for common epochs. This verifies first batches, not the entire data stream.',
 'All frozen files still match manifest; no AMP FP32 fallback or epoch rollback detected.'],
 'unresolved':['The precise trigger of B C-mean epoch4 slowdown is not isolated. Data-prefetch, precision fallback, resize differences and persistent hardware throttling were not supported by the sampled evidence.',
 'CPU-locality trial overlapped some evaluation/epoch transitions. Its short-window percentages are not causal performance estimates; no consistent gain was established.',
 'Native synchronization overhead is established by code, but individual overhead contributions were not separately profiled.'],
 'CPU_locality_trial':trial,'runtime_affinity_retained':False,'all_original_affinities_restored':True,
 'frozen_integrity':{a:{k:v[k] for k in ['frozen_files_checked','changed_frozen_files']} for a,v in integrity.items()},
 'decision':'Keep all six original trainings running with original runtime affinity. No recipe/code/checkpoint changes or restart. Optimizing redundant timer synchronizations is a future separately verified implementation change, not applied to this frozen run.'}
path=E/(now.strftime('%Y%m%dT%H%M%S_%fZ')+'_UAVDT慢速原因与不改配方提速核查.json')
with path.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
text=['UAVDT慢速原因与不改配方提速核查',record['checked_America_Los_Angeles'],
 '主体工期：4747步/轮×约0.40秒/步≈31.65分钟训练；再加约1分钟验证，完整96轮约52–53小时。每端三个任务并行，取慢任务工期，不乘三或乘六。',
 'UAVDT每轮步数约为VisDrone1617步的2.94倍；单任务时间还受三进程共享GPU影响。',
 '旧39–40小时是简化更新循环的40步短测外推，缺正式循环中的计时同步、逐项有限值检查、日志及每轮验证，不是完整训练速度。',
 '代码确认：五个Profile上下文每步合计十次显式CUDA同步；TIME_DEBUG=False仍会执行。逐项isfinite和标量日志还带来同步。各项耗时未单独量化。',
 '排查：固定640/batch4/AMP；数据等待约0.0002秒/步；8CPU配额无持续耗尽；GPU温度49–51℃，未见温度/功耗限速；两端独立显卡。',
 'B的C-mean出现过0.6–0.8秒/步，随后在调度调整之前自行恢复到约0.4；额外慢速具体诱因仍未锁定。',
 '已试CPU NUMA亲和性：原设置45秒→显卡所在CPU节点90秒→恢复原设置45秒。窗口遇到部分验证/换轮，无一致提速证据，不能把短窗口差值当因果幅度。所有原亲和性已恢复，未保留设置。',
 '日志带ANSI颜色导致原试验在线解析没有取得有效吞吐；已按保留的相同时间窗只读重算，不重做调度试验，原记录保留。',
 '冻结完整性：A1656文件、B695文件全部匹配；首批SHA在同种子三臂已共同记录的轮次一致。未发现AMP回退或异常重算，训练没有停止或重启。',
 '处理：维持当前六个正式任务。计时同步等训练实现优化须另做数值/状态等价验证，本轮没有修改冻结实现，也未改变配方。',
 '原始证据与完整限制：'+str(path)]
report=path.with_suffix('.txt');report.write_text('\n'.join(text)+'\n',encoding='utf-8')
print(json.dumps({'report':str(report),'all_original_affinities_restored':True,'frozen_files_unchanged':True,'no_consistent_runtime_tuning_gain_established':True},ensure_ascii=False))
