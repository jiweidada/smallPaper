"""Record a conditional ETA from complete real epochs, without running new experiments."""
import datetime,hashlib,json,re
from pathlib import Path
from zoneinfo import ZoneInfo
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]/'data/07_第二数据集_20261017';E=BASE/'执行_v2'
source=sorted(E.glob('*_eta_actual_epoch_validation_timings.json'))[-1]
raw=json.loads(source.read_text(encoding='utf-8'));snapshot=BASE/'每半小时跟进_v2/latest_snapshot.json'
live=json.loads(snapshot.read_text(encoding='utf-8'));now=datetime.datetime.fromisoformat(live['checked_utc']);rows=[]
for account,result in raw['results'].items():
 assert result['returncode']==0
 for j in json.loads(result['stdout'])['jobs']:
  current=next(v for v in live['accounts'][account]['jobs'] if v['job']==j['job'])
  assert j['actual_output_ok'] and current['output_binding_passed'] and current['process']['alive']
  n=current['batches_per_epoch'];progress=current['epoch_zero_based']+current['batch_zero_based']/n
  epoch=j['structured_last_row']['experiment_epoch_seconds'];train=j['structured_last_row']['experiment_train_seconds']
  rows.append({'account':account,'job':j['job'],'completed_validation_epochs':current['completed_epochs'],
      'current_epoch_user_facing':current['epoch_zero_based']+1,'batch_zero_based':current['batch_zero_based'],
      'last_complete_epoch_minutes':epoch/60,'last_complete_training_minutes':train/60,
      'last_validation_and_bookkeeping_seconds':epoch-train,
      'conditional_remaining_training_plus_epoch_validation_hours':(96-progress)*epoch/3600})
record={'checked_utc':now.isoformat(),'checked_America_Los_Angeles':now.astimezone(ZoneInfo('America/Los_Angeles')).isoformat(),
 'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [source,snapshot]],
 'jobs':rows,'remaining_training_plus_epoch_validation_budget_hours':[50,75],
 'final_independent_val_test_and_size_localization_collection_budget_hours':[2,4],
 'postprocessing_budget_is_not_a_measured_completion_time':True,'rounded_total_remaining_hours':[55,80],
 'projected_finish_LA':[ (now+datetime.timedelta(hours=h)).astimezone(ZoneInfo('America/Los_Angeles')).isoformat() for h in [55,80]],
 'assumptions':['Both GPUs stay available without interruption.','Six jobs run concurrently; use the slowest account/job, do not multiply wall time by six.',
 'Only five real epochs per arm have finished. Augmentation stages and GPU scheduling can change future throughput.',
 'B C-mean last complete epoch took49.24min versus32–35min for other arms; its slowdown cause is not established.',
 '39–40h early projection was a short-batch estimate; actual complete-epoch timings supersede it.'],
 'monitor_correction':'Resolved output path is logged once at initialization. Read the log header as well as its tail; six real paths remain correct. Training code/configuration/processes unchanged.'}
path=E/(now.strftime('%Y%m%dT%H%M%S_%fZ')+'_六任务剩余工期估算.json')
with path.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
print(json.dumps({'evidence':str(path),'checked_LA':record['checked_America_Los_Angeles'],'remaining_hours':record['rounded_total_remaining_hours'],
 'finish_LA':record['projected_finish_LA'],'last_whole_epoch_minutes':{r['job']:round(r['last_complete_epoch_minutes'],2) for r in rows}},ensure_ascii=False,indent=2))
