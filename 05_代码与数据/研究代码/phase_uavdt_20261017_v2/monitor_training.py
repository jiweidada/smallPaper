"""Read-only snapshots, completed evidence collection and within-seed native AP70 comparisons."""
import datetime,hashlib,json,subprocess,sys
from pathlib import Path
from zoneinfo import ZoneInfo
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]/'data/07_第二数据集_20261017'
DEST=BASE/'每半小时跟进_v2';DEST.mkdir(exist_ok=True)
p=subprocess.run([sys.executable,'-W','ignore','-X','utf8','-B',str(HERE/'control_uavdt.py'),'script','--account','both','--script',str(HERE/'monitor_training_remote.py'),'--tag','UAVDT_training_snapshot','--timeout','90'],capture_output=True,text=True,encoding='utf-8',timeout=210)
try:envelope=json.loads(p.stdout)
except ValueError:print(p.stdout);print(p.stderr);raise SystemExit(1)
raw=json.loads(Path(envelope['saved']).read_text(encoding='utf-8'));now=datetime.datetime.fromisoformat(raw['checked_utc'])
record={'checked_utc':raw['checked_utc'],'checked_America_Los_Angeles':now.astimezone(ZoneInfo('America/Los_Angeles')).isoformat(),'source':envelope['saved'],'accounts':{},'paired_results':[]}
for account,r in raw['results'].items():
 if r.get('returncode')!=0:record['accounts'][account]={'read_failed':True,'error':r.get('error') or r.get('stderr')};continue
 value=json.loads(r['stdout']);metrics={};collected=[]
 for item in value.pop('collectible',[]):
  rel=Path(item['relative']);assert not rel.is_absolute() and '..' not in rel.parts
  target=DEST/'collected'/account/rel;content=item['text'].encode();assert hashlib.sha256(content).hexdigest()==item['sha256']
  target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():assert target.read_bytes()==content
  else:target.write_bytes(content)
  collected.append({k:v for k,v in item.items() if k!='text'})
  if rel.name=='metrics.json' and rel.parent.name=='test':
   m=json.loads(item['text']);job=rel.parts[1];metrics[job]=m
 value['collected_files']=collected;record['accounts'][account]=value
 seed=value['seed']
 for arm,ref in [('c','baseline'),('c','c_mean'),('c_mean','baseline')]:
  a=f'uavdt_{arm}_full96_seed{seed}';b=f'uavdt_{ref}_full96_seed{seed}'
  if a in metrics and b in metrics:
   left,right=metrics[a],metrics[b]
   assert left['ground_truth_sha256']==right['ground_truth_sha256'] and left['protocol']==right['protocol']
   record['paired_results'].append({'account':account,'seed':seed,'comparison':arm+' - '+ref,'metric':'official_test_AP70',
       'difference_percentage_points':100*(left['metrics_raw_0_to_1']['AP70']-right['metrics_raw_0_to_1']['AP70']),
       'candidate_checkpoint_sha256':left['checkpoint_sha256'],'reference_checkpoint_sha256':right['checkpoint_sha256']})
path=DEST/(now.strftime('%Y%m%dT%H%M%S_%fZ')+'_snapshot.json')
with path.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
temp=DEST/'latest_snapshot.next';temp.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(DEST/'latest_snapshot.json')
summary={'checked_utc':record['checked_utc'],'checked_America_Los_Angeles':record['checked_America_Los_Angeles'],'evidence':str(path),'accounts':{},'paired_results':record['paired_results']}
for a,v in record['accounts'].items():
 if v.get('read_failed'):summary['accounts'][a]=v;continue
 summary['accounts'][a]={'seed':v['seed'],'queue_state':(v.get('queue') or {}).get('state'),'queue_process':v['queue_process'],'GPU':v['GPU'],'GPU_compute':v['GPU_compute'],
 'jobs':[{k:j.get(k) for k in ['job','status','process','output_binding_passed','completed_epochs','epoch_zero_based','batch_zero_based','batches_per_epoch','seconds_per_step','latest_complete_evaluation','console_bytes','initialization_log_tail','failure_log_last50']} for j in v['jobs']]}
 if 'controller_log_last50' in v:summary['accounts'][a]['controller_log_last50']=v['controller_log_last50']
print(json.dumps(summary,ensure_ascii=False,indent=2))
if p.returncode:raise SystemExit(p.returncode)
