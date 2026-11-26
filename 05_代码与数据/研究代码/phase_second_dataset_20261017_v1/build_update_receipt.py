"""Local evidence index for the completed dataset preparation; no remote actions."""
import datetime
import hashlib
import json
from pathlib import Path
import tomllib

HERE=Path(__file__).resolve().parent
DATA=HERE.parents[1]/'data/07_第二数据集_20261017'
EVIDENCE=DATA/'协议核查_v1'
def read(name):return json.loads((EVIDENCE/name).read_text(encoding='utf-8'))
accept={a:read('TinyPerson最终CPU验收_'+a+'.json') for a in ['A','B']}
assert len({v['initialization_sha256'] for v in accept.values()})==1
integration={a:json.loads((EVIDENCE/('TinyPerson原生保存集成核验_'+a+'.json')).read_text(encoding='utf-8').splitlines()[-1]) for a in ['A','B']}
assert all(v['actual_native_solver_integration_passed'] for v in integration.values())
uav=read('UAVDT原始标注与工具更新_B.json')
automation=tomllib.loads((Path.home()/'.codex/automations/automation/automation.toml').read_text(encoding='utf-8'))
assert automation['status']=='ACTIVE' and 'handoff_and_monitor_tiny.py' in automation['prompt']
last=json.loads((DATA/'每半小时跟进/latest_snapshot.json').read_text(encoding='utf-8'))
result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selected_second_dataset':'TinyPerson_v1',
 'TinyPerson':{'state':'prepared_waiting_for_both_original_queues','acceptance':accept,'native_solver_integration':integration,
  'portable_archive':read('TinyPerson双账号数据包_B.json'),
  'source':'Existing OpenDataLab-attributed original-format assets, checked against author schema and evaluator; no author byte-level checksum available.',
  'train_images':586,'validation_images':160,'official_test_images':786,'test_normal_ground_truth':13687,
  'training_arms':['baseline','c','c_mean','lowpass'],'seeds':[0,1],'epochs':96,'training_jobs':8,
  'main_metric':'AP50_tiny','official_metrics':42,'GPU_work_started':False,'training_started':False},
 'UAVDT':{'state':uav['status'],'ready_for_training':False,'official_toolkit_downloaded':True,
  'image_package_blocker':uav['image_download_blocker'],
  'minimum_proven_missing_test_frames':sum(len(x['missing_up_to_annotation_max']) for x in uav['coverage'] if x['split']=='test'),
  'image_download_url':uav['author_images_link'],'author_baidu_alternative':'https://pan.baidu.com/s/1Lydaht5ZGf0hPqGuHB8hQg',
  'baidu_code':'uwn8','baidu_download_verified':False,'official_evaluator_execution_verified':False},
 'automation':{'id':automation['id'],'status':automation['status'],'owner_thread':automation['target_thread_id'],
  'interval_minutes':30,'existing_automation_updated':True,'duplicate_created':False,
  'prompt_sha256':hashlib.sha256(automation['prompt'].encode()).hexdigest()},
 'last_guard_check':{'checked_los_angeles':last['checked_los_angeles'],
  'both_original_queues_complete':last['both_original_queues_complete'],'launches':last['launches']},'files':[]}
for folder in [EVIDENCE,HERE]:
 for path in sorted(folder.rglob('*')):
  if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ['.partial']:
   result['files'].append({'relative_to_workspace':str(path.relative_to(HERE.parents[2])),
                          'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
target=DATA/'数据集更新与接续验收_20261017_v1.json'
with target.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({'receipt':str(target),'evidence_files':len(result['files']),
                  'TinyPerson_prepared':True,'UAVDT_images_pending':True,'automation_status':automation['status']},ensure_ascii=False))
