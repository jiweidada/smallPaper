"""Read-only verify inputs for the already scheduled remaining postprocessing."""
import hashlib
import json
import pathlib
import sys

account=sys.argv[1]
phase=pathlib.Path('/25085400120/DEIM/experiments/phase_mechanism_20261012' if account=='A'
                   else '/24085403014/experiments/phase_mechanism_20261012')
stage=phase/'artifacts/round2_20261014_v1/execution_v5'
plan=json.loads((stage/'queue_plan.json').read_text())
resume=json.loads((stage/'resume_plan.json').read_text())
report={'account':account,'models':[],'data':[],'fresh_pending':[],
        'analysis_output_exists':(phase/'round2_results').exists()}
for row in plan['existing_models']:
    checkpoint=pathlib.Path(row['checkpoint'])
    config=phase/'artifacts/round2_20261014_v1/execution_v3/code'/row['config']
    report['models'].append({'name':row['name'],'checkpoint':str(checkpoint),'exists':checkpoint.is_file(),
                            'bytes':checkpoint.stat().st_size if checkpoint.is_file() else None,
                            'config_exists':config.is_file()})
for key in ['val_annotations','testdev_annotations','testdev_images']:
    path=pathlib.Path(plan[key])
    report['data'].append({'key':key,'path':str(path),'exists':path.exists()})
for item in plan['jobs']:
    job=item.replace(':','_seed')
    if job in resume['completed_jobs'] or any(r['job']==job for r in resume['resume_jobs']):
        continue
    report['fresh_pending'].append({'job':job,'output_exists':(phase/'outputs'/job).exists(),
                                   'job_json_exists':(phase/'jobs'/(job+'.json')).exists(),
                                   'console_exists':(phase/'logs'/(job+'.log')).exists()})
report['remaining_inputs_verified']=all(r['exists'] and r['config_exists'] for r in report['models']) and all(r['exists'] for r in report['data'])
print(json.dumps(report,ensure_ascii=False))
