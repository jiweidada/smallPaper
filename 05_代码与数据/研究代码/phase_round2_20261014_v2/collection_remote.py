"""Read-only manifest for completed/failed jobs; checkpoint files are excluded."""
import hashlib
import json
import pathlib
import time

COLLECTED = []  # INSERT_COLLECTED_IDS
account = 'A' if pathlib.Path('/25085400120/DEIM').is_dir() else 'B'
storage = pathlib.Path('/25085400120' if account == 'A' else '/24085403014')
phase = next(p for p in [storage/'DEIM/experiments/phase_mechanism_20261012', storage/'experiments/phase_mechanism_20261012'] if p.is_dir())
stage = phase/'artifacts/round2_20261014_v1'
pointer = stage/'current_execution.json'
execution_name = json.loads(pointer.read_text())['execution_directory'] if pointer.exists() else 'execution_v2'
assert execution_name in ['execution_v2', 'execution_v3']
execution = stage/execution_name
plan = json.loads((stage/'addons_v2/queue_plan.json').read_text())
queue = json.loads((execution/'queue_state.json').read_text()) if (execution/'queue_state.json').exists() else {'state':'not_started'}
files = []
def add(source, target):
    if not source.is_file():
        return
    assert source.suffix.lower() in ['.json','.jsonl','.txt','.csv','.log']
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024), b''):
            digest.update(chunk)
    files.append({'remote':str(source),'local_relative':target,'sha256':digest.hexdigest(),'bytes':source.stat().st_size})

new_ids = []
for spec in plan['jobs']:
    arm,seed = spec.split(':'); job = arm+'_seed'+seed
    status = phase/'jobs'/(job+'.json')
    if not status.exists() or job in COLLECTED:
        continue
    state = json.loads(status.read_text())
    if state.get('state') not in ['complete','failed']:
        continue
    new_ids.append(job)
    out = phase/'outputs'/job
    add(out/'log.txt', job+'/log.txt')
    add(out/'args.json', job+'/args.json')
    add(status, job+'/jobs_status.json')
    add(out/'round2_train_acceptance.json', job+'/training_acceptance.json')
    for split in ['val','testdev']:
        root = phase/'round2_results'/job/split
        add(root/'project_eval.json', job+'/'+split+'_independent_eval.json')
        add(root/'manifest.json', job+'/'+split+'_prediction_manifest.json')
    label = ('C_minus_Cmean_seed'+seed if arm=='c_mean_full96' else
             'rank16_minus_rank16mean_seed'+seed if arm=='rank16_mean_full96' else
             'C_minus_SOEPlite_seed'+seed if arm=='soep_lite_full96' else None)
    if label:
        root = phase/'round2_analysis'/label
        add(root/'analysis.json', job+'/D_analysis.json')
        add(root/'gt_matching.csv', job+'/D_gt_matching.csv')
existing_id = 'existing_analyses_seed'+str(plan['primary_seed'])
if existing_id not in COLLECTED and (phase/'round2_analysis'/('existing_C_minus_singlephase_seed'+str(plan['primary_seed']))/'analysis.json').is_file():
    new_ids.append(existing_id)
    for model in plan['existing_models']:
        name = model['name']
        destination = existing_id+'/'+name
        out = pathlib.Path(model['checkpoint']).parent
        add(out/'log.txt',destination+'/log.txt')
        add(out/'args.json',destination+'/args.json')
        for split in ['val','testdev']:
            add(phase/'round2_results'/name/split/'project_eval.json', destination+'/'+split+'_independent_eval.json')
            add(phase/'round2_results'/name/split/'manifest.json', destination+'/'+split+'_prediction_manifest.json')
    for path in (phase/'round2_analysis').glob('existing_*seed'+str(plan['primary_seed'])+'/*'):
        if path.name in ['analysis.json','gt_matching.csv']:
            add(path,existing_id+'/D/'+path.parent.name+'/'+path.name)
translation_id = 'translation_seed0'
if account=='A' and translation_id not in COLLECTED and (phase/'round2_translation/lowpass/dx0_dy32_analysis/analysis.json').is_file():
    new_ids.append(translation_id)
    for path in (phase/'round2_translation').glob('*/*_analysis/*'):
        if path.name in ['analysis.json','gt_matching.csv']:
            add(path, translation_id+'/'+str(path.relative_to(phase/'round2_translation')))
print(json.dumps({'account':account,'remote_time':time.time(),'queue':queue,'new_collection_ids':new_ids,'files':files},ensure_ascii=False))
