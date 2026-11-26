"""Freeze the requested 12-job queue and generate append-only addon deployment."""
import base64
import hashlib
import json
import pathlib
import yaml

HERE = pathlib.Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
EVIDENCE = WORKSPACE/'英文小论文/data/06_第二轮_20261014/execution_v1'
files = ['activate_remote.py', 'dump_preds.py', 'analyze_preds.py', 'test_prediction_analysis.py',
         'run_translation_suite.py', 'soep_lite_round2.py', 'test_soep_lite.py', 'train_round2.py',
         'benchmark_soep_lite.py', 'run_round2_queue.py']
payload = {name: {'base64': base64.b64encode((HERE/name).read_bytes()).decode(),
                  'sha256': hashlib.sha256((HERE/name).read_bytes()).hexdigest()} for name in files}
plans = {}
configurations = {}
for account, primary_seed in [('A', 0), ('B', 1)]:
    snapshot = json.loads((EVIDENCE/('readonly_'+account+'_verified.json')).read_text(encoding='utf-8-sig'))
    audit_name = 'pair_audit_A_yaml.json' if account == 'A' else 'pair_audit_B_effective.json'
    audit = json.loads((EVIDENCE/audit_name).read_text(encoding='utf-8-sig'))
    matching = [row for row in audit['comparisons'] if row['seed'] == primary_seed and (primary_seed == 0 or 'affine_epoch3_resume' in row['path'])]
    assert len(matching) == 1
    old = matching[0]
    accepted = old.get('all_effective_fields_equal', old['all_listed_fields_equal'])
    assert accepted
    phase = snapshot['phase_root']
    config = {'__include__': ['proto96.yml'], 'output_dir': phase+'/outputs/soep_lite_full96',
              'DEIM': {'backbone': 'HGNetv2', 'encoder': 'HybridEncoder_SOEP_Lite'},
              'HGNetv2': {'return_idx': [0, 1, 2, 3], 'pretrained': False},
              'HybridEncoder_SOEP_Lite': {**audit['resolved']['HybridEncoder'], 'p2_channels': 64, 'bridge_width': 16},
              'optimizer': {'params': [
                  {'params': r'^(?=.*(?:backbone|encoder\.p2_fusion))(?!.*(?:bn|norm|bias)).*$', 'lr': .0001},
                  {'params': r'^(?=.*(?:backbone|encoder\.p2_fusion))(?=.*(?:bn|norm|bias)).*$', 'lr': .0001, 'weight_decay': 0.},
                  {'params': r'^(?!.*(?:backbone|encoder\.p2_fusion))(?=.*(?:bn|norm|bias)).*$', 'weight_decay': 0.}]}}
    configurations[account] = yaml.safe_dump(config, sort_keys=False)
    (HERE/('soep_lite_full96_'+account+'.yml')).write_text(configurations[account], encoding='utf-8')
    existing = [
        {'name': 'existing_c', 'config': 'configs/experiments/phase/c_full96.yml', 'checkpoint': str(pathlib.PurePosixPath(old['path']).parent/'best_stg2.pth')},
        {'name': 'existing_lowpass', 'config': 'configs/experiments/phase/lowpass_full96.yml', 'checkpoint': phase+'/outputs/lowpass_full96_seed'+str(primary_seed)+'/best_stg2.pth'},
        {'name': 'existing_singlephase', 'config': 'configs/experiments/phase/singlephase_full96.yml', 'checkpoint': phase+'/outputs/singlephase_full96_seed'+str(primary_seed)+'/best_stg2.pth'}]
    baseline = (phase+'/outputs/baseline_full96_seed1/best_stg2.pth' if account == 'B' else
                '/25085400120/DEIM/experiments/visdrone_light_20260928_090716/outputs/baseline_main_seed0/best_stg2.pth')
    existing.insert(0, {'name': 'existing_baseline', 'config': 'configs/experiments/phase/baseline_full96.yml', 'checkpoint': baseline})
    acceptance = json.loads((EVIDENCE/('acceptance_'+account+'.json')).read_text(encoding='utf-8-sig'))
    assert len(acceptance['testdev_candidates']) == 1
    plans[account] = {'account': account, 'primary_seed': primary_seed,
                     'common_init_sha256': snapshot['init_sha256'], 'effective_recipe_audit_accepted': accepted,
                     'recipe_audit_source': audit_name, 'old_C_args': old['path'],
                     'old_C_seed1_limit': 'Historical resumed trajectory lacks complete RNG restoration; not a claim of uninterrupted equivalence' if account == 'B' else None,
                     'val_annotations': audit['resolved']['val_dataloader']['dataset']['ann_file'],
                     'testdev_annotations': acceptance['testdev_candidates'][0], 'testdev_images': acceptance['testdev_images'],
                     'existing_jobs': ['lowpass_full96_seed0', 'singlephase_full96_seed0'] if account == 'A' else
                                      ['baseline_full96_seed1', 'lowpass_full96_seed1', 'singlephase_full96_seed1'],
                     'existing_models': existing,
                     'jobs': [mode+':'+str(primary_seed) for mode in ['c_mean_full96','rank16_full96','rank16_mean_full96','soep_lite_full96']]
                             + (['c_full96:2', 'c_mean_full96:2'] if account == 'A' else ['rank16_full96:2', 'rank16_mean_full96:2']),
                     'training_epochs': 96, 'retries': 0, 'baseline_seed2': False,
                     'human_authorization': '2026-10-14: 把这些实验都接在两个账号上面，让他们跑完这个实验继续下一个实验'}
    with (HERE/('queue_plan_'+account+'.json')).open('x', encoding='utf-8') as f:
        json.dump(plans[account], f, ensure_ascii=False, indent=2)
template = (HERE/'deploy_addons_template.py').read_text(encoding='utf-8')
template = template.replace('PAYLOAD = {}  # INSERT_FILES', 'PAYLOAD = '+repr(payload))
template = template.replace('PLANS = {}  # INSERT_PLANS', 'PLANS = '+repr(plans))
template = template.replace('CONFIGS = {}  # INSERT_CONFIGS', 'CONFIGS = '+repr(configurations))
with (HERE/'deploy_queue_addons_remote.py').open('x', encoding='utf-8', newline='\n') as f:
    f.write(template)
print(json.dumps({'training_jobs': sum(len(plan['jobs']) for plan in plans.values()), 'addon_files': len(files)}))
