"""Final static audit. Does not import torch or rerun detector checks."""
import ast
import hashlib
import json
from pathlib import Path
from integration import HERE, sha, verify_sources


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def run():
    sources=verify_sources()
    checks=[]
    def check(name,ok):
        checks.append(dict(name=name,passed=bool(ok)))
        if not ok:raise AssertionError(name)
    paths=read(HERE/'local_paths.json')
    for k in ('common_init','train_ann','val_ann'):
        check(k+'_unchanged',sha(paths[k])==paths[k+'_sha256'])
    for p in HERE.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'))
    check('all_new_python_parses',True)
    binding=read(HERE/'plan_binding.json')
    cfgs=sorted((HERE/'configs').glob('*.json'))
    check('six_plans',len(cfgs)==6)
    for p in cfgs:
        check(p.name+'_bound',sha(p)==binding[p.name])
        plan=read(p)
        args_path=next(Path(s) for s in sources if f'seed{plan["seed"]}' in s and s.endswith('args.json'))
        check(p.name+'_native_recipe_unmodified',read(args_path)['yaml_cfg']==plan['native_baseline_config'])
        bundle=HERE/'prepared_local'/p.stem
        rec=read(bundle/'receipt.json');resolved=read(bundle/'resolved.yml')
        check(p.stem+'_bundle_matches_plan',sha(bundle/'plan.json')==sha(p) and sha(bundle/'resolved.yml')==rec['resolved_sha256'])
        check(p.stem+'_implementation_bound',all(sha(HERE/name)==digest for name,digest in rec['implementation'].items()))
        check(p.stem+'_model_arm_wiring',resolved['model']=='P3ReadoutDEIM' and
              resolved['P3ReadoutDEIM']==dict(recipe=plan['model_recipe'],arm=plan['arm']))
        check(p.stem+'_no_run_output',not Path(resolved['output_dir']).exists())
    check('failed_engineering_receipt_preserved',read(HERE/'engineering_attempt1.json')['status']=='FAIL' and
          (HERE/'test_cpu_attempt1.py.txt').is_file())
    good=[read(HERE/x) for x in ('engineering_attempt2.json','engineering_connections.json')]
    check('engineering_acceptance_passed',all(x['status']=='PASS' and all(c['passed'] for c in x['checks']) for x in good))
    check('49_engineering_checks',sum(len(x['checks']) for x in good)==49)
    check('review_and_plan_exist',all((HERE/x).is_file() for x in ('00_代码入口与复审说明.txt','01_实验计划_三条件双seed.txt')))
    result=dict(status='PASS',scope='code_review_implementation_CPU_acceptance_experiment_plan',checks=checks,
        scientific_efficacy='UNTESTED',new_GPU_training=False,optimizer_constructed=False,
        source_files_verified=len(sources),passed_engineering_checks=49,
        full_detector_forwards_including_preserved_failed_attempt=19,
        scientific_model_inference_images=0,configuration_bundles=6,
        limitations=['full native training runner/evaluator integration pending','GPU/AMP and cost pending',
                     'deploy versus undeployed prediction equivalence unverified','no AP claim'])
    with (HERE/'02_交付审计.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    manifest={str(p.relative_to(HERE)).replace('\\','/'):sha(p) for p in HERE.rglob('*') if p.is_file()}
    with (HERE/'03_文件SHA256清单.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':result['status'],'checks':len(checks),'files':len(manifest)}))


if __name__=='__main__':run()
