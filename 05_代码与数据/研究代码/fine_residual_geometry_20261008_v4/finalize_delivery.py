"""Finalize static evidence after engineering checks; never imports a model."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from experiment import check_code, verify_bundle
from runtime import HERE, implementation_hashes, sha


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inventory(root):
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def main():
    old = {}
    for version, name in [('v1','03_交付回执与SHA.json'),
                          ('v2','02_交付回执与SHA.json'),('v3','02_交付回执与SHA.json')]:
        root = HERE.parent/('fine_residual_geometry_20261008_'+version)
        receipt = read(root/name)
        bound = receipt['files_sha256']
        assert all(sha(root/rel)==expected for rel,expected in bound.items())
        old[version] = {'unchanged_files':len(bound), 'receipt_sha256':sha(root/name)}
    previous = HERE.parent/'fine_residual_geometry_20261008_v3'
    unchanged = ['geometry.py','residual.py','stage_trace.py'] + [
        p.relative_to(HERE).as_posix() for p in sorted((HERE/'configs').glob('*.json'))]
    assert all(sha(HERE/name)==sha(previous/name) for name in unchanged)
    bundles = sorted((HERE/'prepared_local').iterdir())
    assert len(bundles)==6 and not (HERE/'runs').exists()
    identities = []
    for bundle in bundles:
        receipt,plan,paths=verify_bundle(bundle)
        identities.append({'arm':plan['arm'],'seed':plan['seed'],
                           'evaluator_binding':receipt['identity']['evaluator_binding']})
    assert all(item['evaluator_binding']==identities[0]['evaluator_binding'] for item in identities)
    static = check_code(paths['native_root'])
    integrity = read(HERE/'engineering_evaluator/report.json')
    native = read(HERE/'engineering_evaluator/native_config_03.json')
    fresh = read(HERE/'engineering_native_models/report.json')
    assert integrity['check_count']==21 and integrity['status'].startswith('PASS_')
    assert native['status']=='PASS_NATIVE_IMPORTS_AND_SIX_CONFIGURATION_READS_ONLY'
    assert fresh['check_count']==13 and fresh['full_fresh_detector_forwards']==3
    assert fresh['implementation']==implementation_hashes()
    assert not native['guard_attempts'] and not fresh['guard_attempts']
    audit = HERE.parents[1]/'data'/'00_实验计划与证据索引'/\
            '单一候选评价依赖补齐与构建核验_20261008_v1'
    dependency = read(audit/'05_依赖交付回执与SHA.json')
    assert inventory(audit/'isolated_runtime')==dependency['isolated_runtime_sha256']
    original = Path(read(audit/'03_隔离构建回执.json')['original_source'])
    assert inventory(original)==read(audit/'03_隔离构建回执.json')['original_source_sha256']
    freeze = Path('D:/BaiduNetdiskDownload/DEIM_1572862469/平台迁移包_20260929/pip_freeze.txt')
    default = {name:importlib.util.find_spec(name) is not None
               for name in ('pycocoeval','dill','easydict','pynvml')}
    assert not any(default.values()) and 'torch' not in sys.modules
    report = {'status':'SINGLE_CANDIDATE_EVALUATOR_BOUND_NATIVE_CPU_INTERFACE_VERIFIED_NOT_TRAINED',
        'goal_status':'active_not_complete',
        'previous_goal_turn':'progress_evaluator_source_and_isolated_build',
        'this_goal_turn':'progress_complete_evaluator_binding_and_native_interface_integration',
        'main_candidate':'fine_residual','controls':['baseline','fused_residual'],'seeds':[1,2],
        'old_versions_unchanged':old, 'same_model_plan_and_stage_files':unchanged,
        'implementation':implementation_hashes(),'static_check':static,
        'prepared_bundles':len(bundles),'prepared_identities':identities,
        'package_identity_checks':21,'native_configurations_read':6,
        'native_fresh_model_interface_checks':13,'fresh_CPU_forwards':3,
        'trained_checkpoint_forwards':0,'checkpoint_loads':0,'gradient_calls':0,
        'optimizer_constructions':0,'parameter_updates_after_initialization':0,
        'training_started':False,'new_scientific_checks':0,'remote_access_this_turn':False,
        'source_dependency_build_unchanged':True,
        'default_interpreter_module_availability':default,
        'supplementary_dependency_version_source':str(freeze),
        'supplementary_dependency_version_source_sha256':sha(freeze),
        'wheel_index_used':'https://pypi.tuna.tsinghua.edu.cn/simple',
        'native_import_wheels_sha256':inventory(HERE/'native_import_wheels'),
        'native_import_dependencies_sha256':inventory(HERE/'native_import_deps'),
        'native_full_training_verified':False,'historical_AP_reproduction_verified':False,
        'AP_improvement_verified':False,'causal_mechanism_proven':False,
        'novelty_established':False,'cost_verified':False,
        'files_sha256':{p.relative_to(HERE).as_posix():sha(p) for p in sorted(HERE.rglob('*'))
                        if p.is_file() and not any(d in p.parts for d in
                            ('native_import_wheels','native_import_deps'))}}
    with (HERE/'01_交付回执与SHA.json').open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ('status','old_versions_unchanged','prepared_bundles',
        'fresh_CPU_forwards','native_full_training_verified')},ensure_ascii=False))


if __name__=='__main__':
    main()
