"""Integrity verification and receipt after the rectangular-input regression fix."""
import json
from pathlib import Path
import sys
from experiment import HERE, verify_bundle, check_code
from runtime import implementation_hashes, sha


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inventory(root):
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def main():
    old_versions={}
    for version,filename in [('v1','03_交付回执与SHA.json'),('v2','02_交付回执与SHA.json'),
                             ('v3','02_交付回执与SHA.json'),('v4','01_交付回执与SHA.json')]:
        root=HERE.parent/('fine_residual_geometry_20261008_'+version)
        old=read(root/filename)
        assert all(sha(root/k)==v for k,v in old['files_sha256'].items())
        if version=='v4':
            assert inventory(root/'native_import_deps')==old['native_import_dependencies_sha256']
            assert inventory(root/'native_import_wheels')==old['native_import_wheels_sha256']
        assert not (root/'runs').exists()
        old_versions[version]={'unchanged_bound_files':len(old['files_sha256']),
                               'receipt_sha256':sha(root/filename),'runs_present':False}
    previous=HERE.parent/'fine_residual_geometry_20261008_v4'
    old=read(previous/'01_交付回执与SHA.json')
    current=implementation_hashes()
    changed=[name for name,digest in current.items() if old['implementation'][name]!=digest]
    assert changed==['stage_trace.py'],changed
    assert inventory(HERE/'native_import_deps')==old['native_import_dependencies_sha256']
    same=['geometry.py','residual.py']+[p.relative_to(HERE).as_posix() for p in sorted((HERE/'configs').glob('*.json'))]
    assert all(sha(HERE/p)==sha(previous/p) for p in same)
    bundles=sorted((HERE/'prepared_local').iterdir())
    assert len(bundles)==6 and not (HERE/'runs').exists()
    for bundle in bundles:
        receipt,plan,paths=verify_bundle(bundle)
    static=check_code(paths['native_root'])
    regression=read(HERE/'engineering_stages/report.json')
    native=read(HERE/'engineering_native_sizes/report.json')
    assert regression['status']=='PASS_SYNTHETIC_PARAMETER_FREE_CPU_AND_ADAPTER_STUBS'
    assert regression['check_count']==45 and native['check_count']==33
    assert native['status']=='PASS_NATIVE_RECTANGULAR_DATA_AND_STAGE_ARCHIVE_INTERFACE'
    assert native['old_failure_reproduced']
    assert regression['capture_code_sha256']==sha(HERE/'stage_trace.py')
    assert all(sha(HERE/k)==v for k,v in native['code_sha256'].items())
    assert sha(previous/'stage_trace.py')==native['old_stage_sha256']
    assert sha(paths['val_ann'])==native['real_annotation_metadata_only']['sha256']
    assert 'torch' not in sys.modules
    report={'status':'SIZE_ORDER_FIXED_NATIVE_RECTANGULAR_INTERFACE_PASS_NOT_TRAINED',
        'goal_status':'active_not_complete',
        'previous_goal_turn':'progress_evaluator_binding_and_native_interfaces',
        'this_goal_turn':'progress_native_dimension_contract_bug_reproduced_and_fixed',
        'main_candidate':'fine_residual','controls':['baseline','fused_residual'],'seeds':[1,2],
        'old_versions_unchanged':old_versions,'implementation':current,
        'changed_implementation_files_vs_v4':changed,'same_model_and_plan_files':same,
        'old_bug_scope':'new v3/v4 stage observer, not original baseline inference or evaluation',
        'real_validation_dimension_counts':native['real_annotation_metadata_only']['dimension_counts'],
        'prepared_bundles':6,'static_check':static,
        'parameter_free_regression_checks':45,'native_rectangular_interface_checks':33,
        'parameter_free_toy_forwards':17,'full_detector_forwards':0,
        'real_research_images_read':0,'trained_checkpoint_forwards':0,'checkpoint_loads':0,
        'optimizer_constructions':0,'parameter_updates':0,'gradient_calls':0,
        'new_scientific_checks':0,'training_started':False,'remote_access':False,
        'native_import_dependencies_sha256':inventory(HERE/'native_import_deps'),
        'whole_training_integration_verified':False,'historical_AP_reproduction_verified':False,
        'AP_improvement_verified':False,'unique_root_cause_proven':False,'novelty_established':False,
        'files_sha256':{p.relative_to(HERE).as_posix():sha(p) for p in sorted(HERE.rglob('*'))
                        if p.is_file() and 'native_import_deps' not in p.parts}}
    with (HERE/'01_交付回执与SHA.json').open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ('status','changed_implementation_files_vs_v4',
        'prepared_bundles','parameter_free_regression_checks','native_rectangular_interface_checks',
        'full_detector_forwards')},ensure_ascii=False))


if __name__=='__main__':
    main()
