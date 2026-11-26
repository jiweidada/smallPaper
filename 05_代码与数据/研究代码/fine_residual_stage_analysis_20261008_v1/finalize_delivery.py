"""Static delivery verification; no real scientific geometry analysis."""
import ast
import json
from pathlib import Path
import sys
from analyze_stages import HERE, read, sha, write


def main():
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    experiment=HERE.parent/'fine_residual_geometry_20261008_v5'
    receipt=read(experiment/'01_交付回执与SHA.json')
    binding=read(HERE/'experiment_binding.json')
    assert sha(experiment/'01_交付回执与SHA.json')==binding['delivery_receipt_sha256']
    assert all(sha(experiment/k)==v for k,v in receipt['files_sha256'].items())
    assert all(sha(experiment/'native_import_deps'/k)==v for k,v in receipt['native_import_dependencies_sha256'].items())
    assert not (experiment/'runs').exists()
    assert all(sha(experiment/k)==v for k,v in binding['implementation'].items())
    check=read(HERE/'engineering_synthetic/report.json')
    assert check['status']=='PASS_PURE_SYNTHETIC_GEOMETRY_AND_IDENTITY_JOIN'
    assert check['check_count']==30
    assert all(sha(HERE/k)==v for k,v in check['code_sha256'].items())
    assert 'torch' not in sys.modules
    write(HERE/'01_交付回执与SHA.json',{
        'status':'FUTURE_STAGE_TO_NATIVE_TP_ANALYSIS_IMPLEMENTED_SYNTHETICALLY_VERIFIED',
        'goal_status':'active_not_complete','main_candidate':'fine_residual',
        'previous_goal_turn':'progress_native_size_order_bug_fixed',
        'this_goal_turn':'progress_stage_geometry_to_native_TP_analysis_and_counterexamples',
        'bound_experiment':experiment.name,'experiment_delivery_sha256':binding['delivery_receipt_sha256'],
        'v5_bound_files_unchanged':len(receipt['files_sha256']),'v5_runs_present':False,
        'synthetic_checks':30,'real_baseline_arrays_read':0,'real_research_images_read':0,
        'model_calls':0,'evaluator_calls':0,'optimizer_calls':0,'checkpoint_loads':0,
        'training_started':False,'new_scientific_checks':0,'remote_access':False,
        'formal_six_run_end_to_end_execution_verified':False,
        'AP_improvement_verified':False,'causal_mechanism_proven':False,
        'files_sha256':{p.relative_to(HERE).as_posix():sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()}})
    print(json.dumps({'status':'DELIVERED_SYNTHETICALLY_VERIFIED_NOT_RUN_ON_REAL_EXPERIMENTS',
                      'checks':30,'v5_files_unchanged':len(receipt['files_sha256'])}))


if __name__=='__main__':
    main()
