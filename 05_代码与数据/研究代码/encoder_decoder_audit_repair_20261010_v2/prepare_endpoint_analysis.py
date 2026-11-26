"""Prepare a separate CPU-only endpoint reader; active training code stays frozen."""
from pathlib import Path
import ast,hashlib,json,datetime
ROOT=Path(__file__).resolve().parents[3]
NEW=ROOT/'英文小论文/code/small_object_encoder_decoder12_analysis_20261010_v1'
OUT=ROOT/'outputs/small_object_encoder_decoder12_analysis_20261010_v1'
OLD=ROOT/'英文小论文/code/fine_residual_stage_analysis_20261008_v1'
RUN=ROOT/'英文小论文/code/small_object_encoder_decoder12_20261010_v3'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    NEW.mkdir(exist_ok=False);OUT.mkdir(exist_ok=False)
    original=(OLD/'analyze_stages.py').read_text(encoding='utf8')
    core=original[:original.index('def load_bound_experiment(')]
    (NEW/'geometry_core.py').write_text(core,encoding='utf8')
    tests=(OLD/'test_stage_analysis.py').read_text(encoding='utf8')
    tests=tests.replace('import analyze_stages as a','import geometry_core as a')
    a=tests.index('    # Exercise the real v5 eligibility gate')
    b=tests.index("    check('synthetic analysis imports no torch",a)
    tests=tests[:a]+tests[b:]
    tests=tests.replace("'analyze_stages.py':a.sha(HERE/'analyze_stages.py')", "'geometry_core.py':a.sha(HERE/'geometry_core.py')")
    tests=tests.replace('formal_six_run_end_to_end_execution_verified','formal_fixed12_pair_execution_verified')
    (NEW/'test_geometry_core.py').write_text(tests.replace("a.sha(__file__)","a.sha(__file__)"),encoding='utf8')
    for name in ('results.py','stage_trace.py','evaluator_binding.py','evaluator_sources.json'):
        (NEW/name).write_bytes((RUN/name).read_bytes())
    for p in NEW.glob('*.py'):ast.parse(p.read_text())
    protocol={'status':'DESCRIPTIVE_ENDPOINT_ANALYSIS_FROZEN_BEFORE_FIXED12',
        'frozen_local_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'existing_geometry_source_sha256':sha(OLD/'analyze_stages.py'),
        'geometry_core_functions_preserved':True,
        'stages':['encoder_initial_boxes','pre_after_first_layer_boxes','final_boxes'],
        'thresholds':[.5+.05*i for i in range(10)],
        'groups':['all','COCO_small_original_area_le_1024','each_native_category'],
        'join':'Per annotation ID; never pair query slots across trained models.',
        'primary_gate_unchanged':'Same-seed fixed12 EMA APs strictly improves and overallAP does not decline; both seeds before capacity-matched controls.',
        'scope':'Descriptive localization of changes only. Geometric support may share queries and ignores ranking/class/one-to-one assignment; it is not native recall or AP. No causal-root claim.',
        'partial_training_results_already_seen':'Seed2 epoch1 AP andAPs positive; final12 not yet available. Analysis does not alter any promotion/stopping gate.',
        'new_model_forwards':0,'new_evaluator_calls':0,'new_training_updates':0}
    (NEW/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
    (OUT/'protocol.json').write_bytes((NEW/'protocol.json').read_bytes())
    print(json.dumps({'status':'CPU_ANALYSIS_CORE_PREPARED','files':len(list(NEW.iterdir()))}))
if __name__=='__main__':main()
