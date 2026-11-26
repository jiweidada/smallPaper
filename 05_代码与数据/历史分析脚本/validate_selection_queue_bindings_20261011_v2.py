"""Exercise the real parent verifier on three independent synthetic bindings."""
from pathlib import Path
import argparse, importlib.util, json, shutil, sys

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()

code = Path('英文小论文/code/native_selection_diagnostic12_20261011_v2').resolve()
sys.path.insert(0, str(code))
spec = importlib.util.spec_from_file_location('diagnostic_parent_queue', code / 'queue.py')
queue = importlib.util.module_from_spec(spec); spec.loader.exec_module(queue)
protocol = queue.verify_tools()
out = Path(args.output).resolve()
assert not out.exists()
out.mkdir()
assert out.is_relative_to(Path('outputs').resolve())
prefix = out / 'fixture_prefix'
seed = 1

def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value))

def expect_reject(name, mutate):
    originals = {}
    def remember(path):
        originals[Path(path)] = Path(path).read_bytes()
    mutate(remember)
    try:
        queue.verify_candidates(str(prefix), seed, protocol)
    except (AssertionError, FileNotFoundError):
        cases.append(dict(case=name, rejected_before_any_launch=True))
    else:
        raise AssertionError('Invalid fixture was accepted: ' + name)
    finally:
        for path, value in originals.items():
            path.write_bytes(value)

for bundle, arm in queue.paths(str(prefix), seed):
    native_code = bundle.parent.parent / 'code'
    source = Path('英文小论文/code') / native_code.parent.name
    native_code.mkdir(parents=True)
    allowed = protocol['allowed_native_roots'][source.name]
    for name in allowed['files']:
        shutil.copyfile(source / name, native_code / name)
    assert {p.name: queue.sha(p) for p in native_code.iterdir()} == allowed['files']
    native = queue.read(native_code / 'training_protocol.json')
    plan = dict(arm=arm, seed=seed, schema=native['schema'], max_epochs=12)
    dump(bundle / 'plan.json', plan); dump(bundle / 'resolved.yml', dict(fixture=True))
    identity = dict(arm=arm, seed=seed, plan_sha256=queue.sha(bundle / 'plan.json'),
        implementation=allowed['files'], protocol_sha256=queue.sha(native_code / 'training_protocol.json'))
    screen = bundle.parent; run = screen / 'run'
    dump(bundle / 'receipt.json', dict(identity=identity, resolved_sha256=queue.sha(bundle / 'resolved.yml'),
        paths=dict(screen_root=str(screen), output=str(run))))
    dump(screen / 'controller_start.json', dict(pid=2147483647, start_ticks='0'))
    dump(screen / 'candidate_start.json', dict(child_pid=2147483647, child_start_ticks='0'))
    dump(screen / 'controller_completion.json', dict(status='COMPLETE_FIXED12'))
    dump(screen / 'candidate_completion.json', dict(exit_code=0, timed_out=False))
    run.mkdir(); (run / 'last.pth').write_bytes(b'SYNTHETIC_BINDING_FIXTURE_NOT_MODEL_WEIGHTS')
    dump(run / 'endpoint_epoch12/manifest.json', dict(fixture=True))
    dump(run / 'stages_epoch12/manifest.json', dict(fixture=True))
    dump(run / 'training_completion.json', dict(status='NATIVE_FIT_FINISHED_EXACT12', completed_epochs=12,
        entered_epochs=list(range(12)), identity=identity, last_checkpoint_sha256=queue.sha(run / 'last.pth'),
        endpoint_manifest_sha256=queue.sha(run / 'endpoint_epoch12/manifest.json'),
        stage_manifest_sha256=queue.sha(run / 'stages_epoch12/manifest.json')))
    dump(screen / 'post_training/completion.json', dict(status='NATIVE_STEM_SCRATCH_FIXED12_RELOAD_AND_COLLECTION_FINISHED'
        if arm == 'native_stem_scratch' else 'STEM_FIXED12_RELOAD_AND_COLLECTION_FINISHED'))
    dump(screen / 'candidate_reload_v1/verification.json', dict(status='PASS_SAVED_CANDIDATE_RELOAD'))
    dump(screen / 'candidate_reload_v1/completion.json', dict(exit_code=0, timed_out=False))

jobs = queue.verify_candidates(str(prefix), seed, protocol)
assert [j['arm'] for j in jobs] == ['original_stem', 'native_stem_scratch', 'highfreq_rgb']
cases = [dict(case='all_three_distinct_bindings_checked', passed=True)]
bundles = queue.paths(str(prefix), seed)
def missing_second(remember):
    path = bundles[1][0].parent / 'post_training/completion.json'; remember(path); path.unlink()
expect_reject('second_configuration_closeout_missing', missing_second)
def wrong_third_seed(remember):
    path = bundles[2][0] / 'plan.json'; remember(path); value = queue.read(path)
    value['seed'] = 2; dump(path, value)
expect_reject('third_configuration_wrong_seed', wrong_third_seed)
def cap(remember):
    path = bundles[2][0] / 'plan.json'; remember(path); value = queue.read(path)
    value['max_epochs'] = 13; dump(path, value)
expect_reject('third_configuration_exceeds_epoch_cap', cap)
def source_change(remember):
    path = bundles[2][0].parent.parent / 'code/runtime.py'; remember(path)
    path.write_bytes(path.read_bytes() + b'\n# fixture source mismatch\n')
expect_reject('third_configuration_source_mismatch', source_change)
def incomplete_reload(remember):
    path = bundles[1][0].parent / 'candidate_reload_v1/completion.json'; remember(path)
    dump(path, dict(exit_code=1, timed_out=False))
expect_reject('second_configuration_saved_EMA_reload_failed', incomplete_reload)
def false_endpoint(remember):
    path = bundles[2][0].parent / 'run/training_completion.json'; remember(path)
    value = queue.read(path); value['entered_epochs'] = list(range(11)); dump(path, value)
expect_reject('third_configuration_has_no_exact12_endpoint', false_endpoint)
assert len(queue.verify_candidates(str(prefix), seed, protocol)) == 3
assert not any(name in sys.modules for name in ('torch', 'runtime', 'engine'))
old = Path('英文小论文/code/native_selection_diagnostic12_20261011_v1')
capture = {name: queue.sha(code / name) == queue.sha(old / name) for name in
           ('selection_trace.py', 'selection_stage_trace.py', 'validate_integration_cpu.py')}
assert all(capture.values())
report = dict(status='PASS_ALL_THREE_JSON_BINDINGS_AND_NEGATIVE_GUARD_CASES', cases=cases,
    queue_sha256=queue.sha(code / 'queue.py'), diagnostic_protocol_sha256=queue.sha(code / 'diagnostic_protocol.json'),
    capture_sources_unchanged=capture, native_runtime_or_torch_imported=False,
    model_forwards=0, GPU_calls=0, optimizer_updates=0, epochs_added=0,
    scope='Synthetic metadata fixtures exercise production parent verification only; no actual checkpoints or detection accuracy verified.')
dump(out / 'verification.json', report)
print(json.dumps(dict(status=report['status'], cases=len(cases), model_forwards=0, source_bindings=3)))
