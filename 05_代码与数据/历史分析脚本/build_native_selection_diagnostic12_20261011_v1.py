"""Build an isolated derivative of the pinned candidate-only saved-EMA replay."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[1];CODE=ROOT/'英文小论文/code/native_selection_diagnostic12_20261011_v1';OUT=ROOT/'outputs/native_selection_diagnostic12_20261011_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
observer=ROOT/'英文小论文/code/native_selection_trace_20261011_v1/selection_trace.py'
assert sha(observer)=='b4ce79bd59f76392a1b6b34c78a1f82dfd357bca01f51d92d43d6d395cc10cba'
with (CODE/'selection_trace.py').open('xb') as f:f.write(observer.read_bytes())
original=ROOT/'英文小论文/code/original_stem12_20261011_v1/verify_candidate_reload.py'
text=original.read_text(encoding='utf8'); changes=[]
def replace(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new);changes.append(dict(old=old,new=new))
replace("protocol = read(HERE/'closeout_protocol.json')", "protocol = read(HERE/'diagnostic_protocol.json')")
replace("assert protocol['schema'] == 'original_stem12_closeout_v1'", "assert protocol['schema'] == 'native_selection_diagnostic12_v1'")
replace("assert code.is_dir()", "assert code.is_dir() and code.parent.name in verify_tools()['allowed_native_roots']\n    allowed = verify_tools()['allowed_native_roots'][code.parent.name]\n    assert {p.name:sha(p) for p in code.iterdir() if p.is_file()} == allowed['files']")
replace("assert plan['arm'] == 'original_stem'", "assert plan['arm'] == allowed['arm'] and plan['seed'] in (1,2)")
replace("from stage_trace import StageTrace,TracedLoader,verify_trace", "from stage_trace import TracedLoader,verify_trace\n    from selection_stage_trace import SelectionStageTrace")
assert text.count("out = screen/'candidate_reload_v1'")==2
text=text.replace("out = screen/'candidate_reload_v1'","out = screen/'native_selection_diagnostic12_v1'")
changes.append(dict(old="out = screen/'candidate_reload_v1' (2 occurrences)",new="out = screen/'native_selection_diagnostic12_v1'"))
replace("payload['original_stem12_identity']", "payload[dict(original_stem='original_stem12_identity',native_stem_scratch='native_stem_scratch12_identity')[plan['arm']]]")
replace("trace = StageTrace(model,receipt['identity'],paths['val_ann']); model.eval()", "model.eval(); trace = SelectionStageTrace(model,receipt['identity'],paths['val_ann'])")
replace("trace.finish(out/'stages_epoch12')", "trace.finish(out/'stages_epoch12')\n        selection_manifest = trace.finish_selection(out/'selection_epoch12')\n        assert trace.selection_observer.calls == trace.selected_bbox_calls == forward_calls")
replace("verification_script_sha256=sha(__file__),closeout_protocol_sha256=sha(HERE/'closeout_protocol.json'),", "verification_script_sha256=sha(__file__),diagnostic_protocol_sha256=sha(HERE/'diagnostic_protocol.json'),\n            selection_manifest_sha256=sha(selection_manifest),selection_images=len(trace.selection_records),\n            actual_selector_calls=trace.selection_observer.calls,existing_selected_bbox_head_calls=trace.selected_bbox_calls,\n            dense_anchors_are_unregressed_priors=True,additional_bbox_head_or_topk_calls=0,")
replace("status='PASS_SAVED_CANDIDATE_RELOAD' if passed else 'CANDIDATE_RELOAD_MISMATCH_NO_RETRY'", "status='PASS_SELECTION_DIAGNOSTIC_EXACT_SAVED_EMA_REPLAY' if passed else 'SELECTION_DIAGNOSTIC_REPLAY_MISMATCH_NO_RETRY'")
with (CODE/'replay.py').open('x',encoding='utf8',newline='\n') as f:f.write(text)
for p in CODE.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
allowed={}
for root,arm in [('original_stem12_20261011_v1','original_stem'),('native_stem_scratch12_20261011_v3','native_stem_scratch')]:
    native=ROOT/'英文小论文/code'/root;files={p.name:sha(p) for p in native.iterdir() if p.is_file()}
    closeout=read(native/'closeout_protocol.json')
    assert all(files[n]==h for n,h in closeout['files'].items())
    allowed[root]=dict(arm=arm,files=files)
protocol=dict(schema='native_selection_diagnostic12_v1',candidate_forward_batches=137,validation_images=548,reload_timeout_seconds=1800,
    stats_fraction_tolerance=0.0,files={p.name:sha(p) for p in CODE.glob('*.py')},allowed_native_roots=allowed,
    original_replay_source_sha256=sha(original),original_stage_trace_sha256='d9668777da440da708ecb3779e6cefc4810abfe65480b6b11520463ac7b896a5',
    authorizes='One isolated candidate-only fixed12 EMA validation replay per completed enhanced/random-native arm and seed, after original training exits and GPU is idle.',
    maximum_training_epochs_added=0,pretrained_BASE_forwards=0,automatic_retry=False,
    equality_required='All12 native statistics, all native GT flags and all548 saved stage arrays must equal original training outputs exactly before selection records are valid.',
    excluded_claims=['8400 predicted boxes','Cross-run query identity','Selection neighborhood as geometric IoU coverage','Causal Stem benefit inferred from observation alone'])
write(CODE/'diagnostic_protocol.json',protocol);OUT.mkdir(exist_ok=False)
write(OUT/'build_manifest.json',dict(code_sha256=protocol['files'],protocol_sha256=sha(CODE/'diagnostic_protocol.json'),source_changes=changes,
    training_code_changed=False,actual_detector_replays=0,CPU_integration_checks_pending=True))
print(json.dumps(dict(status='ISOLATED_CANDIDATE_DIAGNOSTIC_REPLAY_PREPARED',files=len(protocol['files']),source_changes=len(changes))))
