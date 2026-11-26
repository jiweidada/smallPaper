"""Seal the new offline analysis version, without modifying any live experiment."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files={n:sha(HERE/n) for n in ('selection_core.py','analyze.py','validate_cpu.py','register_protocol.py')}
diagnostic=ROOT/'英文小论文/code/native_selection_diagnostic12_20261011_v1'
dependencies=['英文小论文/code/native_selection_diagnostic12_20261011_v1/diagnostic_protocol.json',
 '英文小论文/code/native_selection_diagnostic12_20261011_v1/replay.py',
 '英文小论文/code/native_selection_diagnostic12_20261011_v1/selection_stage_trace.py',
 '英文小论文/code/scale_protection12_monitor_20261010_v1/geometry_core.py',
 '英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py',
 '英文小论文/code/stem_init_stage_analysis_20261011_v2/protocol.json',
 '英文小论文/code/p2_candidate_seed1_full132_20261007_v1/engine/deim/dfine_decoder.py']
protocol=dict(schema='actual_selection_offline_analysis_v1',files=files,dependencies={n:sha(ROOT/n) for n in dependencies},
 diagnostic_protocol_sha256=sha(diagnostic/'diagnostic_protocol.json'),diagnostic_replay_sha256=sha(diagnostic/'replay.py'),
 selection_stage_trace_sha256=sha(diagnostic/'selection_stage_trace.py'),
 terminal_status={'original_stem':'PASS_ORIGINAL_STEM_FIXED12_DOWNLOADED_EVIDENCE_AND_SAVED_EMA_RELOAD',
                  'native_stem_scratch':'PASS_NATIVE_STEM_SCRATCH_FIXED12_DOWNLOADED_EVIDENCE_AND_SAVED_EMA_RELOAD'},
 required_pairs=[dict(arm=a,seed=s) for a in ('original_stem','native_stem_scratch') for s in (1,2)],
 fixed_endpoint=dict(epochs=12,zero_based_epoch=11,weights='EMA'),grid_shapes=[[80,80],[40,40],[20,20]],
 neighborhood_radii=[0,1],native_selected_indices_preserved=True,no_dense_bbox_head_readout=True,
 scope='Post-training spatial neighborhood/rank and selected-box geometry descriptions across all548 images and all validGT. Nearest-cell/3x3 windows are proxies, not GT ownership or IoU coverage. Actual native TopK indices and tied score intervals are preserved; selected boxes may regress far from their anchors. Full/ref same-GT joins do not align queries between runs. Association does not isolate a causal Stem mechanism, AP decomposition, or training supervision. No new model/evaluator/optimizer calls.')
with (HERE/'protocol.json').open('x',encoding='utf8') as f:json.dump(protocol,f,ensure_ascii=False,indent=2,allow_nan=False)
print('OFFLINE_ANALYSIS_PROTOCOL_SEALED')

