"""Freeze an independent six-record consumer; never edit live experiment sources."""
from pathlib import Path
import hashlib, json
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
diag = '英文小论文/code/native_selection_diagnostic12_20261011_v3/'
stage = '英文小论文/code/stem_init_stage_analysis_20261011_v2/'
aggregate = '英文小论文/code/native_selection_analysis12_20261011_v1/analyze.py'
dependencies = [diag + n for n in ('diagnostic_queue.py', 'import_smoke.py', 'replay.py', 'selection_trace.py', 'selection_stage_trace.py',
                                      'validate_integration_cpu.py', 'diagnostic_protocol.json')]
dependencies += [stage + 'compare_saved_stages.py', stage + 'protocol.json', aggregate,
    '英文小论文/code/native_selection_analysis12_20261011_v1/selection_core.py',
    '英文小论文/code/p2_candidate_seed1_full132_20261007_v1/engine/deim/dfine_decoder.py']
protocol = dict(schema='actual_selection_offline_analysis_v3',
    files={n: sha(HERE / n) for n in ('selection_core.py', 'analyze.py', 'validate_cpu.py', 'register_protocol.py')},
    dependencies={n: sha(ROOT / n) for n in dependencies}, stage_analysis_source=stage + 'compare_saved_stages.py',
    aggregate_source=aggregate, diagnostic_protocol_sha256=sha(ROOT / (diag + 'diagnostic_protocol.json')),
    diagnostic_replay_sha256=sha(ROOT / (diag + 'replay.py')), selection_stage_trace_sha256=sha(ROOT / (diag + 'selection_stage_trace.py')),
    required_pairs=[dict(arm=a, kind=k, seed=s) for a, k in
        (('original_stem', 'full'), ('native_stem_scratch', 'reference'), ('highfreq_rgb', 'control')) for s in (1, 2)],
    fixed_endpoint=dict(epochs=12, zero_based_epoch=11, weights='EMA'),
    neighborhood_radii=[0, 1], small_area_interval=[0, 1024], small_area_upper_inclusive=True,
    original_four_record_consumer_unchanged=True, native_selected_indices_preserved=True,
    no_dense_bbox_head_readout=True, all_six_formal_terminals_and_exact_replays_required=True,
    scope='All548 images and38759 validGT per arm/seed. Nearest-cell/3x3 windows describe spatial neighborhoods, '
          'not query ownership, recall or AP. Actual native TopK IDs and tied score intervals are preserved. '
          'Dense anchors are unregressed priors; only selected300 boxes are observed. Same-GT comparisons '
          'cover scratch-to-full and RGB-to-full separately; queries are never aligned across runs. '
          'TP flags use native all-area matching stratified by GT size. Associations and earliest observed '
          'geometry changes do not isolate a causal Stem/attention mechanism or decompose AP. '
          'No model, evaluator, optimizer or additional training calls; cached BASE evidence reused only.')
with (HERE / 'protocol.json').open('x', encoding='utf8') as f:
    json.dump(protocol, f, ensure_ascii=False, indent=2, allow_nan=False)
print('SIX_RECORD_OFFLINE_PROTOCOL_SEALED')
