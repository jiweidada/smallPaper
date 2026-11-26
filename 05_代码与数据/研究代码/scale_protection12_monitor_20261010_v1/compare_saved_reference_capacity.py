"""Compare two saved reference supports with exact physical matching certificates."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path.cwd()
BASE = ROOT / 'outputs/saved_fgl_coefficient_analysis_20261010_v2/reference_stage_trace_v1'
OUT = BASE / 'capacity_comparison_v1'
SRC = Path(__file__).with_name('trace_saved_reference_support.py')
read = lambda p: json.loads(Path(p).read_text(encoding='utf8'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest = read(BASE / 'manifest.json')
assert sha(SRC) == manifest['source_sha256']
assert all(sha(BASE / name) == row['sha256'] for name, row in manifest['files'].items())
report = read(BASE / 'report.json')
assert all(sha(ROOT / name) == digest for name, digest in report['source_files'].items())
nodes = [node for node in ast.parse(SRC.read_text(encoding='utf8')).body
         if isinstance(node, ast.FunctionDef) and node.name in ('required_distances', 'certified_max_matching')]
assert len(nodes) == 2
# Import only the SHA-bound numeric functions; never execute the old file's work.
namespace = {'np': np}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SRC), 'exec'), namespace)
distance, matching = namespace['required_distances'], namespace['certified_max_matching']
certificates = read(BASE / 'tiny_query_matching_certificates.json')
rows, initial_certificates = [], []
for old in report['per_image']:
    seed, batch, slot = (old[k] for k in ('seed', 'batch', 'slot'))
    folder = ROOT / f'outputs/native_augmented_supervision_20261010_v1/collection_seed{seed}/capture_v1/endpoint12/batch_{batch:03d}'
    fm = read(folder / 'manifest.json')
    assert all(sha(folder / name) == digest for name, digest in fm['files'].items())
    ev = read(folder / 'evidence.json')
    _, _, h, w = ev['metadata']['input_shape']
    with np.load(folder / 'arrays.npz', allow_pickle=False) as z:
        gt = z[f'target.{slot}.boxes']
        tiny = np.flatnonzero((gt[:, 2:].astype(np.float64) * np.array([w, h])).min(1) < 4)
        edges = distance(z['model.enc_aux_outputs.0.pred_boxes'][slot], gt)[tiny] < 4 - 1e-6
        assigned, cover_g, cover_q = matching(edges)
    previous = [row for row in certificates if (row['seed'], row['batch'], row['slot']) == (seed, batch, slot)]
    assert len(previous) == 1 and previous[0]['tiny_GT_rows'] == tiny.tolist()
    pre_count = sum(query >= 0 for query in previous[0]['matched_query_rows'])
    assert pre_count == old['certified_max_distinct_queries_tiny_only']
    initial_count = int((assigned >= 0).sum())
    rows.append(dict(seed=seed, batch=batch, slot=slot, tiny_GT=len(tiny),
        encoder_reference_proxy_max_distinct=initial_count, native_pre_reference_max_distinct=pre_count,
        difference=pre_count-initial_count))
    initial_certificates.append(dict(seed=seed, batch=batch, slot=slot, tiny_GT_rows=tiny.tolist(),
        matched_query_rows=assigned.tolist(), minimum_vertex_cover_GT_rows=tiny[cover_g].tolist(),
        minimum_vertex_cover_query_rows=np.flatnonzero(cover_q).tolist(), graph_sha256=hashlib.sha256(edges.tobytes()).hexdigest()))
totals = []
for seed in (1, 2):
    selected = [row for row in rows if row['seed'] == seed]
    totals.append(dict(seed=seed, images=len(selected), **{k:sum(row[k] for row in selected) for k in
        ('tiny_GT', 'encoder_reference_proxy_max_distinct', 'native_pre_reference_max_distinct', 'difference')}))
value = dict(status='SAVED_ENCODER_PROXY_VS_NATIVE_PRE_REFERENCE_CAPACITY_CERTIFIED', totals=totals, per_image=rows,
    parent_manifest_sha256=sha(BASE/'manifest.json'), numeric_helper_source_sha256=sha(SRC),
    encoder_support_counterfactual_not_native_encoder_limit=True,
    class_agnostic_tiny_only_matching_not_AP_or_native_cost_assignment=True,
    existing_train_mode_augmented_BASE12_fixture_not_actual_history=True,
    optimality_certificates=len(initial_certificates), model_forwards=0, backwards=0, epochs=0, optimizer_steps=0)
OUT.mkdir(exist_ok=False)
def write(p, v):
    with p.open('x', encoding='utf8') as stream:
        json.dump(v, stream, ensure_ascii=False, indent=2, allow_nan=False)
assert 'torch' not in sys.modules
write(OUT/'report.json', value)
write(OUT/'encoder_proxy_matching_certificates.json', initial_certificates)
write(OUT/'manifest.json', dict(files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in OUT.iterdir() if p.is_file()}, source_sha256=sha(__file__)))
print(json.dumps(dict(status=value['status'], totals=totals, manifest_sha256=sha(OUT/'manifest.json'))), flush=True)
