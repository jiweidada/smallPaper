"""Bounded NumPy algebra controls, not author-model execution or fitted probes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'A14_新编码器位置保留近邻审计_v1'
PHASES = [(0, 0), (1, 0), (0, 1), (1, 1)]

def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def pack(x):
    return np.concatenate([x[:, :, a::2, b::2] for a, b in PHASES], axis=1)

def unpack(z, channels):
    b, _, h, w = z.shape
    x = np.zeros((b, channels, h * 2, w * 2), dtype=z.dtype)
    for phase, (a, c) in enumerate(PHASES):
        x[:, :, a::2, c::2] = z[:, phase*channels:(phase+1)*channels]
    return x

def conv(x, weight, stride, padding):
    """Explicit cross-correlation on a padded tensor."""
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    batch, _, h, w = xp.shape
    d, _, kh, kw = weight.shape
    oh, ow = (h-kh)//stride + 1, (w-kw)//stride + 1
    result = np.zeros((batch, d, oh, ow), dtype=np.float64)
    for i in range(oh):
        for j in range(ow):
            for n in range(batch):
                for o in range(d):
                    result[n, o, i, j] = np.sum(
                        xp[n, :, i*stride:i*stride+kh, j*stride:j*stride+kw] * weight[o])
    return result

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def attention(q, k, v):
    return softmax(q @ k.T / np.sqrt(k.shape[1])) @ v

def ln(x):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)

assert not (OUT / 'completed.json').exists(), 'Completed audit exists; do not rerun.'
OUT.mkdir(exist_ok=True)
inputs = []
for name in ['LPS', 'SPD', 'SAPA', 'LoftUp']:
    p = ROOT / 'comparison_sources_v1' / (name + '_A14_operator_source_receipt.json')
    receipt = json.loads(p.read_text(encoding='utf-8'))
    assert receipt['status'] == 'ALL_LISTED_OPERATOR_FILES_GIT_BLOB_VERIFIED'
    for f in receipt['files']:
        data = (Path(receipt['local']) / f['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == f['sha256']
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == f['git_blob_sha']
    inputs.append({'name': name, 'path': str(p), 'sha256': digest(p),
        'commit': receipt['commit'], 'files': len(receipt['files'])})
protocol = {'utc': datetime.now(timezone.utc).isoformat(),
    'status': 'FROZEN_BEFORE_ALGEBRA_CONTROL_RESULTS', 'source_receipts': inputs,
    'seeds': [1400, 1401, 1402], 'dtype': 'float64', 'tolerance': 1e-10,
    'questions': [
        'Is lossless phase packing plus an otherwise identical full reader only a permutation?',
        'Is the linear pack-then-conv operator expressible by a larger stride2 kernel?',
        'Does row selection commute with a fixed-query independent crossattention/pointwise stack?',
        'Does a dimension-reducing linear map after packing guarantee invertibility?'],
    'limits': [
        'No author-code execution,weights,detector forward,datafitortraining.',
        'No measured real-image aliasing,encoderfailure,APornewalgorithmgain.',
        'Linear conversion excludes other reorderings/intermediate nonlinearities in a full network.',
        'Query subset equality assumes identical precomputed query inputs,fixedkeyvalues,deterministiceval andnoquery-querycoupling;no end-to-end sparseLoftUp speedclaim.'],
    'source_code_sha256': digest(Path(__file__))}
write('algebra_protocol.json', protocol)

rows = []
for seed in protocol['seeds']:
    rng = np.random.default_rng(seed)
    for batch, c, d, h, w in [(1, 1, 1, 6, 8), (1, 3, 2, 8, 6), (2, 2, 4, 6, 8)]:
        x = rng.normal(size=(batch, c, h, w))
        z = pack(x)
        assert np.array_equal(unpack(z, c), x)
        for kernel in [1, 3]:
            weights = rng.normal(size=(d, 4*c, kernel, kernel))
            dense = np.empty((d, c, 2*kernel, 2*kernel))
            for phase, (a, b) in enumerate(PHASES):
                dense[:, :, a::2, b::2] = weights[:, phase*c:(phase+1)*c]
            coarse = conv(z, weights, stride=1, padding=kernel//2)
            original = conv(x, dense, stride=2, padding=2*(kernel//2))
            error = float(np.max(np.abs(coarse-original)))
            assert error < protocol['tolerance']
            rows.append({'kind': 'linear_pack_conv_reindex', 'seed': seed,
                'shape': list(x.shape), 'kernel': kernel, 'max_error': error})

    # Same feature/key/value/position rows, only reindexed by their phase label.
    f = rng.normal(size=(8, 6, 8))
    coords = np.stack(np.meshgrid(np.arange(8), np.arange(6), indexing='ij'), axis=-1)
    flat = np.concatenate([f, coords], axis=-1).reshape(-1, 10)
    idx = np.arange(48).reshape(8, 6)
    order = np.concatenate([idx[a::2, b::2].reshape(-1) for a, b in PHASES])
    key = flat @ rng.normal(size=(10, 8))
    value = flat @ rng.normal(size=(10, 5))
    query = rng.normal(size=(7, 8))
    dense_result = attention(query, key, value)
    packed_result = attention(query, key[order], value[order])
    error = float(np.max(np.abs(dense_result-packed_result)))
    negative_error = float(np.max(np.abs(dense_result-attention(query, key[order], value))))
    assert error < protocol['tolerance'] and negative_error > 1e-7
    rows.append({'kind': 'phase_relabelled_reader', 'seed': seed,
        'max_error': error, 'inconsistent_value_permutation_control': negative_error})

    # The inspected LoftUp attention stack has no query-to-query interaction.
    q = rng.normal(size=(17, 8)); k = rng.normal(size=(11, 8)); v = rng.normal(size=(11, 8))
    w1 = rng.normal(size=(8, 12)); w2 = rng.normal(size=(12, 8))
    keep = np.array([0, 3, 9, 16])
    def stack(qi):
        for _ in range(2):
            qi = qi + attention(ln(qi), ln(k), ln(v))
            qi = qi + np.maximum(ln(qi) @ w1, 0) @ w2
        return ln(qi)
    error = float(np.max(np.abs(stack(q)[keep]-stack(q[keep]))))
    assert error < protocol['tolerance']
    rows.append({'kind': 'query_independent_stack_subset', 'seed': seed, 'max_error': error,
        'scope': 'Algebra controlforrowindependence;not sourceGELU/MHA reproduction or fullsparseupsampler.'})

    # A narrower projection has a nullspace even though packing itself is exact.
    projection = rng.normal(size=(5, 12))
    _, _, vh = np.linalg.svd(projection, full_matrices=True)
    null = vh[-1]
    error = float(np.max(np.abs(projection @ null)))
    assert error < protocol['tolerance'] and np.linalg.norm(null) > .99
    rows.append({'kind': 'compression_nullspace', 'seed': seed, 'max_error': error,
        'nonzero_input_norm': float(np.linalg.norm(null))})

write('algebra_cases.json', rows)
summary = {'utc': datetime.now(timezone.utc).isoformat(),
    'status': 'PASS_SCOPED_OPERATOR_EQUIVALENCE_AND_NONINVERTIBILITY_CONTROLS',
    'case_counts': {kind: sum(r['kind']==kind for r in rows) for kind in sorted({r['kind'] for r in rows})},
    'max_equal_operator_error': max(r['max_error'] for r in rows),
    'verified_git_blobs': sum(i['files'] for i in inputs),
    'decision': 'GENERIC_PHASE_PROVENANCE_OR_QUERY_SUBSETTING_IS_NOT_A_DISTINCT_OPERATOR',
    'real_image_mechanism_validated': False, 'mainline_validated': False,
    'new_full_model_forwards': 0, 'cumulative_full_model_forwards': 213,
    'new_fits': 0, 'main_model_updates': 0, 'limits': protocol['limits']}
write('algebra_summary.json', summary)
write('completed.json', {'status': 'COMPLETED_SOURCE_AND_SCOPED_ALGEBRA_ONLY',
    'protocol_sha256': digest(OUT/'algebra_protocol.json'),
    'summary_sha256': digest(OUT/'algebra_summary.json'),
    'cases_sha256': digest(OUT/'algebra_cases.json'), 'new_model_runs': 0})
print(json.dumps(summary, ensure_ascii=False))
