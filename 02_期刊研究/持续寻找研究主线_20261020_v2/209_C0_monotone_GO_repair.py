"""208: exact cached monotone GO repair and two unrestricted controls.

Written only, root reviews before execution. No model/checkpoint/gradient.
Forbidden edges form the first exact integer priority; G0 proves feasibility.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import hashlib
import importlib.util
import itertools
import pathlib
import sys
import time
import traceback
from fractions import Fraction

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent
PROTOCOL = ROOT / '208_单调可达修复_事前协议.txt'
REUSE = ROOT / '192_C0_support_assignment_exact.py'
REUSE_SHA = '07255fb486d12b3ef25acead31a3f707bdc558de55dc9d21202809f469bdd1c7'
OLD = ROOT / '193_C0_support_assignment_exact'
OLD_HASHES = {
    '00_results.json': '660884b5412275dccdf16e91ddb979346ad869b1153c11a5642f2ed680c9b4bc',
    '02_true_GO_assignments_and_certificates.json': 'cfc41cbb0409c46041b003a0b7b805e3a41b26ebb3672e364a2b3eb17934509e',
    '03_pairwise_graphs.npz': 'f21a2a9fed59153e55b147c535a01c8832091ca07dd8e5fa1b6dcf9a7c9673ad',
}
if hashlib.sha256(REUSE.read_bytes()).hexdigest() != REUSE_SHA:
    raise RuntimeError('192 changed')
_spec = importlib.util.spec_from_file_location('_audit209_reuse192', REUSE)
if _spec is None or _spec.loader is None:
    raise RuntimeError('192 import unavailable')
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)  # Definitions/imports only, no previous main.
H, A, B = M.H, M.A, M.B
np, torch, psutil = M.np, M.torch, M.psutil
require, sha, save = M.require, M.sha, M.save
ARMS = ('G0', 'G', 'ref_IoU_face', 'U_cost', 'U_change', 'G_repair')
COUNTS = ('repaired_old_outside', 'lost_old_inside', 'changed_still_outside', 'forbidden_edges', 'changed_owners')
TUPLE = ('outside', 'owner_changes', 'native_cost')


def solve_new(F, cost, caps, owner):
    change = np.arange(F.shape[1])[None, :] != owner[:, None]
    allowed = F | ~change
    outside = (~F).astype(np.int64)
    specifications = {
        'U_cost': [('outside', outside), ('native_cost', cost)],
        'U_change': [('outside', outside), ('priority_owner_changes', change.astype(np.int64)), ('native_cost', cost)],
        'G_repair': [('forbidden_edges', (~allowed).astype(np.int64)), ('outside', outside),
                     ('priority_owner_changes', change.astype(np.int64)), ('native_cost', cost)],
    }
    cols, certificates = {}, {}
    for arm, objectives in specifications.items():
        cols[arm], certificates[arm] = M.exact_flow(objectives, caps, owner)
    require(bool(allowed[np.arange(len(owner)), cols['G_repair']].all()), 'repair selected a forbidden edge')
    require(M.fraction_read(certificates['G_repair']['objectives'][0]['exact_sum']) == 0, 'hardcost priority not zero')
    return cols, certificates, allowed, specifications


def transition_counts(F, owner, selected):
    rows = np.arange(len(owner))
    old_in, new_in = F[rows, owner], F[rows, selected]
    changed = selected != owner
    return {'repaired_old_outside': int((~old_in & new_in).sum()), 'lost_old_inside': int((old_in & ~new_in).sum()),
            'changed_still_outside': int((~old_in & ~new_in & changed).sum()),
            'forbidden_edges': int((changed & ~new_in).sum()), 'changed_owners': int(changed.sum())}


def same_face(left, right):
    return tuple(left[key] for key in TUPLE) == tuple(right[key] for key in TUPLE)


def self_checks():
    # No real cache or historical result content has been read at this point.
    examples = [
        (np.zeros((2, 2), dtype=bool), np.array([[0., -1e100], [-1e100, 0.]]), [1, 1], [0, 1]),
        (np.array([[0, 1], [1, 1]], dtype=bool), np.array([[0., 3.], [2., 0.]]), [1, 1], [0, 1]),
        (np.array([[1, 1, 0], [0, 0, 0], [1, 0, 0]], dtype=bool), np.zeros((3, 3)), [1, 1, 1], [0, 1, 2]),
        (np.array([[0, 1], [1, 0], [1, 0]], dtype=bool), np.zeros((3, 2)), [2, 1], [0, 0, 1]),
    ]
    answers = []
    for F, cost, caps, original in examples:
        owner = np.asarray(original, dtype=np.int64)
        selected, certs, allowed, specs = solve_new(F, cost, caps, owner)
        feasible = [p for p in itertools.product(range(F.shape[1]), repeat=len(owner))
                    if np.bincount(p, minlength=F.shape[1]).tolist() == caps]
        require(any(np.array_equal(p, owner) for p in feasible) and bool(allowed[np.arange(len(owner)), owner].all()), 'G0 feasible')
        for arm, objectives in specs.items():
            full = objectives + [('owner_changes', np.arange(F.shape[1])[None, :] != owner[:, None])]
            def objective(p):
                return tuple(M.exact_sum(np.asarray(values)[np.arange(len(owner)), p]) for _, values in full)
            require(objective(selected[arm]) == min(objective(p) for p in feasible), 'exact flow differs from brute force')
            require(certs[arm]['dual_certificate']['zero_exact_gap'], 'synthetic certificate')
        repaired = selected['G_repair']
        counts = transition_counts(F, owner, repaired)
        require(counts['forbidden_edges'] == counts['lost_old_inside'] == counts['changed_still_outside'] == 0, 'support monotonicity')
        for g in range(F.shape[1]):
            require(int(F[np.arange(len(owner)), repaired][repaired == g].sum()) >= int(F[np.arange(len(owner)), owner][owner == g].sum()), 'GT support monotonicity')
        answers.append((selected, counts))
    require(answers[0][0]['G_repair'].tolist() == [0, 1], 'huge negative cost defeated hard constraint')
    require(answers[1][1]['repaired_old_outside'] == 1 and answers[3][1]['repaired_old_outside'] == 2, 'positive repair/multiplicity')
    require(answers[2][1]['repaired_old_outside'] == 0 and answers[2][0]['U_change'].tolist() != [0, 1, 2], 'restriction can reduce maximum support')
    left = {'outside': Fraction(1), 'owner_changes': Fraction(2), 'native_cost': Fraction(7, 3)}
    require(same_face(left, dict(left)), 'exact equal objective face')
    return {'integer_priority_and_dual': True, 'bruteforce_complete_assignments': True,
            'G0_always_feasible': True, 'huge_lower_cost_cannot_use_forbidden_edge': True,
            'positive_cycle_and_multi_GT_budget': True, 'restricted_support_upper_bound_can_drop': True,
            'query_and_GT_support_monotonicity': True, 'same_face_is_not_noncopy': True}


def cols_from_pairs(pairs, qs, gs):
    mapping = dict(pairs)
    require(len(mapping) == len(pairs) == len(qs) and set(mapping) == set(qs.tolist()), 'cached q ownership')
    cols = np.searchsorted(gs, [mapping[int(q)] for q in qs])
    require(np.all(cols < len(gs)) and np.array_equal(gs[cols], [mapping[int(q)] for q in qs]), 'cached GT ownership')
    return cols


def aggregate_metrics(groups):
    return {arm: {key: sum((M.fraction_read(g['metrics'][arm][key]) for g in groups), Fraction(0))
                  for key in M.METRICS} for arm in ARMS}


def aggregate_counts(groups):
    return {arm: {key: sum(g['transitions'][arm][key] for g in groups) for key in COUNTS} for arm in ARMS}


def analyze(cache, old, originals, graph_file):
    records = {int(r['image_index']): r for r in cache['records']}
    images, query_rows = [], []
    for image in old['images']:
        M.guard()
        index, record = int(image['image_index']), records[int(image['image_index'])]
        require(image['input_sha256'] == record['input_sha256'] and image['event'] == str(record['event']), '188/193 input identity')
        original = dict(originals[index].tolist())
        require(original == dict(image['original_pairs']), '193/188 GO ownership')
        labels = record['target']['labels'].numpy()
        groups = []
        for prior in image['groups']:
            cls = int(prior['class_id'])
            prefix = f'true_GO_image{index}_class{cls}_'
            data = {key: graph_file[prefix + key] for key in ('q_ids', 'GT_ids', 'capacities', 'owner_cols', 'physical64',
                'tight', 'expanded', 'native32', 'v64', 'final_IoU32', 'ref_IoU32', 'native_cost32', 'refs64', 'GT64')}
            qs, gs, caps, owner = [data[key] for key in ('q_ids', 'GT_ids', 'capacities', 'owner_cols')]
            F, v, iou, ref_iou, cost = [data[key] for key in ('physical64', 'v64', 'final_IoU32', 'ref_IoU32', 'native_cost32')]
            require(np.array_equal(qs, prior['queries']) and np.array_equal(gs, prior['GT_ids']) and np.array_equal(caps, prior['m_g'])
                    and len(set(qs.tolist())) == len(qs) and len(set(gs.tolist())) == len(gs), '193 group budget')
            require(np.array_equal(gs[owner], [original[int(q)] for q in qs]) and np.all(labels[gs] == cls)
                    and np.bincount(owner, minlength=len(gs)).tolist() == caps.tolist(), '188 group ownership/class/m_g')
            require(F.dtype == np.bool_ and F.shape == (len(qs), len(gs))
                    and all(np.array_equal(F, data[key]) for key in ('tight', 'expanded', 'native32'))
                    and prior['near_boundary_edges'] == prior['physical64_native32_edge_disagreements'] == 0, 'frozen identical physical domains')
            require(np.array_equal(data['refs64'], record['ref_points_initial'][qs].double().numpy())
                    and np.array_equal(data['GT64'], record['target']['boxes'][gs].double().numpy()), 'actual188 refs/GT')
            require(all(values.shape == F.shape and np.isfinite(values).all() for values in (v, iou, ref_iou, cost)), 'finite matrix schema')
            selected, certificates, allowed, _ = solve_new(F, cost, caps, owner)
            selected['G0'] = owner.copy()
            for arm in ('G', 'ref_IoU_face'):
                selected[arm] = cols_from_pairs(prior['domains']['physical64']['pairs'][arm], qs, gs)
                certificates[arm] = {'source': '193 existing exact certificate, not rerun',
                    'certificate': prior['domains']['physical64']['certificates'][arm]}
                require(certificates[arm]['certificate']['dual_certificate']['zero_exact_gap'] is True, '193 exact certificate')
            certificates['G0'] = {'source': '188 original GO, no optimality claim'}
            stats = {arm: M.metrics_for(selected[arm], F, v, iou, ref_iou, cost, owner) for arm in ARMS}
            transitions = {arm: transition_counts(F, owner, selected[arm]) for arm in ARMS}
            for arm in ('G0', 'G', 'ref_IoU_face'):
                require(all(stats[arm][key] == M.fraction_read(prior['domains']['physical64']['metrics'][arm][key]) for key in M.METRICS), '193 exact metric reproduction')
            require(tuple(stats['U_change'][key] for key in TUPLE) <= tuple(stats['G_repair'][key] for key in TUPLE), 'restricted objective beats unrestricted optimum')
            require(transitions['G_repair']['forbidden_edges'] == transitions['G_repair']['lost_old_inside'] == transitions['G_repair']['changed_still_outside'] == 0, 'repair contract')
            GT_support = {}
            for arm in ARMS:
                col = selected[arm]
                require(np.bincount(col, minlength=len(gs)).tolist() == caps.tolist(), 'm_g changed')
                GT_support[arm] = [{'GT': int(g), 'm_g': int(cap), 'support_slots': int(F[np.arange(len(qs)), col][col == j].sum())}
                                   for j, (g, cap) in enumerate(zip(gs, caps))]
            require(all(new['support_slots'] >= oldrow['support_slots'] for new, oldrow in zip(GT_support['G_repair'], GT_support['G0'])), 'per-GT support lost')
            for row, q in enumerate(qs):
                query_rows.append({'image_index': index, 'class_id': cls, 'query': int(q), 'original_GT': int(gs[owner[row]]),
                    'arms': {arm: {'GT': int(gs[selected[arm][row]]), 'original_sub4': bool(record['original_sub4'][int(gs[selected[arm][row]])]),
                        'changed': bool(selected[arm][row] != owner[row]), 'old_in': bool(F[row, owner[row]]), 'new_in': bool(F[row, selected[arm][row]]),
                        'forbidden_edge': bool(not allowed[row, selected[arm][row]]), 'v': float(v[row, selected[arm][row]]),
                        'final_IoU': float(iou[row, selected[arm][row]]), 'ref_IoU': float(ref_iou[row, selected[arm][row]]),
                        'native_cost': float(cost[row, selected[arm][row]])} for arm in ARMS}})
            groups.append({'class_id': cls, 'q_ids': qs, 'GT_ids': gs, 'm_g': caps,
                'metrics': M.report_metrics(stats), 'transitions': transitions, 'GT_support': GT_support,
                'pairs': {arm: [[int(q), int(gs[g])] for q, g in zip(qs, selected[arm])] for arm in ARMS},
                'G_repair_same_U_change_optimal_face': same_face(stats['G_repair'], stats['U_change']),
                'owner_difference_vs_repair': {arm: int((selected[arm] != selected['G_repair']).sum()) for arm in ARMS},
                'certificates': certificates})
        stats, counts = aggregate_metrics(groups), aggregate_counts(groups)
        for arm in ARMS:
            pairs = [pair for group in groups for pair in group['pairs'][arm]]
            mapping = dict(pairs)
            require(len(pairs) == len(mapping) == len(original) and set(mapping) == set(original)
                    and sorted(mapping.values()) == sorted(original.values())
                    and all(labels[g] == labels[original[q]] for q, g in mapping.items()), 'global fixed q/class/m_g')
        images.append({'image_index': index, 'event': image['event'], 'input_sha256': image['input_sha256'],
            'GT': image['GT'], 'slots': image['slots'], 'unique_GT': image['unique_GT'], 'groups': groups,
            'metrics': M.report_metrics(stats), 'transitions': counts,
            'G_repair_same_U_change_optimal_face': same_face(stats['G_repair'], stats['U_change'])})
        print('monotone repair image', index, 'complete', flush=True)
    totals, transitions = aggregate_metrics(images), aggregate_counts(images)
    require(sum(image['slots'] for image in images) == len(query_rows) == 832 and sum(image['unique_GT'] for image in images) == 749, 'realGO total budget')
    repaired = transitions['G_repair']['repaired_old_outside'] > 0
    controls = {arm: {'already_satisfies_E': transitions[arm]['forbidden_edges'] == 0,
                      'forbidden_edges': transitions[arm]['forbidden_edges']} for arm in ('U_cost', 'U_change')}
    equal_face = same_face(totals['G_repair'], totals['U_change'])
    gate = repaired and all(not row['already_satisfies_E'] for row in controls.values()) and not equal_face
    primary = {'necessary_gate': gate, 'repairs_at_least_one_original_outside': repaired, 'controls': controls,
        'same_U_change_exact_optimal_face': equal_face, 'algorithmic_monotonicity_verified': True,
        'decision': 'restricted_noncopy_premise_only_not_training' if gate else 'constraint_increment_not_supported_stop_this_exact_revision',
        'training_authorized': False, 'cannot_infer_learning_or_AP': True, '193_G_result_unchanged': True}
    return {'state': 'complete', 'primary': primary, 'images': images, 'metrics': M.report_metrics(totals), 'transitions': transitions,
            'scope': 'Closed FDR support monotonicity only, not prediction/classification/loss/AP monotonicity.'}, query_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=pathlib.Path, default=ROOT / '210_C0_monotone_GO_repair')
    args = parser.parse_args()
    args.out = args.out.resolve()
    require(args.out.parent == ROOT and args.out.name.startswith('210_') and not args.out.exists(), 'new local210 output required')
    args.out.mkdir(exist_ok=False)
    files = {'script': pathlib.Path(__file__), 'protocol': PROTOCOL, 'reuse192': REUSE, 'reuse183': M.REUSE,
             'reuse159': B.REUSE, 'helper136': A.HELPER, 'network_simplex': pathlib.Path(M.nx.__file__).parent / 'algorithms/flow/networksimplex.py',
             'collector187': M.COLLECTOR, 'GO_cache': M.GO / '01_cache.pt', 'GO_summary': M.GO / '02_summary.json',
             'GO_execution': M.GO / '04_execution.json', 'GO_manifest': M.GO / '05_output_sha256.json', 'GO_inputs': M.GO / '00_inputs_sha256.json',
             **{name: OLD / name for name in OLD_HASHES}}
    content_keys = set(OLD_HASHES) | {'GO_cache', 'GO_summary', 'GO_execution', 'GO_manifest', 'GO_inputs'}
    before, checks, resources, loaded = {}, None, None, False
    started, stage = time.monotonic(), 'resources'
    result = {'state': 'error', 'primary': {'necessary_gate': False}}
    try:
        resources = H.resource_limits()
        require(resources['priority'] == 'below_normal' and torch.version.cuda is None and M.nx.__version__ == '3.3'
                and A.scipy.__version__ == '1.13.1', 'CPU/package contract')
        before = {key: sha(path) for key, path in files.items() if key not in content_keys}
        expected = {'reuse192': REUSE_SHA, 'reuse183': M.REUSE_SHA, 'reuse159': B.REUSE_SHA, 'helper136': A.HELPER_SHA,
                    'network_simplex': M.NX_SHA, 'collector187': M.COLLECTOR_SHA}
        require(all(before[key] == value for key, value in expected.items()), 'dependency hashes')
        stage = 'synthetic_before_content'
        checks = self_checks()
        before.update({key: sha(files[key]) for key in content_keys})
        require(all(before[key] == value for key, value in OLD_HASHES.items()) and before['GO_cache'] == M.GO_SHA, '188/193 frozen hashes')
        stage, loaded = 'read_fixed_cached_evidence', True
        summary, execution = M.verify_collector(M.GO_SHA)
        prior_result = H.json.loads((OLD / '00_results.json').read_text(encoding='utf-8-sig'))
        require(prior_result['state'] == 'complete' and prior_result['sources_unchanged'] is True
                and prior_result['sha256_before']['script'] == REUSE_SHA
                and prior_result['sha256_before']['GO_cache188'] == M.GO_SHA, '193 producer identity')
        require(all(prior_result['artifact_sha256'][key] == OLD_HASHES[key] for key in ('02_true_GO_assignments_and_certificates.json', '03_pairwise_graphs.npz')), '193 artifact identity')
        cache = torch.load(M.GO / '01_cache.pt', map_location='cpu', weights_only=False)
        originals = M.verify_GO_cache(cache, summary, execution)
        prior = H.json.loads((OLD / '02_true_GO_assignments_and_certificates.json').read_text(encoding='utf-8-sig'))
        stage = 'exact_monotone_repair'
        with np.load(OLD / '03_pairwise_graphs.npz', allow_pickle=False) as graphs:
            report, rows = analyze(cache, prior, originals, graphs)
        del cache
        stage = 'save'
        save(args.out / '01_assignments_and_certificates.json', report)
        save(args.out / '02_per_query_transitions.json', {'queries': rows})
        result = {'state': 'complete', 'primary': report['primary'], 'metrics': report['metrics'], 'transitions': report['transitions'],
            'per_image': [{key: image[key] for key in ('image_index', 'slots', 'unique_GT', 'metrics', 'transitions', 'G_repair_same_U_change_optimal_face')} for image in report['images']],
            'artifact_sha256': {name: sha(args.out / name) for name in ('01_assignments_and_certificates.json', '02_per_query_transitions.json')}}
        stage = 'complete'
    except Exception as exc:
        result = {'state': 'error', 'primary': {'necessary_gate': False}, 'error': str(exc), 'traceback': traceback.format_exc(),
                  'scope': 'Implementation/provenance/certificate failure, not a method negative.'}
    finally:
        after = {key: sha(files[key]) for key in before if files[key].exists()}
        unchanged = bool(before and before == after)
        if result['state'] == 'complete':
            unchanged = unchanged and set(before) == set(files)
        if not unchanged:
            result['state'] = 'error'
            result['primary']['necessary_gate'] = False
            result['hash_error'] = 'source/input missing or changed'
        result.update(stage=stage, sources_unchanged=unchanged, source_paths={key: str(path.resolve()) for key, path in files.items()},
            sha256_before=before, sha256_after=after, resources=resources, synthetic_self_checks=checks, cache_content_loaded=loaded,
            flow_statistics=M.FLOW_STATS, RSS_observed_peak_bytes=M.RSS_PEAK,
            OS_peak_working_set_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', None),
            free_RAM_GiB_at_end=psutil.virtual_memory().available / 2**30, seconds=time.monotonic() - started,
            model_forwards=0, model_backwards=0, model_vjps=0, optimizer_steps=0, new_training=False)
        save(args.out / '00_results.json', result)
        print(H.json.dumps(H.jsonable({'state': result['state'], 'primary': result['primary'], 'seconds': result['seconds']}), ensure_ascii=False), flush=True)
    if result['state'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
