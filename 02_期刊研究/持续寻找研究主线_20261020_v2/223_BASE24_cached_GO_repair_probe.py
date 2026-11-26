"""222: exact monotone GO repair on fixed BASE24 development caches only.

Written only; root reviews/runs. Never load checkpoints, heldout or models.
G0 is the saved CPU native-union ownership, not a newly chosen baseline.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import ast
import copy
import gc
import hashlib
import importlib.util
import logging
import pathlib
import sys
import time
import traceback
from fractions import Fraction
from typing import Dict

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent
WORK = ROOT.parents[1]
PROTOCOL = ROOT / '222_双BASE24既有开发缓存修复_事前协议.txt'
REUSE = ROOT / '209_C0_monotone_GO_repair.py'
REUSE_SHA = '648c407ccc86f24f88c4b54be90e0b256afa0799bab4d7df0206e90ba587c973'
if hashlib.sha256(REUSE.read_bytes()).hexdigest() != REUSE_SHA:
    raise RuntimeError('209 changed')
_spec = importlib.util.spec_from_file_location('_audit223_reuse209', REUSE)
if _spec is None or _spec.loader is None:
    raise RuntimeError('209 import unavailable')
J = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(J)  # Only definitions; never an old main/data reader.
M, H, A, B = J.M, J.H, J.A, J.B
np, torch, psutil = J.np, J.torch, J.psutil
require, sha, save = J.require, J.sha, J.save
BASE = WORK / '英文小论文/data/08_匹配聚合覆盖探针_20261019_v1'
DATA = BASE / 'isolated_v3'
CODE = WORK / '英文小论文/code/matching_coverage_probe_20261019_v1'
ARMS = ('G0', 'G_repair', 'U_cost', 'U_change', 'ref_IoU_face', 'final_IoU_face', 'native_cost_face')
DOMAINS = M.DOMAINS
COHORTS = ('original_sub4', 'actual_input_sub4', 'normalized_640_equivalent_sub4')
COHORT_COUNTS = ('assigned_slots','support_slots','repaired_from_cohort','repaired_to_cohort','lost_from_cohort','changed_from_cohort')
SHAPES = ROOT / '228_BASE24开发输入shape_只读元数据.json'
SHAPES_SHA = 'd775a4bdf496ad93221eb76e7dd9455d29a8ad387edc534417d6f056046f4f7c'
CP_SHA = {1: '741d57a2d6ecb7439de17fedbc8462315ab05309da2ba35b9d38d2c3e2dd9d87',
          2: 'bcf1d9a7a20115835d9e9df3e09a71419f41d1612c2e7b687c47ff3c41952cf0'}
# Exact path/hash whitelist; no directory discovery or alternative inputs.
INPUT_HASHES = {
 'A/frozen_predictions/development/batch00_predictions.pt': '5b925b87a5db5c73f4b5733e4b5a94c693d81d606f879e5d3cc4a32b72bc1e01',
 'A/development/batch00_analysis.json': 'ee5039a0743014b3acd38f7273b8da803651dae53681f95de615fb48f834a97b',
 'A/frozen_predictions/development/batch01_predictions.pt': '7fe40e89f0219662d23c6f95d187a6c1b6d6029c35bfb3310c0b82664cfc5dbc',
 'A/development/batch01_analysis.json': '10a6353a3885d89ba301501bf13bde98974f040f170c707c0d41ccac9bcb2af9',
 'A/frozen_predictions/development/batch02_predictions.pt': '722ef29d82cebbfb0f126f61c0c7ee9bc377e4e8caab7a40ad30d08ef74812b2',
 'A/development/batch02_analysis.json': '7e96203809e86ab3e3e5c33ff6263a71ca87de70e55a01c92e338e6551e1c354',
 'A/frozen_predictions/development/batch03_predictions.pt': '8d6ed2514db94b8d0a648fb447aa898e3974243c7bbbd3230e12ce3edbaf0782',
 'A/development/batch03_analysis.json': 'b65fd9dbdde04feea4bf3b197ea19d21ecefbc6b06361a0689a41affcc62ca3a',
 'A/frozen_predictions/development/batch04_predictions.pt': '9692313860e396ff141a0d66ceb428b7a07de0dcfc59940d5fd85eaf96e82e2f',
 'A/development/batch04_analysis.json': 'b64a9d32ed345cfc5c55446a1ac70fe4e40f0438ee0532ea3648650d7612b439',
 'A/frozen_predictions/development/batch05_predictions.pt': 'a42ec02feed3188640912f90987099b35836ae5251d78ed14428219d6b55e3ff',
 'A/development/batch05_analysis.json': '24d54637054236a683bfc41c1804e8be548fb289b7b6c18d53c69dc955b2ecc8',
 'A/frozen_predictions/development/batch06_predictions.pt': '966ba319d5879b2fe5be01bae982d2799e524bb343b7f2ecabdb12f91e1815d6',
 'A/development/batch06_analysis.json': '4332a361732c9f80d0352bf1fb7b96693fd854f434e6b5224802b1348e7ac596',
 'A/frozen_predictions/development/batch07_predictions.pt': '29d559b12c5caeb12568a15e5a10994e97403ca51417cffd27deadf9787cf518',
 'A/development/batch07_analysis.json': '45256c13ebdb85ff6e06567bc20e3b233bb622ecfadeed83686bf39525cb30f7',
 'B/frozen_predictions/development/batch00_predictions.pt': '567616a0f1e888a1a9f20cdd3509d911883b0f749ae611b8ba7901d93bab6195',
 'B/development/batch00_analysis.json': '176a9538d11b109713b8d8542c9f534715991abbf84dbc608d89b9a704e1c2b9',
 'B/frozen_predictions/development/batch01_predictions.pt': '0fd9a538bafc1e698c286ae1bb6f71a85383f7322c4ca6279926e7080b71e5b1',
 'B/development/batch01_analysis.json': 'ea199d818844d2aeed4789f2d40ff29933a381020e4411e0b3f6bc6a203905e1',
 'B/frozen_predictions/development/batch02_predictions.pt': 'c56eb8bebb8211af6f2a6b824e018f0c4d9cec0c6715cabc70a6fabdc757d5fa',
 'B/development/batch02_analysis.json': '5e4d0dbfe52b08f738da7fe555f5a9725e42cffbe4457ab11d711eb424646b1b',
 'B/frozen_predictions/development/batch03_predictions.pt': '55780a77b2d1f048c31c6ef8eab23dc5863d960f619cf8241604fcd6119dee9f',
 'B/development/batch03_analysis.json': '772f737b7ed13e4c05f56707a9cf0c75344cf53a6f60e789656da708fbecf56e',
 'B/frozen_predictions/development/batch04_predictions.pt': '85c9819536ee317cde94693b5390aa4aefa082f40d34fbffd77fb105bf850e3b',
 'B/development/batch04_analysis.json': '8c49c3ab9db53ada6e35c3fedb487cde6809922113863f82f74f21e462f76944',
 'B/frozen_predictions/development/batch05_predictions.pt': 'c9c1db7d46766a2d21d18aa0bee92bf635ccceffe83b58ff9ed962b741bd29d5',
 'B/development/batch05_analysis.json': '4c206e6bac1f2b852ebe98d9227c83d1e87c555c054c4c114c4e52e3dcbaf344',
 'B/frozen_predictions/development/batch06_predictions.pt': '9dd0922a33cd489a3b7e81c720dca6a4741a08c1ff1337f0dc342c002241d8ca',
 'B/development/batch06_analysis.json': '2be01d060cefb4d139a4a65655aa30c9ef84dc9e09b299e1979e78c45cb49b85',
 'B/frozen_predictions/development/batch07_predictions.pt': 'f39d5c799444b7f25d1e074e2bd0482b3bb38e3b0262755de9289f259780d7dd',
 'B/development/batch07_analysis.json': '10284ec8151effd876db3c827d691b235141da4bd954007983020106d75b9f6b',
 'A/frozen_predictions/development/seal.json': 'dfd098301473dd62c0278ce17ebf7e65d1de3b0c22b6ee2d176e584697fabf07',
 'B/frozen_predictions/development/seal.json': '33f58f4dc71b5551d3005aa4431cad92d0c70a930853df3ea4cb6e0ddc385720',
 'A/development/complete.json': '6ad7a4d4dca65638f7412041dbb9b4a74fabfc9f1984a0ded31c258492dc6fb3',
 'B/development/complete.json': 'e73847368e0e8592855aafa4c94c540ccba371e4efa8551e6aab38bc1b62c32e',
 'generation_anchor/A/complete.json': '6cf87f562c1f0790b5254c77ff8ea99c12eceed833b50088803a4b8fce8a3dbe',
 'generation_anchor/B/complete.json': 'b5c7ebfa0d47579e8b88d075dacae9373576fd03137c6f6f547f89dd5c11c598',
}
PRODUCERS = {
 'probe_forward_v3.py': 'b1711ea14740014536cf8f3d9958c889ed100236927ef82e38b16b1df9bb9c8a',
 'evaluate_rules.py': 'c8e4ca196cb8f33b225ee0133849058eb65c1f1a2707236051f85de8d89c95b8',
 'analyze_partition_v3.py': 'f6c3d9530e0937075ae74316f6e18d28790461688dd5b241e2449bcc7f93f0d9',
 'probe_runtime.py': 'b37debc377341195ace499324eb49c5c67d255c65c7ffcfbd593723dfaac41d0',
 'diagnostic_extension_v2.py': 'e7a3becd950ae3525eee936893298cb1916b3a951b8431e6ce712974f1d3a364',
 'make_inputs.py': '7ea1f227955b759db8fc0cbf7ab7a94253dabe0a476b20f5d6cda17b3a946db5',
}


def manifest():
    files = {'script': pathlib.Path(__file__), 'protocol': PROTOCOL, 'reuse209': REUSE,
             'reuse192': J.REUSE, 'reuse183': M.REUSE, 'reuse159': B.REUSE, 'helper136': A.HELPER,
             'network_simplex': pathlib.Path(M.nx.__file__).parent / 'algorithms/flow/networksimplex.py',
             'recipe': BASE / 'baseline_recipes.json', 'package': BASE / 'package_manifest.json', 'input_shapes228': SHAPES}
    expected = {'reuse209': REUSE_SHA, 'reuse192': J.REUSE_SHA, 'reuse183': M.REUSE_SHA,
        'reuse159': B.REUSE_SHA, 'helper136': A.HELPER_SHA, 'network_simplex': M.NX_SHA,
        'recipe': '46e5c21aa0308061ac3523871c3c9966309a0a26b4a0777245edff4c3997f4bf',
        'package': '7c4b4c3a6968acf7f5269cc8070caebba24a1b9e5466d41532a366b6b46ad02a', 'input_shapes228': SHAPES_SHA}
    for key, filename in [('matcher','matcher.py'), ('criterion','deim_criterion.py'),
                          ('box_ops','box_ops.py'), ('dfine_utils','dfine_utils.py')]:
        files[key], expected[key] = A.NATIVE / filename, A.EXPECTED[key]
    for relative, digest in INPUT_HASHES.items():
        require('heldout' not in relative and (relative.startswith('generation_anchor/') or '/development/' in relative), 'path whitelist')
        files['input:' + relative], expected['input:' + relative] = DATA / relative, digest
    for name, digest in PRODUCERS.items():
        files['producer:' + name], expected['producer:' + name] = CODE / name, digest
    return files, expected


def read_json(path):
    return H.json.loads(path.read_text(encoding='utf-8-sig'))


def native_readers(box, config):
    observed = []
    def solver(cost):
        require(torch.is_tensor(cost) and cost.dtype == torch.float32 and bool(torch.isfinite(cost).all()), 'native FP32 cost')
        observed.append(cost.detach().cpu().numpy().copy())
        return A.linear_sum_assignment(cost)
    namespace = {'__name__': '_audit223_native', 'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional,
        'np': np, 'os': os, 'Dict': Dict, 'linear_sum_assignment': solver, 'register': lambda: (lambda cls: cls),
        'logger': logging.getLogger('audit223'), 'VIS_MATCH': False, 'point_sample': A.forbidden}
    for name in ('batch_dice_loss','batch_sigmoid_ce_loss','box_cxcywh_to_xyxy','generalized_box_iou','box_iou','generalized_box_inner_iou'):
        namespace[name] = getattr(box, name)
    tree = ast.parse((A.NATIVE / 'matcher.py').read_text(encoding='utf-8-sig'))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'HungarianMatcher')
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(A.NATIVE / 'matcher.py'), 'exec'), namespace)
    kwargs = copy.deepcopy(config['DEIMCriterion']['matcher'])
    require(kwargs.pop('type') == 'HungarianMatcher' and config['num_classes'] == 10 and config['use_focal_loss'] is True, 'BASE24 matcher recipe')
    matcher = namespace['HungarianMatcher'](**kwargs, use_focal_loss=True)
    tree = ast.parse((A.NATIVE / 'deim_criterion.py').read_text(encoding='utf-8-sig'))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'DEIMCriterion')
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '_get_go_indices')
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(A.NATIVE / 'deim_criterion.py'), 'exec'), namespace)
    def compute(output, targets):
        observed.clear()
        indices = matcher({'pred_boxes': output['pred_boxes'], 'pred_logits': output['pred_logits']}, targets, epoch=24)['indices']
        require(len(observed) == len(targets), 'native batch cost capture')
        return list(observed), indices
    return compute, namespace['_get_go_indices']


def aggregate(items):
    stats = {arm: {key: sum((M.fraction_read(x['metrics'][arm][key]) for x in items), Fraction(0))
                   for key in M.METRICS} for arm in ARMS}
    counts = {arm: {key: sum(x['transitions'][arm][key] for x in items) for key in J.COUNTS} for arm in ARMS}
    differences = {arm: sum(x['owner_difference_vs_repair'][arm] for x in items) for arm in ARMS}
    cohorts = {}
    for name in COHORTS:
        parts = [x['cohort_transitions'][name] for x in items]
        cohorts[name] = None if any(x is None for x in parts) else {
            arm:{key:sum(x[arm][key] for x in parts) for key in COHORT_COUNTS} for arm in ARMS}
    return {'metrics': M.report_metrics(stats), 'transitions': counts, 'owner_difference_vs_repair': differences,
            'cohort_transitions':cohorts, 'G_repair_same_U_change_exact_optimal_face': J.same_face(stats['G_repair'], stats['U_change'])}


def input_shapes():
    data = read_json(SHAPES)
    require(data['scope'] == 'development input metadata only' and data['heldout_reads'] is False
        and data['model_builds'] == data['model_forwards'] == data['optimizer_steps'] == 0, 'shape collector scope')
    rows = {int(x['batch']):x for x in data['rows']}
    require(len(data['rows']) == 8 and set(rows) == set(range(8)), 'eight development input shapes')
    for batch,row in rows.items():
        size = 512 if batch == 3 else (544 if batch == 5 else 640)
        require(row['group'] == 'development' and row['source_unchanged'] is True and row['shape'] == [4,3,size,size]
            and pathlib.Path(row['path']).resolve() == (BASE/'shared_inputs/inputs'/f'batch{batch:02d}.pt').resolve(), 'fixed actual input shape')
    return rows


def screen(summary):
    tr = summary['transitions']
    repaired = tr['G_repair']['repaired_old_outside'] > 0
    different = {arm: tr[arm]['forbidden_edges'] > 0 for arm in ('U_cost', 'U_change')}
    face = summary['G_repair_same_U_change_exact_optimal_face']
    require(tr['G_repair']['lost_old_inside'] == tr['G_repair']['forbidden_edges'] == tr['G_repair']['changed_still_outside'] == 0, 'monotonicity')
    return {'necessary_noncopy': repaired and all(different.values()) and not face,
            'repairs_original_outside': repaired, 'unrestricted_control_uses_forbidden_edges': different,
            'same_U_change_exact_optimal_face': face, 'training_authorized': False,
            'scope': 'development cached constraint premise only, not learning/AP/novelty'}


def metadata(letter, seed):
    seal = read_json(DATA / letter / 'frozen_predictions/development/seal.json')
    complete = read_json(DATA / letter / 'development/complete.json')
    generation = read_json(DATA / 'generation_anchor' / letter / 'complete.json')
    require(seal['group'] == complete['group'] == 'development' and seal['batches'] == complete['batches'] == 8
            and seal['inputs'] == complete['formal_inputs'] == 32 and complete['state'] == generation['state'] == 'complete', 'complete development partitions')
    require(complete['seed'] == generation['seed'] == seed and generation['checkpoint_sha256'] == CP_SHA[seed]
            and generation['optimizer_steps'] == generation['model_gradients'] == complete['optimizer_steps'] == 0
            and generation['BN_and_state_restored'] is True, 'generation identity')
    require(generation['partition_seal_sha256']['development'] == complete['input_seal_sha256']
            == INPUT_HASHES[f'{letter}/frozen_predictions/development/seal.json'], 'generation seal identity')
    predrows = {int(x['batch']): x for x in seal['files']}
    analysisrows = {int(x['batch']): x for x in complete['files']}
    require(set(predrows) == set(analysisrows) == set(range(8)) and len(seal['files']) == len(complete['files']) == 8, 'eight exact batches')
    for batch in range(8):
        p, a = predrows[batch], analysisrows[batch]
        require(p['relative'] == f'batch{batch:02d}_predictions.pt' and a['relative'] == f'batch{batch:02d}_analysis.json'
            and p['sha256'] == INPUT_HASHES[f'{letter}/frozen_predictions/development/{p["relative"]}']
            and a['sha256'] == INPUT_HASHES[f'{letter}/development/{a["relative"]}']
            and p['source_ids'] == a['source_ids'], 'sealed batch identity')
    provenance = {'seed': seed, 'checkpoint_sha256_from_original_generation': CP_SHA[seed], 'checkpoint_loaded': False,
        'weight_key': 'raw[model]', 'checkpoint_last_epoch': 23, 'matching_epoch_argument': 24,
        'original_forward': {'whole_model': 'train', 'autograd': False, 'AMP': False, 'TF32': False,
                            'BN_and_RNG_restored_each_batch': True, 'torch': generation['torch']},
        'G0_source': 'saved CPU native union reconstructed from frozen GPU predictions; not live GPU criterion',
        'development_only': True, 'independent_validation': False, 'single_checkpoint_variable_causality': False}
    return predrows, analysisrows, provenance


def validate_batch(value, prior, batch, seed, predrow, recipe, compute, union):
    require(value['group'] == prior['group'] == 'development' and value['tag'] == prior['tag'] == f'batch{batch:02d}'
            and prior['seed'] == seed and prior['prediction_sha256'] == predrow['sha256']
            and prior['source_ids'] == predrow['source_ids'], 'batch content identity')
    output, targets = value['outputs'], value['targets']
    require(len(targets) == len(prior['inputs']) == len(prior['branch_matches']) == 4, 'batch4 schema')
    require(output['pred_boxes'].shape == output['ref_points'].shape == (4, 300, 4)
            and output['pred_logits'].shape == (4, 300, 10), 'prediction schema')
    for tensor in (output['pred_boxes'], output['ref_points'], output['pred_logits']):
        require(tensor.device.type == 'cpu' and tensor.dtype == torch.float32 and not tensor.requires_grad
                and bool(torch.isfinite(tensor).all()), 'CPU frozen FP32')
    require(torch.equal(output['ref_points'], output['pre_outputs']['pred_boxes']), 'own pre/FDR refs')
    require(all(torch.equal(output['ref_points'], aux['ref_points']) for aux in output['aux_outputs']), 'shared FDR refs')
    names = ['final'] + [f'aux_{i}' for i in range(len(output['aux_outputs']))] + ['pre'] + [f'enc_{i}' for i in range(len(output['enc_aux_outputs']))]
    require(prior['branch_names'] == names == ['final','aux_0','aux_1','pre','enc_0'], 'native GO branches')
    matches = [[tuple(torch.tensor(x, dtype=torch.int64) for x in row[name]) for row in prior['branch_matches']] for name in names]
    rebuilt = union(None, matches[0], matches[1:])
    costs, rematched = compute(output, targets)
    original, l1_terms = [], []
    for i, (target, saved, own, again, oldunion) in enumerate(zip(targets, prior['inputs'], matches[0], rematched, rebuilt)):
        gt, labels = target['boxes'], target['labels']
        require(gt.dtype == torch.float32 and gt.device.type == 'cpu' and gt.shape == (len(labels),4)
            and labels.dtype == torch.int64 and labels.device.type == 'cpu' and bool(torch.isfinite(gt).all())
            and bool((gt[:,2:] > 0).all()) and bool(((labels >= 0) & (labels < 10)).all()), 'GT schema')
        require(all(torch.equal(a,b) for a,b in zip(own,again)), 'same-cache own final Hungarian reproduction')
        saved_u = saved['branches']['final']['native_union']
        qs, gs = np.asarray(saved_u['pair_query'], dtype=np.int64), np.asarray(saved_u['pair_GT'], dtype=np.int64)
        require(qs.ndim == gs.ndim == 1 and qs.shape == gs.shape and len(set(qs.tolist())) == len(qs)
            and np.array_equal(qs, np.sort(qs)) and np.all((qs >= 0) & (qs < 300)) and np.all((gs >= 0) & (gs < len(gt))), 'saved G0 indices')
        require(saved['input_within_batch'] == i and saved['GT'] == saved_u['GT'] == len(gt)
            and saved['K'] == saved_u['pairs'] == len(qs)
            and saved_u['covered'] == saved['native_covered'] == len(set(gs.tolist()))
            and saved_u['GT_repetitions'] == np.bincount(gs, minlength=len(gt)).tolist(), 'saved G0 budgets')
        require(dict(zip(qs.tolist(),gs.tolist())) == dict(zip(oldunion[0].tolist(),oldunion[1].tolist())), 'saved native union reproduction')
        original.append(np.stack((qs,gs), axis=1))
        l1_terms.append((output['pred_boxes'][i,qs] - gt[gs]).abs().reshape(-1))
    go_denominator = max(sum(len(p) for p in original), 1)
    ordinary_denominator = max(sum(len(t['labels']) for t in targets), 1)
    l1 = float(torch.cat(l1_terms).sum() / go_denominator * recipe['DEIMCriterion']['weight_dict']['loss_bbox'])
    expected = float(prior['batch_losses']['native_union']['losses']['loss_bbox'])
    require(abs(l1-expected) <= 1e-5 + 1e-5*abs(expected), 'native final L1/GO denominator replay')
    denominator_audit = {'GO': go_denominator, 'ordinary_GT': ordinary_denominator, 'all_arms_fixed': True,
        'GO_rule': 'max(sum image K,1), single CPU process', 'ordinary_rule': 'max(sum image GT,1)',
        'native_weighted_final_L1_replay': l1, 'saved_native_weighted_final_L1': expected,
        'absolute_difference': abs(l1-expected), 'fixed_FP32_reduction_atol_rtol': [1e-5,1e-5]}
    return output, targets, original, costs, denominator_audit


def solve_domain(F, v, iou, ref_iou, cost, caps, owner):
    selected, certs, allowed, _ = J.solve_new(F, cost, caps, owner)
    selected['G0'], certs['G0'] = owner.copy(), {'source': 'saved original CPU native union; not an optimum claim'}
    outside = (~F).astype(np.int64)
    for arm, objective in [('ref_IoU_face',-ref_iou), ('final_IoU_face',-iou), ('native_cost_face',cost)]:
        selected[arm], certs[arm] = M.exact_flow([('primary',objective), ('outside',outside)], caps, owner)
    stats = {arm: M.metrics_for(selected[arm],F,v,iou,ref_iou,cost,owner) for arm in ARMS}
    transitions = {arm: J.transition_counts(F,owner,selected[arm]) for arm in ARMS}
    require(tuple(stats['U_change'][k] for k in J.TUPLE) <= tuple(stats['G_repair'][k] for k in J.TUPLE), 'restricted beats unrestricted exact optimum')
    for arm in ARMS:
        require(np.bincount(selected[arm],minlength=len(caps)).tolist() == caps.tolist(), 'GT multiplicity changed')
    require(all(transitions['G_repair'][k] == 0 for k in ('lost_old_inside','changed_still_outside','forbidden_edges')), 'repair contract')
    return selected, {'metrics': M.report_metrics(stats), 'transitions': transitions,
        'owner_difference_vs_repair': {arm:int((selected[arm] != selected['G_repair']).sum()) for arm in ARMS},
        'G_repair_same_U_change_exact_optimal_face': J.same_face(stats['G_repair'], stats['U_change']), 'certificates':certs}, allowed


def analyze_image(index, output, target, original, cost_all, recipe, native, box, saved, matrices, input_shape):
    M.guard()
    boxes, refs, gt, labels = output['pred_boxes'][index], output['ref_points'][index], target['boxes'], target['labels'].numpy()
    require(bool((boxes[:,2:] > 0).all()) and bool((refs[:,2:] > 0).all()), 'valid final/ref widths')
    rm = int(recipe['DFINETransformer']['reg_max'])
    require(rm == int(recipe['DEIMCriterion']['reg_max']) == 32, 'reg_max identity')
    up, scale = H.scalar(output['up'],'up'), abs(H.scalar(output['reg_scale'],'reg_scale'))
    require(up == .5 and scale == float(recipe['DFINETransformer']['reg_scale']) > 0, 'native support configuration')
    up32, scale32 = torch.tensor([up],dtype=torch.float32), torch.tensor([scale],dtype=torch.float32)
    nodes = native.weighting_function(rm,up32,scale32)
    a, b = float(nodes.min()), float(nodes.max())
    require(a == -scale and b == scale and bool((nodes[1:] > nodes[:-1]).all()), 'native physical endpoints')
    tolerance = B.NEAR_REL*max(1.,scale)
    old_sub4 = target.get('original_sub4')
    if old_sub4 is not None:
        require(torch.is_tensor(old_sub4) and old_sub4.dtype == torch.bool and old_sub4.shape == (len(gt),), 'explicit original_sub4 schema')
        old_sub4 = old_sub4.numpy()
    wh64 = gt[:,2:].double().numpy()
    actual = (wh64 * np.array([input_shape[3],input_shape[2]],dtype=np.float64)).min(axis=1) < 4.
    current = wh64.min(axis=1) * 640 < 4.
    masks = {'original_sub4':old_sub4,'actual_input_sub4':actual,'normalized_640_equivalent_sub4':current}
    def cohort(g):
        return {'original_sub4': None if old_sub4 is None else bool(old_sub4[g]),
                'actual_input_sub4':bool(actual[g]), 'normalized_640_equivalent_sub4': bool(current[g])}
    all_iou = box.box_iou(box.box_cxcywh_to_xyxy(boxes),box.box_cxcywh_to_xyxy(gt))[0].numpy()
    all_ref_iou = box.box_iou(box.box_cxcywh_to_xyxy(refs),box.box_cxcywh_to_xyxy(gt))[0].numpy()
    require(np.allclose(all_iou[original[:,0],original[:,1]],np.asarray(saved['branches']['final']['native_union']['pair_iou']),atol=3e-5,rtol=0), 'saved G0 final IoU replay')
    owner_map, groups, rows = dict(original.tolist()), [], []
    for cls in sorted(set(int(labels[g]) for g in owner_map.values())):
        qs = np.array(sorted(q for q,g in owner_map.items() if labels[g] == cls),dtype=np.int64)
        gs, caps = np.unique([owner_map[q] for q in qs],return_counts=True)
        owner = np.searchsorted(gs,[owner_map[q] for q in qs])
        require(np.all(labels[gs] == cls) and int(caps.sum()) == len(qs), 'fixed class/q/GT budgets')
        refs64, gt64 = refs[qs].double().numpy(), gt[gs].double().numpy()
        raw = B.raw_physical64(refs64,gt64,scale)
        raw32, legacy, interpolation = M.encoding_rect(native,refs64,gt64,rm,up32,scale32)
        margin = B.margin(raw,a,b)
        graphs = {'physical64':margin >= 0., 'tight':margin >= tolerance, 'expanded':margin >= -tolerance,
                  'native32':B.margin(raw32,a,b) >= 0.}
        v = (np.maximum(a-raw,0.) + np.maximum(raw-b,0.)).sum(axis=-1)/scale
        iou, ref_iou, cost = [x[np.ix_(qs,gs)] for x in (all_iou,all_ref_iou,cost_all)]
        domains, solved = {}, {}
        for domain,F in graphs.items():
            equal = next((d for d in solved if np.array_equal(F,graphs[d])),None)
            if equal is None:
                selected, details, allowed = solve_domain(F,v,iou,ref_iou,cost,caps,owner)
            else:
                selected, details, allowed = solved[equal]
            solved[domain] = (selected,details,allowed)
            gt_support = {arm:[{'GT':int(g), 'm_g':int(cap), 'support_slots':int(F[np.arange(len(qs)),selected[arm]][selected[arm] == c].sum()),
                'support_rate':float(F[np.arange(len(qs)),selected[arm]][selected[arm] == c].sum()/cap), **cohort(g)}
                for c,(g,cap) in enumerate(zip(gs,caps))] for arm in ARMS}
            require(all(new['support_slots'] >= old['support_slots'] for new,old in zip(gt_support['G_repair'],gt_support['G0'])), 'per-GT support monotonicity')
            cohort_transitions = {}
            old_in = F[np.arange(len(qs)),owner]
            for name,mask in masks.items():
                if mask is None:
                    cohort_transitions[name] = None
                    continue
                cohort_transitions[name] = {}
                old_cohort = mask[gs[owner]]
                for arm,col in selected.items():
                    new_in, new_cohort = F[np.arange(len(qs)),col], mask[gs[col]]
                    repaired, lost = ~old_in & new_in, old_in & ~new_in
                    cohort_transitions[name][arm] = {'assigned_slots':int(new_cohort.sum()),
                        'support_slots':int((new_cohort & new_in).sum()),
                        'repaired_from_cohort':int((repaired & old_cohort).sum()),
                        'repaired_to_cohort':int((repaired & new_cohort).sum()),
                        'lost_from_cohort':int((lost & old_cohort).sum()),
                        'changed_from_cohort':int(((col != owner) & old_cohort).sum())}
            domains[domain] = {**details, 'pairs':{arm:[[int(q),int(gs[c])] for q,c in zip(qs,selected[arm])] for arm in ARMS},
                'GT_support':gt_support, 'cohort_transitions':cohort_transitions, 'certificate_reused_from_domain':equal,
                'certificates':details['certificates'] if equal is None else None}
        selected, _, allowed = solved['physical64']
        F = graphs['physical64']
        for r,q in enumerate(qs):
            rows.append({'image_index':index, 'query':int(q), 'class_id':cls, 'original_GT':int(gs[owner[r]]),
                'original_cohort':cohort(gs[owner[r]]), 'arms':{arm:{'GT':int(gs[selected[arm][r]]), **cohort(gs[selected[arm][r]]),
                    'old_in':bool(F[r,owner[r]]), 'new_in':bool(F[r,selected[arm][r]]),
                    'changed':bool(selected[arm][r] != owner[r]), 'forbidden':bool(not allowed[r,selected[arm][r]]),
                    'current_IoU':float(iou[r,selected[arm][r]]), 'ref_IoU':float(ref_iou[r,selected[arm][r]]),
                    'native_cost':float(cost[r,selected[arm][r]]), 'v':float(v[r,selected[arm][r]])} for arm in ARMS}})
        groups.append({'class_id':cls, 'q_ids':qs, 'GT_ids':gs, 'm_g':caps, 'domains':domains,
            'near_boundary_edges':int((np.minimum(np.abs(raw-a),np.abs(raw-b)).min(axis=-1) <= tolerance).sum()),
            'physical64_native32_disagreement':int((F != graphs['native32']).sum()),
            'native_interpolation_max_error':interpolation, 'raw32_minus_physical64_max_abs':float(np.max(np.abs(raw32.astype(np.float64)-raw))),
            'legacy_eps64_minus_physical64_max_abs':float(np.max(np.abs(legacy-raw)))})
        prefix = f'image{index}_class{cls}_'
        matrices.update({prefix+k:x for k,x in {'q_ids':qs,'GT_ids':gs,'m_g':caps,'owner_cols':owner,'ref64':refs64,'GT64':gt64,
            'raw_physical64':raw,'raw_native32':raw32,'v':v,'current_IoU32':iou,'ref_IoU32':ref_iou,'native_cost32':cost,**graphs}.items()})
    summaries = {d:aggregate([g['domains'][d] for g in groups]) for d in DOMAINS}
    for domain in DOMAINS:
        for arm in ARMS:
            pairs = [p for g in groups for p in g['domains'][domain]['pairs'][arm]]
            mapping = dict(pairs)
            require(len(pairs) == len(mapping) == len(original) and set(mapping) == set(owner_map)
                and sorted(mapping.values()) == sorted(owner_map.values())
                and all(labels[g] == labels[owner_map[q]] for q,g in mapping.items()), 'full image budgets')
    return {'image_index':index, 'GT':len(gt), 'slots':len(original), 'unique_GT':len(set(original[:,1].tolist())),
        'cohort':{'original_sub4':None if old_sub4 is None else int(old_sub4.sum()), 'actual_input_sub4':int(actual.sum()),
                  'normalized_640_equivalent_sub4':int(current.sum()), 'actual_input_shape':input_shape},
        'support':{'reg_max':rm,'up':up,'scale':scale,'endpoints':[a,b],'near_boundary_tolerance':tolerance},
        'groups':groups, 'domains':summaries}, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=pathlib.Path,default=ROOT/'224_BASE24_cached_GO_repair_probe')
    args = parser.parse_args()
    args.out = args.out.resolve()
    require(args.out.parent == ROOT and args.out.name.startswith('224_') and not args.out.exists(), 'new local224 output only')
    args.out.mkdir(exist_ok=False)
    files, expected = manifest()
    before, artifacts, batches, provenance = {}, {}, [], {}
    resources, checks, loaded = None, None, 0
    started, stage = time.monotonic(), 'resources'
    result = {'state':'error'}
    try:
        resources = H.resource_limits()
        require(resources['priority'] == 'below_normal' and torch.version.cuda is None
                and M.nx.__version__ == '3.3' and A.scipy.__version__ == '1.13.1', 'CPU/package contract')
        stage = 'fixed_path_hash_manifest_before_content'
        before = {key:sha(path) for key,path in files.items()}
        require(all(before[key] == digest for key,digest in expected.items()), 'fixed source/input hashes')
        save(args.out/'00_input_manifest.json',{'paths':{k:str(p.resolve()) for k,p in files.items()},'sha256':before,
             'tensor_or_result_content_read':False,'heldout_file_reads':0})
        stage = 'synthetic_before_real_content'
        native = H.load_native_math()
        box = sys.modules[native.__package__+'.box_ops']
        checks = {'physical_support':B.self_checks(native), 'integer_repair':J.self_checks()}
        stage = 'metadata_after_synthetic'
        recipes, package = read_json(files['recipe']), read_json(files['package'])
        shapes = input_shapes()
        package_hashes = {x['relative']:x['sha256'] for x in package['files']}
        for key in ('matcher','criterion','box_ops','dfine_utils'):
            require(package_hashes['native/engine/deim/'+files[key].name] == before[key], 'original package/native math identity')
        for letter,seed in (('A',1),('B',2)):
            predrows, analysisrows, provenance[letter] = metadata(letter,seed)
            recipe = recipes[str(seed)]
            provenance[letter]['recipe'] = {k:recipe[k] for k in ('DEIMCriterion','DFINETransformer','use_focal_loss','eval_spatial_size')}
            compute, union = native_readers(box,recipe)
            for batch in range(8):
                M.guard()
                require(predrows[batch]['input_sha256'] == shapes[batch]['sha256'], 'actual shape/input seal identity')
                stage = f'{letter}_development_batch{batch:02d}'
                predpath = DATA/letter/'frozen_predictions/development'/predrows[batch]['relative']
                analysispath = DATA/letter/'development'/analysisrows[batch]['relative']
                require(sha(predpath) == predrows[batch]['sha256'] and sha(analysispath) == analysisrows[batch]['sha256'], 'batch hashes before read')
                prior = read_json(analysispath)
                value = torch.load(predpath,map_location='cpu',weights_only=False)
                loaded += 1
                output, targets, original, costs, denom = validate_batch(value,prior,batch,seed,predrows[batch],recipe,compute,union)
                images, query_rows, matrices = [], [], {}
                for index in range(4):
                    image, rows = analyze_image(index,output,targets[index],original[index],costs[index],recipe,native,box,prior['inputs'][index],matrices,shapes[batch]['shape'])
                    images.append(image)
                    query_rows.extend(rows)
                require(sum(x['slots'] for x in images) == sum(len(p) for p in original) == len(query_rows), 'batch slot count')
                summaries = {d:aggregate([x['domains'][d] for x in images]) for d in DOMAINS}
                compact = {'seed':seed,'letter':letter,'batch':batch,'input_archive_sha256':predrows[batch]['input_sha256'],
                    'prediction_sha256':predrows[batch]['sha256'],'source_ids':predrows[batch]['source_ids'],
                    'forward_DN_seed':int(value['forward_DN_seed']),'denominators':denom,'actual_input_shape':shapes[batch]['shape'],
                    'GT':sum(x['GT'] for x in images),'slots':sum(x['slots'] for x in images),
                    'domains':summaries,'per_image':[{k:x[k] for k in ('image_index','GT','slots','unique_GT','cohort','domains')} for x in images]}
                stem = f'{letter}_batch{batch:02d}'
                save(args.out/(stem+'_assignments.json'),{**compact,'images':images,'queries':query_rows})
                np.savez_compressed(args.out/(stem+'_matrices.npz'),**matrices)
                for suffix in ('_assignments.json','_matrices.npz'):
                    artifacts[stem+suffix] = sha(args.out/(stem+suffix))
                batches.append(compact)
                print(stage,'complete',flush=True)
                del value,prior,output,targets,original,costs,images,query_rows,matrices,image,rows
                gc.collect()
                M.guard()
        require(loaded == len(batches) == 16 and sum(len(x['per_image']) for x in batches) == 64, 'all fixed development exposures')
        seeds = {}
        for seed in (1,2):
            chosen = [x for x in batches if x['seed'] == seed]
            summaries = {d:aggregate([x['domains'][d] for x in chosen]) for d in DOMAINS}
            seeds[str(seed)] = {'batches':8,'input_exposures':32,'GT':sum(x['GT'] for x in chosen),
                'slots':sum(x['slots'] for x in chosen),'domains':summaries,'screens':{d:screen(summaries[d]) for d in DOMAINS}}
        flags = [seeds[str(seed)]['screens']['physical64']['necessary_noncopy'] for seed in (1,2)]
        result = {'state':'complete','primary':{'per_seed_necessary_noncopy':dict(zip(('1','2'),flags)),
            'status':'both' if all(flags) else ('mixed' if any(flags) else 'neither'),
            'training_authorized':False,'scope':'existing development cached premise only; 210 not inherited'},
            'seeds':seeds,'batches':batches,'provenance':provenance,'artifact_sha256':artifacts}
        stage = 'complete'
    except Exception as exc:
        result = {'state':'error','error':str(exc),'traceback':traceback.format_exc(),
            'scope':'implementation/provenance/precision/certificate failure; not a method negative',
            'completed_batches':batches,'artifact_sha256':artifacts}
    finally:
        after = {key:sha(files[key]) for key in before if files[key].exists()}
        unchanged = bool(before and before == after and set(before) == set(files))
        if not unchanged:
            result['state'], result['hash_error'] = 'error','source/input missing or changed'
        result.update(stage=stage,sources_unchanged=unchanged,sha256_before=before,sha256_after=after,
            resources=resources,synthetic_self_checks=checks,loaded_prediction_batches=loaded,heldout_file_reads=0,
            checkpoint_loads=0,model_forwards=0,model_backwards=0,model_vjps=0,optimizer_steps=0,new_training=False,
            flow_statistics=M.FLOW_STATS,RSS_observed_peak_bytes=M.RSS_PEAK,
            OS_peak_working_set_bytes=getattr(psutil.Process().memory_info(),'peak_wset',None),
            free_RAM_GiB_at_end=psutil.virtual_memory().available/2**30,seconds=time.monotonic()-started)
        save(args.out/'01_results.json',result)
        print(H.json.dumps(H.jsonable({'state':result['state'],'primary':result.get('primary'),'seconds':result['seconds']}),ensure_ascii=False),flush=True)
    if result['state'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
