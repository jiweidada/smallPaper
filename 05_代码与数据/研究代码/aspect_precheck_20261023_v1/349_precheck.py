"""349 frozen CPU aspect precheck; exclusive writes and SHA-verified resume.
No training, optimizer, GPU calls, test-dev access, or edits to existing sources.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
import sys
sys.dont_write_bytecode = True
import argparse
import collections
import datetime
import gc
import hashlib
import importlib
import json
import time
import traceback
import uuid
from pathlib import Path
import numpy as np
import torch
import PIL
from PIL import Image
import torchvision.transforms.functional as TF
torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = ROOT / '持续寻找研究主线_20261020_v2'
WORK = ROOT / 'code/unit_instance_completion_20261021_v1'
STAGE3 = WORK / '342_codex_stage3_20261022_v1'
STAGE343 = WORK / '343_codex_stage4_3b_20261022_v1'
MAINRUNS = WORK / '348_cpu_training_20261023_v1/348_runs'
PLAN = SPEC / '349_候选1_保持宽高比_零训练预检_Codex执行计划.txt'
INDEX = SPEC / '327_训练实施输入索引.json'
REFERENCE = WORK / '344_codex_closeout_20261022_v1/344_result_EMA诊断_v2.json'
SHAPES = {'A0': (640, 640), 'A1': (832, 480), 'A2': (480, 832), 'A3': (960, 544)}
CHECKPOINT_SHAS = {1: '741d57a2d6ecb7439de17fedbc8462315ab05309da2ba35b9d38d2c3e2dd9d87',
                   2: 'bcf1d9a7a20115835d9e9df3e09a71419f41d1612c2e7b687c47ff3c41952cf0'}
for folder in (WORK, STAGE343):
    sys.path.insert(0, str(folder))
S = importlib.import_module('stage1_cache_v3')
F = importlib.import_module('343_fusion')
E = importlib.import_module('343_evaluator')
G = F.G
COUNTS = {'model_loads': 0, 'forwards': 0, 'optimizer_steps': 0, 'COCO_evaluations': 0,
          'image_decodes': 0, 'raw_cache_skips': 0, 'completed_unit_skips': 0}
STARTED = time.perf_counter()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def say(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


class HardStop(RuntimeError):
    pass


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def write_text(path, value):
    path = Path(path).resolve()
    check(path.is_relative_to(HERE), 'Output escaped new directory')
    with open(path, 'x', encoding='utf-8', newline='\n') as f:
        f.write(value)


def write(path, value):
    write_text(path, json_text(value))


def write_or_verify(path, value):
    if Path(path).exists():
        check(Path(path).read_text(encoding='utf-8') == json_text(value), 'Existing output differs: ' + str(path))
    else:
        write(path, value)


def sealed_json(path, value):
    write_or_verify(path, value)
    write_or_verify(str(path) + '.sha256.json', {'sha256': sha(path)})


def read_sealed(path):
    check(sha(path) == read(str(path) + '.sha256.json')['sha256'], 'Result SHA mismatch: ' + str(path))
    return read(path)


def local_sources():
    sources = {str(Path(__file__).resolve()): sha(__file__)}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name and str(name).endswith(('.py', '.pyd')):
            p = Path(name).resolve()
            if p.is_relative_to(ROOT) and p.is_file():
                sources[str(p)] = sha(p)
    return dict(sorted(sources.items()))


def state_sha(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def payload_sha(entry):
    h = hashlib.sha256()
    for key in ('final300_boxes', 'final300_logits'):
        t = entry[key]
        h.update(key.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.contiguous().numpy().tobytes())
    metadata = {k: v for k, v in entry.items() if k not in ('final300_boxes', 'final300_logits', 'payload_sha256')}
    h.update(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode())
    return h.hexdigest()


def verify_mainline():
    records = []
    for seed in (1, 2):
        for arm in ('G', 'N', 'R', 'S'):
            folder = MAINRUNS / f'{arm}_seed{seed}'
            vp, sp = folder / '348_run_verified.json', folder / '348_run_封存SHA256.json'
            check(vp.is_file() and sp.is_file(), 'Mainline unfinished/unsealed: ' + folder.name)
            v, seal = read(vp), read(sp)
            check(v['status'] == '已验证' and v['full_budget_completed'] is True
                  and v['step'] == 1218 and v['epochs'] == 6, 'Mainline completion: ' + folder.name)
            check(seal['status'] == '已封存', 'Mainline seal: ' + folder.name)
            rec = seal['artifacts'][str(vp)]
            check(sha(vp) == rec['sha256'] and vp.stat().st_size == rec['bytes'], 'Verification/seal mismatch')
            records.append({'run': folder.name, 'verification_path': str(vp), 'verification_sha256': sha(vp),
                            'seal_path': str(sp), 'seal_sha256': sha(sp), 'step': 1218, 'epochs': 6})
    return records


def prepare():
    mainline = verify_mainline()
    idx = read(INDEX)
    ar = next(a for a in idx['annotations'] if a['split'] == 'val')
    ap = Path(ar['path'])
    check(sha(ap) == ar['expected_SHA256_from_existing_audit'], 'Annotation SHA')
    dataset = read(ap)
    images = dataset['images']
    check(len(images) == 548 and len({im['id'] for im in images}) == 548, 'Fixed val548')
    by = collections.defaultdict(list)
    for annotation in dataset['annotations']:
        by[annotation['image_id']].append(annotation)
    checkpoints = {c['seed']: c for c in idx['checkpoints'] if c['seed'] in (1, 2)}
    for seed, cp in checkpoints.items():
        check(cp['state_key'] == 'model' and cp['expected_SHA256_from_288'] == CHECKPOINT_SHAS[seed], 'Raw identity')
        check(sha(cp['path']) == CHECKPOINT_SHAS[seed], 'Checkpoint SHA seed' + str(seed))
    reference = {x['seed']: x['raw_stats12'] for x in read(REFERENCE)['per_seed']}
    binding_path = HERE / '349_inputs.json'
    if binding_path.exists():
        binding = read_sealed(binding_path)
        for p, digest in {**binding['input_sha256'], **binding['source_sha256']}.items():
            check(sha(p) == digest, 'Input/source changed: ' + p)
        for rec in binding['cohort_files']:
            check(sha(rec['path']) == rec['sha256'], 'Cohort SHA')
        check(mainline == binding['mainline_completion'], 'Mainline verification changed')
        cohort = read(HERE / '349_cohort_metadata.json')
        say({'event': 'prepared_sha_verified', 'images': 548, 'T': len(cohort['T']), 'S': len(cohort['S'])})
        return binding, cohort, dataset, by, checkpoints, reference, ap.parent.parent
    inputs = {str(p): sha(p) for p in (PLAN, INDEX, REFERENCE, ap)}
    for rec in mainline:
        inputs[rec['verification_path']] = rec['verification_sha256']
        inputs[rec['seal_path']] = rec['seal_sha256']
    for cp in checkpoints.values():
        p = Path(cp['path'])
        inputs[str(p)] = sha(p)
        bundle = p.parent.parent / 'bundle/resolved.yml'
        inputs[str(bundle)] = sha(bundle)
    T, small = [], []
    bin_ids = {'[0,2)': [], '[2,3)': [], '[3,4)': []}
    unbinned, widths, gt_count = [], {}, 0
    for pos, im in enumerate(images, 1):
        iid = im['id']
        p = STAGE3 / '342_cache_val_v2_seed1' / f'342_{iid:06d}.pt'
        inputs[str(p)] = sha(p)
        d = torch.load(p, map_location='cpu', weights_only=True)
        ids = [a['id'] for a in by[iid]]
        check(d['gt_annotation_ids'] == ids and d['orig_wh'] == [im['width'], im['height']], 'Cohort alignment')
        check(d['annotation_sha256'] == inputs[str(ap)], 'Cohort annotation SHA')
        short = d['gt_short_side_640_all'].tolist()
        width = (d['gt_boxes_640'][:, 2] - d['gt_boxes_640'][:, 0]).tolist()
        check(len(short) == len(width) == len(ids), 'Cohort lengths')
        for gid, ss, w in zip(ids, short, width):
            gt_count += 1
            if ss < 4:
                T.append(gid)
                widths[str(gid)] = w
                placed = False
                for lo, hi, label in ((0, 2, '[0,2)'), (2, 3, '[2,3)'), (3, 4, '[3,4)')):
                    if lo <= w < hi:
                        bin_ids[label].append(gid)
                        placed = True
                        break
                if not placed:
                    unbinned.append(gid)
            elif 4 <= ss < 16:
                small.append(gid)
        src = ap.parent.parent / 'images' / im['file_name']
        digest = sha(src)
        check(digest == d['source_image_sha256'], 'Image identity')
        inputs[str(src)] = digest
        for seed in (1, 2):
            p = STAGE3 / f'342_raw_val_seed{seed}' / f'342_{iid:06d}.pt'
            inputs[str(p)] = sha(p)
        if pos % 128 == 0:
            say({'event': 'input_hashing', 'images': pos, 'of': 548})
    check(len(T) == len(set(T)) == 4952, 'Fixed T must contain 4952 GT')
    check(len(small) == len(set(small)) and not (set(T) & set(small)), 'Disjoint cohorts')
    cohort = {'T': sorted(T), 'S': sorted(small), 'T_width_bins': {k: sorted(v) for k, v in bin_ids.items()},
              'T_outside_requested_width_bins': sorted(unbinned), 'T_width_640': widths, 'all_annotation_count': gt_count,
              'definition': 'T: seed1 gt_short_side_640_all<4; S:4<=short<16. Bins use cached GT WIDTH in 640 stretch plane.'}
    write_or_verify(HERE / '349_cohort_T_ids.json', cohort['T'])
    write_or_verify(HERE / '349_cohort_S_ids.json', cohort['S'])
    write_or_verify(HERE / '349_cohort_metadata.json', cohort)
    cohort_files = [{'path': str(HERE / name), 'sha256': sha(HERE / name)} for name in
                    ('349_cohort_T_ids.json', '349_cohort_S_ids.json', '349_cohort_metadata.json')]
    write_or_verify(HERE / '349_cohort_SHA256.json', cohort_files)
    binding = {'created_utc': now(), 'mainline_completion': mainline, 'input_sha256': inputs,
               'source_sha256': local_sources(), 'cohort_files': cohort_files, 'shapes_WH': SHAPES,
               'state_key': 'model', 'preprocessing': 'RGB; TF.resize([H,W]) PIL bilinear; TF.to_tensor; no pad',
               'postprocessing': 'Direct 343_fusion.fuse and coco_rows; no added predictions',
               'evaluator': E.evidence(), 'thread_settings': {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2',
                   'MKL_NUM_THREADS': '2', 'MKL_THREADING_LAYER': 'SEQUENTIAL', 'torch_num_threads': 2},
               'environment': {'python': sys.executable, 'python_version': sys.version, 'torch': torch.__version__,
                               'numpy': np.__version__, 'pillow': PIL.__version__}}
    sealed_json(binding_path, binding)
    say({'event': 'prepared', 'images': 548, 'T': len(T), 'S': len(small),
         'T_width_bins': {k: len(v) for k, v in bin_ids.items()}, 'T_outside_width_bins': len(unbinned)})
    return binding, cohort, dataset, by, checkpoints, reference, ap.parent.parent


def completion_path(seed, arm):
    return HERE / f'349_completed_seed{seed}_{arm}.json'


def verified_completion(seed, arm, binding_sha):
    report = read_sealed(completion_path(seed, arm))
    check(report['binding_sha256'] == binding_sha and report['seed'] == seed and report['arm'] == arm, 'Unit binding')
    for p, digest in {**report['artifact_sha256'], **report['source_sha256']}.items():
        check(sha(p) == digest, 'Completed file SHA mismatch: ' + p)
    check(report['images'] == 548 and report['executed_counts']['optimizer_steps'] == 0, 'Completed budget')
    return report


def require_A0(binding_sha):
    for seed in (1, 2):
        check(completion_path(seed, 'A0').is_file(), 'Both A0 seeds must finish before other arms')
        r = verified_completion(seed, 'A0', binding_sha)
        if not r['A0_consistency']['passed']:
            raise HardStop('A0 failed: seed' + str(seed))


def raw_entry(seed, arm, im, model, binding, binding_sha, data_root):
    iid = im['id']
    folder = HERE / f'349_raw_seed{seed}_{arm}'
    folder.mkdir(exist_ok=True)
    path = folder / f'349_{iid:06d}.pt'
    src = data_root / 'images' / im['file_name']
    ih = binding['input_sha256'][str(src)]
    check(sha(src) == ih, 'Image modified after preflight')
    if path.exists():
        entry = torch.load(path, map_location='cpu', weights_only=True)
        check(payload_sha(entry) == entry['payload_sha256'], 'Raw cache payload SHA')
        COUNTS['raw_cache_skips'] += 1
    else:
        check(model is not None, 'Missing model')
        with Image.open(src) as f:
            pil = f.convert('RGB')
        COUNTS['image_decodes'] += 1
        check(pil.size == (im['width'], im['height']), 'Decoded original image dimensions')
        w, h = SHAPES[arm]
        x = TF.to_tensor(TF.resize(pil, [h, w])).unsqueeze(0)
        check(x.shape == (1, 3, h, w) and x.dtype == torch.float32 and x.device.type == 'cpu', 'Input CPU FP32')
        with torch.no_grad():
            t0 = time.perf_counter()
            outputs = model(x)
            elapsed = time.perf_counter() - t0
        COUNTS['forwards'] += 1
        boxes = outputs['pred_boxes'][0].detach().contiguous()
        logits = outputs['pred_logits'][0].detach().contiguous()
        check(boxes.shape == (300, 4) and logits.shape == (300, 10), 'Final query shapes')
        check(bool(torch.isfinite(boxes).all() and torch.isfinite(logits).all()), 'Finite output')
        entry = {'seed': seed, 'arm': arm, 'image_id': iid, 'shape_WH': list(SHAPES[arm]),
                 'orig_wh': [im['width'], im['height']], 'source_image_sha256': ih,
                 'checkpoint_sha256': CHECKPOINT_SHAS[seed], 'binding_sha256': binding_sha,
                 'forward_seconds': elapsed, 'final300_boxes': boxes, 'final300_logits': logits}
        entry['payload_sha256'] = payload_sha(entry)
        temporary = folder / f'349_incomplete_{uuid.uuid4().hex}.pt'
        with open(temporary, 'xb') as f:
            torch.save(entry, f)
        os.rename(temporary, path)
    check(entry['seed'] == seed and entry['arm'] == arm and entry['image_id'] == iid
          and entry['shape_WH'] == list(SHAPES[arm]) and entry['checkpoint_sha256'] == CHECKPOINT_SHAS[seed]
          and entry['source_image_sha256'] == ih and entry['binding_sha256'] == binding_sha, 'Raw entry binding')
    return entry, path


def run_unit(seed, arm, prepared):
    binding, cohort, dataset, by, checkpoints, reference, data_root = prepared
    binding_sha = sha(HERE / '349_inputs.json')
    if arm != 'A0':
        require_A0(binding_sha)
    if completion_path(seed, arm).exists():
        completed = verified_completion(seed, arm, binding_sha)
        COUNTS['completed_unit_skips'] += 1
        say({'event': 'completed_unit_sha_verified_skip', 'seed': seed, 'arm': arm})
        return completed
    start = time.perf_counter()
    before_counts = dict(COUNTS)
    selected_ids = set(cohort['T']) | set(cohort['S'])
    geo0, artifacts, per_image, rows, comparisons = set(), {}, [], [], []
    model, state_before = None, None
    if not all((HERE / f'349_raw_seed{seed}_{arm}' / f'349_{im["id"]:06d}.pt').is_file() for im in dataset['images']):
        model = S.load_real_model(checkpoints[seed]['path'])
        COUNTS['model_loads'] += 1
        check(model.geometry_arm == 'baseline' and not model.training, 'Baseline eval mode')
        check(model.encoder.eval_spatial_size == [640, 640] and model.decoder.eval_spatial_size == [640, 640], 'Initial eval size')
        model.encoder.eval_spatial_size = None
        model.decoder.eval_spatial_size = None
        check(all(p.device.type == 'cpu' and not p.requires_grad for p in model.parameters()), 'Frozen CPU parameters')
        state_before = state_sha(model)
        load_receipt = HERE / f'349_model_load_seed{seed}_{arm}_{uuid.uuid4().hex}.json'
        write(load_receipt, {'seed': seed, 'arm': arm, 'utc': now(), 'binding_sha256': binding_sha, 'state_sha': state_before})
        say({'event': 'raw_strict_model_loaded_dynamic', 'seed': seed, 'arm': arm, 'shape_WH': SHAPES[arm]})
    for position, im in enumerate(dataset['images'], 1):
        entry, raw_path = raw_entry(seed, arm, im, model, binding, binding_sha, data_root)
        artifacts[str(raw_path)] = sha(raw_path)
        pb, logits = entry['final300_boxes'], entry['final300_logits']
        if arm == 'A0':
            oldpath = STAGE3 / f'342_raw_val_seed{seed}' / f'342_{im["id"]:06d}.pt'
            check(sha(oldpath) == binding['input_sha256'][str(oldpath)], 'A0 reference SHA')
            prior = torch.load(oldpath, map_location='cpu', weights_only=True)
            check(prior['checkpoint_sha256'] == CHECKPOINT_SHAS[seed] and
                  prior['source_image_sha256'] == entry['source_image_sha256'], 'A0 reference identity')
            diff = {k: float((entry[k] - prior[k]).abs().max()) for k in ('final300_boxes', 'final300_logits')}
            comparisons.append({'image_id': im['id'], **diff})
            if max(diff.values()) > 1e-4:
                write(HERE / f'349_A0_seed{seed}_hard_stop.json', {'seed': seed, 'reason': 'raw_difference',
                      'threshold': 1e-4, 'per_image_differences_so_far': comparisons, 'counts': dict(COUNTS)})
                raise HardStop('A0 raw failed: ' + json.dumps(comparisons[-1]))
        out, _ = F.fuse({'final300_boxes': pb, 'final300_logits': logits,
                         'orig_wh': entry['orig_wh'], 'window_xyxy_640': torch.tensor([0., 0., 64., 64.])})
        rows.extend(F.coco_rows(im['id'], out))
        items = [a for a in by[im['id']] if a['id'] in selected_ids]
        if items:
            xywh = torch.tensor([a['bbox'] for a in items], dtype=torch.float64)
            gt = torch.cat((xywh[:, :2], xywh[:, :2] + xywh[:, 2:]), -1)
            queries = G.global_queries_to_orig_xyxy(pb.double(), entry['orig_wh'])
            maximum = G.pairwise_iou(queries, gt, allow_raw_a=True).max(0).values
            geo0.update(a['id'] for a, mx in zip(items, maximum.tolist()) if mx < .5)
        per_image.append({'image_id': im['id'], 'forward_seconds': entry['forward_seconds'], 'detections': len(out['scores'])})
        if position % 32 == 0 or position == 548:
            say({'event': 'inference_progress', 'seed': seed, 'arm': arm, 'images': position, 'of': 548,
                 'wall_seconds': round(time.perf_counter() - start, 2), 'forwards_total': COUNTS['forwards']})
    state_after = state_sha(model) if model is not None else None
    if model is not None:
        check(state_before == state_after and all(p.grad is None for p in model.parameters()), 'State/gradient invariant')
        check(model.encoder.eval_spatial_size is None and model.decoder.eval_spatial_size is None, 'Dynamic mode retained')
    for lp in HERE.glob(f'349_model_load_seed{seed}_{arm}_*.json'):
        artifacts[str(lp)] = sha(lp)
    prediction_path = HERE / f'349_predictions_seed{seed}_{arm}.json'
    write_or_verify(prediction_path, rows)
    artifacts[str(prediction_path)] = sha(prediction_path)
    ev, metrics, log, evaluation_seconds = E.evaluate(dataset, rows)
    COUNTS['COCO_evaluations'] += 1
    logs = list(HERE.glob(f'349_COCO_seed{seed}_{arm}_*.txt'))
    logpath = HERE / f'349_COCO_seed{seed}_{arm}_{len(logs):03d}.txt'
    write_text(logpath, log)
    artifacts[str(logpath)] = sha(logpath)
    matches = {}
    all_range = ev.params.areaRng[ev.params.areaRngLbl.index('all')]
    ti = int(np.flatnonzero(np.isclose(ev.params.iouThrs, .5, atol=1e-12, rtol=0))[0])
    for item in ev.evalImgs:
        if item is None or item['aRng'] != all_range or item['maxDet'] != 100:
            continue
        for gid, did in zip(item['gtIds'], item['gtMatches'][ti]):
            if did != 0:
                gid, did = int(gid), int(did)
                check(gid not in matches, 'Multiple category matches for one GT')
                matches[gid] = {'detection_id': did, 'width_orig': float(ev.cocoDt.anns[did]['bbox'][2])}
    matched = set(matches)
    groups = {}
    for name, ids in {'T': cohort['T'], 'S': cohort['S'], **cohort['T_width_bins']}.items():
        selected = set(ids)
        groups[name] = {'n': len(ids), 'Geo0': len(selected & geo0), 'TP_at_0p5': len(selected & matched)}
    tiny_matched = sorted(set(cohort['T']) & matched)
    matched_widths = [matches[g]['width_orig'] for g in tiny_matched]
    consistency = None
    if arm == 'A0':
        metric_diffs = {k: abs(metrics[k] - reference[seed][k]) for k in E.NAMES}
        consistency = {'raw_max_abs_difference': {k: max(x[k] for x in comparisons) for k in ('final300_boxes', 'final300_logits')},
                       'raw_threshold': 1e-4, 'raw_images_compared': len(comparisons), 'per_image_raw_differences': comparisons,
                       'reference_raw_stats12': reference[seed], 'absolute_stats12_differences': metric_diffs,
                       'max_abs_stats12_difference': max(metric_diffs.values()), 'stats12_threshold': 1e-6}
        consistency['passed'] = (max(consistency['raw_max_abs_difference'].values()) <= 1e-4
                                  and consistency['max_abs_stats12_difference'] <= 1e-6)
    report = {'status': 'completed', 'seed': seed, 'arm': arm, 'shape_WH': list(SHAPES[arm]), 'images': 548,
              'binding_sha256': binding_sha, 'checkpoint_sha256': CHECKPOINT_SHAS[seed], 'stats12': metrics,
              'cohorts': groups, 'T_matched_ids': tiny_matched, 'S_matched_ids': sorted(set(cohort['S']) & matched),
              'Geo0_ids_T_and_S': sorted(geo0), 'T_match_records': {str(g): matches[g] for g in tiny_matched},
              'T_matched_prediction_width_orig_median': float(np.median(matched_widths)) if matched_widths else None,
              'T_outside_requested_width_bins': {'n': len(cohort['T_outside_requested_width_bins']),
                    'Geo0': len(set(cohort['T_outside_requested_width_bins']) & geo0),
                    'TP_at_0p5': len(set(cohort['T_outside_requested_width_bins']) & matched)},
              'CPU_forward_seconds': sum(x['forward_seconds'] for x in per_image),
              'COCO_evaluation_seconds': evaluation_seconds, 'unit_wall_seconds': time.perf_counter() - start,
              'per_image': per_image, 'A0_consistency': consistency,
              'executed_counts': {'model_loads': len(list(HERE.glob(f'349_model_load_seed{seed}_{arm}_*.json'))),
                                  'forwards': len(per_image), 'COCO_evaluations': len(logs)+1, 'optimizer_steps': 0},
              'counts_current_invocation': {k: COUNTS[k] - before_counts[k] for k in COUNTS},
              'state_sha_before': state_before, 'state_sha_after': state_after, 'all_grad_none': True,
              'artifact_sha256': artifacts, 'source_sha256': local_sources()}
    sealed_json(completion_path(seed, arm), report)
    say({'event': 'unit_completed', 'seed': seed, 'arm': arm, 'AP': metrics['AP'], 'APs': metrics['APs'],
         'T_TP': groups['T']['TP_at_0p5'], 'T_Geo0': groups['T']['Geo0'], 'S_TP': groups['S']['TP_at_0p5'],
         'A0_consistency_passed': consistency['passed'] if consistency else None})
    del model, rows, ev
    gc.collect()
    if consistency is not None and not consistency['passed']:
        raise HardStop('A0 consistency failed seed' + str(seed) + ': ' + str(consistency['max_abs_stats12_difference']))
    return report


def summarize(prepared):
    binding, cohort, *_ = prepared
    binding_sha = sha(HERE / '349_inputs.json')
    reports = [verified_completion(seed, arm, binding_sha) for seed in (1, 2) for arm in SHAPES]
    checks, effects = [], []
    for seed in (1, 2):
        r = {x['arm']: x for x in reports if x['seed'] == seed}
        a0, a1, a2 = r['A0'], r['A1'], r['A2']
        t0, t1, t2 = [x['cohorts']['T']['TP_at_0p5'] for x in (a0, a1, a2)]
        s0, s1 = [x['cohorts']['S']['TP_at_0p5'] for x in (a0, a1)]
        c = {'seed': seed,
             'T_TP_strict_increase': {'A0': t0, 'A1': t1, 'delta': t1-t0, 'satisfied': t1 > t0},
             'APs_strict_increase': {'A0': a0['stats12']['APs'], 'A1': a1['stats12']['APs'],
                 'delta': a1['stats12']['APs']-a0['stats12']['APs'], 'satisfied': a1['stats12']['APs'] > a0['stats12']['APs']},
             'S_TP_drop_at_most_1_percent': {'A0': s0, 'A1': s1, 'minimum_allowed': .99*s0,
                 'delta': s1-s0, 'relative_change': (s1-s0)/s0 if s0 else None, 'satisfied': s1 >= .99*s0},
             'mechanism_A1_gt_A0_gt_A2': {'A1': t1, 'A0': t0, 'A2': t2, 'satisfied': t1 > t0 > t2}}
        c['main_pass'] = all(c[k]['satisfied'] for k in
                             ('T_TP_strict_increase', 'APs_strict_increase', 'S_TP_drop_at_most_1_percent'))
        checks.append(c)
        ids0, ids1 = set(a0['T_matched_ids']), set(a1['T_matched_ids'])
        effects.append({'seed': seed, 'new_T_matches_A1_over_A0': len(ids1-ids0), 'lost_T_matches_A1_vs_A0': len(ids0-ids1),
                        'new_T_ids': sorted(ids1-ids0), 'lost_T_ids': sorted(ids0-ids1)})
        check(len(ids1-ids0)-len(ids0-ids1) == t1-t0, 'Gain/loss reconciliation')
    passed = all(x['main_pass'] for x in checks)
    mechanism = all(x['mechanism_A1_gt_A0_gt_A2']['satisfied'] for x in checks)
    counts = {k: sum(r['executed_counts'][k] for r in reports) for k in reports[0]['executed_counts']}
    check(counts['forwards'] == 4384 and counts['optimizer_steps'] == 0, 'Exact forward budget')
    result_path = HERE / '349_result.json'
    if result_path.exists():
        result = read_sealed(result_path)
        check(result['status'] == 'completed' and result['binding_file_sha256'] == binding_sha, 'Final result binding')
    else:
        result = {'status': 'completed', 'completed_utc': now(), 'fixed_plan_path': str(PLAN),
                  'conclusion': '值得申请重训检验' if passed else 'mixed/未通过',
                  'mechanism_conclusion': '方向与 x 压缩假设一致' if mechanism else '增益来源未识别',
                  'main_comparison': 'A1 vs A0; both seeds must pass all three criteria', 'A3_descriptive_only': True,
                  'criteria_per_seed': checks, 'A1_vs_A0_T_gain_loss': effects,
                  'A0_consistency': {str(r['seed']): r['A0_consistency'] for r in reports if r['arm'] == 'A0'},
                  'per_seed_arm': reports, 'cohort_n': {'T': len(cohort['T']), 'S': len(cohort['S'])},
                  'cohort_SHA256': binding['cohort_files'], 'binding_file_sha256': binding_sha,
                  'all_input_sha256': binding['input_sha256'], 'all_source_sha256': local_sources(),
                  'executed_counts': counts, 'current_invocation_counts': dict(COUNTS),
                  'actual_CPU_forward_seconds': sum(r['CPU_forward_seconds'] for r in reports),
                  'actual_COCO_evaluation_seconds': sum(r['COCO_evaluation_seconds'] for r in reports),
                  'actual_unit_wall_seconds': sum(r['unit_wall_seconds'] for r in reports),
                  'current_invocation_wall_seconds': time.perf_counter() - STARTED,
                  'interpretation_limit': '权重在640拉伸图上训练。通过只支持申请重训检验；未通过不能否定保持宽高比重训方向。',
                  'stop_point': '已停止；等主控核对；未生成训练命令或启动重训。',
                  'TP_definition': 'evalImgs: areaRng=all, maxDet=100, IoU=0.5, gtMatches nonzero intersect fixed IDs; no added filters.'}
        sealed_json(result_path, result)
    columns = ['seed', '臂', 'W×H'] + E.NAMES + ['T_Geo0', 'T_TP@0.5', 'S_Geo0', 'S_TP@0.5',
               '宽[0,2) Geo0/TP', '宽[2,3) Geo0/TP', '宽[3,4) Geo0/TP', 'T匹配框宽中位数(原图px)', 'CPU前向秒']
    lines = ['# 349 零训练预检', '', '| ' + ' | '.join(columns) + ' |',
             '| ' + ' | '.join(['---']*len(columns)) + ' |']
    for r in reports:
        g = r['cohorts']
        row = [str(r['seed']), r['arm'], '×'.join(map(str, r['shape_WH']))]
        row += [f'{r["stats12"][k]:.12f}' for k in E.NAMES]
        row += [str(g[name][k]) for name in ('T', 'S') for k in ('Geo0', 'TP_at_0p5')]
        row += [f'{g[b]["Geo0"]}/{g[b]["TP_at_0p5"]}' for b in ('[0,2)', '[2,3)', '[3,4)')]
        row += [str(r['T_matched_prediction_width_orig_median']), f'{r["CPU_forward_seconds"]:.6f}']
        lines.append('| ' + ' | '.join(row) + ' |')
    lines += ['', f'T={len(cohort["T"])}；S={len(cohort["S"])}。',
              'T按GT宽度分段的数量：' + '；'.join(f'{k}={len(v)}' for k, v in cohort['T_width_bins'].items()) + '。',
              f'T中GT宽度≥4、未落入上述三段的数量={len(cohort["T_outside_requested_width_bins"])}；T总计仍包含这些GT。', '']
    for c, effect in zip(checks, effects):
        lines += [f'seed{c["seed"]}：', '']
        for k in ('T_TP_strict_increase', 'APs_strict_increase', 'S_TP_drop_at_most_1_percent', 'mechanism_A1_gt_A0_gt_A2'):
            lines.append('- ' + k + '：' + json.dumps(c[k], ensure_ascii=False))
        lines += [f'- T新增={effect["new_T_matches_A1_over_A0"]}；丢失={effect["lost_T_matches_A1_vs_A0"]}。', '']
    for r in reports:
        if r['arm'] == 'A0':
            a = r['A0_consistency']
            lines.append(f'A0 seed{r["seed"]}：boxes最大差={a["raw_max_abs_difference"]["final300_boxes"]}；logits最大差={a["raw_max_abs_difference"]["final300_logits"]}；12项指标最大差={a["max_abs_stats12_difference"]}；通过={a["passed"]}。')
    lines += ['', '判读结论：' + result['conclusion'] + '。', '机制检查：' + result['mechanism_conclusion'] + '。',
              f'模型加载={counts["model_loads"]}；前向={counts["forwards"]}；optimizer_steps=0。',
              f'CPU前向总秒数={result["actual_CPU_forward_seconds"]:.6f}；COCO评估总秒数={result["actual_COCO_evaluation_seconds"]:.6f}。',
              'A3仅描述，不参与判读。', result['interpretation_limit'], result['stop_point']]
    md = '\n'.join(lines) + '\n'
    mdpath = HERE / '349_中文汇总.md'
    if mdpath.exists():
        check(mdpath.read_text(encoding='utf-8') == md, 'Summary differs')
    else:
        write_text(mdpath, md)
    sealpath = HERE / '349_delivery_SHA256.json'
    if not sealpath.exists():
        output_files = {str(p): sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()
                        and 'incomplete_' not in p.name and not p.name.startswith('349_delivery_SHA256')}
        sealed_json(sealpath, {'status': 'sealed', 'files': output_files})
    else:
        for p, digest in read_sealed(sealpath)['files'].items():
            check(sha(p) == digest, 'Delivery seal SHA mismatch: ' + p)
    say({'event': 'all_complete', 'conclusion': result['conclusion'],
         'mechanism': result['mechanism_conclusion'], 'counts': counts})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['all', *SHAPES], default='all')
    parser.add_argument('--seed', type=int, choices=[1, 2])
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    prepared = prepare()
    if args.prepare:
        return
    if args.summarize:
        summarize(prepared)
        return
    seeds = [args.seed] if args.seed else [1, 2]
    arms = list(SHAPES) if args.arm == 'all' else [args.arm]
    for arm in arms:
        for seed in seeds:
            run_unit(seed, arm, prepared)
    if all(completion_path(seed, arm).exists() for seed in (1, 2) for arm in SHAPES):
        summarize(prepared)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        record = {'status': 'hard_stop' if isinstance(error, HardStop) else 'implementation_error', 'utc': now(),
                  'error': repr(error), 'traceback': traceback.format_exc(), 'executed_counts': dict(COUNTS),
                  'elapsed_seconds': time.perf_counter()-STARTED}
        write(HERE / f'349_failure_{uuid.uuid4().hex}.json', record)
        if isinstance(error, HardStop) and not (HERE / '349_result.json').exists():
            sealed_json(HERE / '349_result.json', record)
            write_text(HERE / '349_中文汇总.md', '# 349 硬停止\n\n' + record['error'] + '\n\n' + json_text(COUNTS))
        raise
