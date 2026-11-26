"""A0 finite frozen-model diagnostic. No training, input changes or query changes."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import importlib
import json
import os
import time
import types
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.v2 import functional as VF
from tinyformer_probe_loader import ROOT, PREFIX, load_official_model, state_digest

DATA = ROOT.parent / 'dataset'
RECEIPTS = ROOT / 'receipts'


def save_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


class Budget:
    def __init__(self):
        self.path = RECEIPTS / 'cumulative_forward_budget.json'
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {'attempted_full_model_forwards': 0, 'cap': 200, 'events': []}

    def forward(self, model, x, phase, timed=False):
        assert self.data['attempted_full_model_forwards'] < 200
        self.data['attempted_full_model_forwards'] += 1
        self.data['events'].append({'index': self.data['attempted_full_model_forwards'], 'phase': phase})
        save_json(self.path, self.data)
        if timed:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            out = model(x)
            end.record()
            end.synchronize()
            self.last_model_ms = begin.elapsed_time(end)
        else:
            out = model(x)
        assert all(torch.isfinite(out[k]).all() for k in ['pred_boxes', 'pred_logits'])
        assert out['pred_boxes'].shape == (1, 300, 4) and out['pred_logits'].shape == (1, 300, 10)
        return out


def inputs():
    manifest = json.loads((ROOT / 'diagnostic100_asset_manifest.json').read_text())
    paths = [Path(x) for x in (DATA / 'diagnostic_images.txt').read_text().splitlines() if x]
    expected = {r['file_name']: r for r in manifest['files']}
    assert len(paths) == 100 and {p.name for p in paths} == set(expected)
    rows = []
    for p in paths:
        assert p.parent.resolve() == (DATA / 'train/images').resolve()
        ann = DATA / 'train/annotations' / (p.stem + '.txt')
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected[p.name]['images']['sha256']
        assert hashlib.sha256(ann.read_bytes()).hexdigest() == expected[p.name]['annotations']['sha256']
        with Image.open(p) as im:
            width, height = im.size
        boxes, labels, uids = [], [], []
        for line_no, line in enumerate(ann.read_text(encoding='utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            f = [float(x) for x in line.rstrip(',').split(',')]
            x, y, w, h, score, cat = f[:6]
            x1, y1 = np.clip(x, 0, width), np.clip(y, 0, height)
            x2, y2 = np.clip(x + w, 0, width), np.clip(y + h, 0, height)
            if score == 1 and 1 <= int(cat) <= 10 and x2 > x1 and y2 > y1:
                boxes.append([x1 / width * 640, y1 / height * 640, x2 / width * 640, y2 / height * 640])
                labels.append(int(cat) - 1)
                uids.append(p.name + ':' + str(line_no))
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        small = areas < 1024
        centers = np.floor((boxes[:, :2] + boxes[:, 2:]) * .5 / 16).astype(int)
        cells = Counter(tuple(c) for c in centers[small])
        crowd = np.array([bool(s and cells[tuple(c)] >= 2) for s, c in zip(small, centers)])
        rows.append({'path': p, 'gt_boxes': boxes, 'gt_labels': np.asarray(labels),
                     'gt_uids': np.asarray(uids), 'gt_sqrt_area': np.sqrt(areas), 'gt_crowd': crowd})
    assert sum(len(r['gt_labels']) for r in rows) == manifest['expected_positive_gt'] == 4491
    save_json(RECEIPTS / 'frozen_input_contract.json', {
        'file_names_in_existing_order': [p.name for p in paths],
        'existing_image_list_sha256': hashlib.sha256((DATA / 'diagnostic_images.txt').read_bytes()).hexdigest(),
        'asset_manifest_sha256': hashlib.sha256((ROOT / 'diagnostic100_asset_manifest.json').read_bytes()).hexdigest(),
        'images_and_annotations_sha256_verified': 100, 'positive_gt': 4491,
        'first8_equivalence_images': [p.name for p in paths[:8]],
        'class_names': manifest['classes'], 'source_scope': manifest['scope']})
    return rows


def image_tensor(path):
    with Image.open(path) as im:
        im = VF.resize(im.convert('RGB'), [640, 640], antialias=True)
        x = VF.pil_to_tensor(im).float() / 255.
        x = VF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return x.unsqueeze(0).cuda()


def trace_forward(model, pp, x, budget, phase):
    d, captured, handles = model.decoder, {}, []
    def score_hook(module, args, output):
        captured['memory'], captured['encoder_logits'] = args[0].detach(), output.detach()
    def inner_pre(module, args):
        captured['topk_unact'] = args[1].detach()
    def inner_after(module, args, output):
        captured['inner_out'] = tuple(v.detach() for v in output)
    def make_hook(key):
        def hook(module, args, output):
            captured[key] = output.detach()
        return hook
    handles.append(d.enc_score_head.register_forward_hook(score_hook))
    handles.append(d.decoder.register_forward_pre_hook(inner_pre))
    handles.append(d.decoder.register_forward_hook(inner_after))
    for i in range(4):
        handles.append(d.decoder.layers[i].register_forward_hook(make_hook('layer_' + str(i))))
        handles.append(d.dec_bbox_head[i].register_forward_hook(make_hook('corner_delta_' + str(i))))
    native_select = d._select_topk
    def select_wrapper(self, memory, outputs_logits, anchors, topk):
        result = native_select(memory, outputs_logits, anchors, topk)
        assert self.query_select_method == 'default'
        ids = torch.topk(outputs_logits.max(-1).values, topk, dim=-1).indices
        assert torch.equal(result[0], memory.gather(1, ids.unsqueeze(-1).expand(-1, -1, memory.shape[-1])))
        captured['topk_ids'] = ids.detach()
        return result
    d._select_topk = types.MethodType(select_wrapper, d)
    try:
        out = budget.forward(model, x, phase)
    finally:
        d._select_topk = native_select
        for handle in handles:
            handle.remove()
    # All extra head readouts happen after the untouched native forward finished.
    ppout = pp(out, torch.tensor([[640, 640]], device=x.device))[0]
    flat_ids = torch.topk(out['pred_logits'].sigmoid().flatten(1), 300, dim=-1).indices[0]
    assert torch.equal(ppout['labels'], flat_ids % 10)
    valid = d.valid_mask[0, :, 0]
    assert valid[captured['topk_ids'][0]].all()
    memory, logits = captured['memory'], captured['encoder_logits']
    all_boxes = (d.enc_bbox_head(memory) + d.anchors).sigmoid()
    topids = captured['topk_ids'][0]
    preboxes, prescores = captured['inner_out'][-2:]
    stages = {
        'encoder_all_valid': (all_boxes[0, valid], logits[0, valid]),
        'encoder_topk300': (captured['topk_unact'].sigmoid()[0], logits[0, topids]),
        'decoder_pre': (preboxes[0], prescores[0])}
    mod = importlib.import_module(PREFIX + '.deim.dfine_utils')
    project = mod.weighting_function(d.reg_max, d.up, d.reg_scale)
    corners = None
    for i in range(4):
        delta = captured['corner_delta_' + str(i)]
        corners = delta if corners is None else delta + corners
        boxes = mod.distance2bbox(preboxes, d.integral(corners, project), d.reg_scale)
        scores = d.dec_score_head[i](captured['layer_' + str(i)])
        scores = d.decoder.lqe_layers[i](scores, corners)
        stages['decoder_' + str(i)] = (boxes[0], scores[0])
    assert (stages['decoder_3'][0] - out['pred_boxes'][0]).abs().max().item() <= 1e-6
    assert (stages['decoder_3'][1].sigmoid() - out['pred_logits'][0].sigmoid()).abs().max().item() <= 1e-6
    arrays = {}
    for name, (boxes, scores) in stages.items():
        assert torch.isfinite(boxes).all() and torch.isfinite(scores).all(), name
        arrays[name + '_boxes'] = boxes.cpu().numpy()
        arrays[name + '_logits'] = scores.cpu().numpy()
    arrays.update(encoder_token_ids=torch.where(valid)[0].cpu().numpy(), topk_token_ids=topids.cpu().numpy(),
                  native_pp_boxes=ppout['boxes'].cpu().numpy(), native_pp_scores=ppout['scores'].cpu().numpy(),
                  native_pp_labels=ppout['labels'].cpu().numpy(), native_pp_query_ids=(flat_ids // 10).cpu().numpy())
    return out, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['preflight', 'study'])
    args = parser.parse_args()
    frozen = json.loads((ROOT / 'execution_freeze_manifest.json').read_text())
    for item in frozen['files']:
        assert hashlib.sha256((ROOT / item['remote_name']).read_bytes()).hexdigest() == item['sha256']
    lock = RECEIPTS / (args.mode + '_started.json')
    with lock.open('x') as f:
        json.dump({'pid': os.getpid(), 'started_unix': time.time(), 'mode': args.mode}, f)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    np.random.seed(0)
    rows = inputs()
    model, pp, loaded = load_official_model()
    model, pp = model.cuda(), pp.cuda()
    digest_before = state_digest(model)
    budget = Budget()
    started = time.monotonic()
    with torch.inference_mode():
        if args.mode == 'preflight':
            torch.cuda.reset_peak_memory_stats()
            capacity = budget.forward(model, torch.zeros(1, 3, 640, 640, device='cuda'), 'capacity_preflight')
            del capacity
            capacity_peak = torch.cuda.max_memory_allocated()
            comparisons = []
            for row in rows[:8]:
                x = image_tensor(row['path'])
                native = budget.forward(model, x, 'equivalence_native')
                traced, arrays = trace_forward(model, pp, x, budget, 'equivalence_traced')
                bd = (native['pred_boxes'] - traced['pred_boxes']).abs().max().item()
                sd = (native['pred_logits'].sigmoid() - traced['pred_logits'].sigmoid()).abs().max().item()
                ni = torch.topk(native['pred_logits'].sigmoid().flatten(1), 300, dim=-1).indices
                ti = torch.topk(traced['pred_logits'].sigmoid().flatten(1), 300, dim=-1).indices
                assert bd <= 1e-6 and sd <= 1e-6 and torch.equal(ni, ti)
                comparisons.append({'file_name': row['path'].name, 'boxes_max_abs': bd,
                                    'sigmoid_scores_max_abs': sd, 'native_label_and_query_ids_exact': True})
            for _ in range(4):
                budget.forward(model, x, 'benchmark_warmup')
            times = []
            torch.cuda.reset_peak_memory_stats()
            for _ in range(8):
                budget.forward(model, x, 'benchmark_timed', timed=True)
                times.append(budget.last_model_ms)
            assert state_digest(model) == digest_before
            result = {'status': 'PREFLIGHT_EQUIVALENCE_PASS', 'comparisons': comparisons,
                      'capacity_peak_allocated_bytes': capacity_peak,
                      'benchmark_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'benchmark_model_only_cuda_event_ms': times,
                      'benchmark_scope': 'Unfused FP32 native eval, batch1, fixed single image, 4 warmups and 8 CUDA-event timings; no IO/postprocess/instrumentation',
                      'model_state_unchanged': True, 'state_digest': digest_before,
                      'full_model_forwards_cumulative': budget.data['attempted_full_model_forwards'],
                      'elapsed_seconds': time.monotonic() - started, 'training_updates': 0}
            save_json(RECEIPTS / 'preflight_pass.json', result)
            print(json.dumps(result))
        else:
            preflight = json.loads((RECEIPTS / 'preflight_pass.json').read_text())
            assert preflight['status'] == 'PREFLIGHT_EQUIVALENCE_PASS'
            assert preflight['state_digest'] == digest_before
            traces = ROOT / 'traces'
            traces.mkdir(exist_ok=True)
            torch.cuda.reset_peak_memory_stats()
            for index, row in enumerate(rows):
                out, arrays = trace_forward(model, pp, image_tensor(row['path']), budget, 'study100')
                arrays.update({k: v for k, v in row.items() if k.startswith('gt_')})
                np.savez_compressed(traces / (row['path'].stem + '.npz'), **arrays)
                if (index + 1) % 10 == 0:
                    progress = {'status': 'RUNNING', 'images': index + 1, 'full_model_forwards_cumulative': budget.data['attempted_full_model_forwards'],
                                'elapsed_seconds': time.monotonic() - started}
                    save_json(RECEIPTS / 'study_progress.json', progress)
                    print(json.dumps(progress), flush=True)
            assert state_digest(model) == digest_before
            result = {'status': 'STUDY100_TRACE_COMPLETE', 'images': 100, 'positive_gt': 4491,
                      'training_updates': 0, 'full_model_forwards_cumulative': budget.data['attempted_full_model_forwards'],
                      'instrumented_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'model_state_unchanged': True, 'state_digest': digest_before,
                      'elapsed_seconds': time.monotonic() - started,
                      'scope': 'In-sample mechanism diagnostic; no validation/test and no generalization claim'}
            save_json(RECEIPTS / 'study_complete.json', result)
            print(json.dumps(result))


if __name__ == '__main__':
    main()
