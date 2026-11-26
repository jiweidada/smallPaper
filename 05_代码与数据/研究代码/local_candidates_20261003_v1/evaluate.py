"""Full-original-image evaluation, shared merge controls, raw source-view predictions."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import bootstrap
from factory import build_model, build_postprocessor, load_config, verify_snapshot
from dataset import OriginalCocoImages
from geometry import make_views, windows, map_local_prediction, merge_predictions

METRICS = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']


def prediction_rows(prediction, image_id):
    boxes = prediction['boxes'].detach().cpu()
    xywh = boxes.clone(); xywh[:, 2:] -= xywh[:, :2]
    return [{'image_id': image_id, 'category_id': int(c), 'bbox': b.tolist(), 'score': float(s)}
            for b, s, c in zip(xywh, prediction['scores'].cpu(), prediction['labels'].cpu())]


@torch.inference_mode()
def infer_original(model, image, cfg, post, device):
    h, w = image.shape[-2:]
    choices = windows(w, h, stride=cfg.get('tile_stride', 512))
    full, _, _ = make_views(image, choices[0])
    full = full[None].to(device)
    if cfg['kind'] == 'queries':
        return post(model(full), torch.tensor([[w, h]], device=device))[0], [], 1
    features, outputs = model.cache_global(full)
    global_prediction = post(outputs, torch.tensor([[w, h]], device=device))[0]
    source_predictions = [('global', None, global_prediction)]
    for window in choices:
        _, local, crop = make_views(image, window)
        outputs = model.predict_local(local[None].to(device), crop[None].to(device), features)
        local_prediction = post(outputs, torch.tensor([[640, 640]], device=device))[0]
        mapped = map_local_prediction(local_prediction, window, (h, w))
        source_predictions.append(('local', window, mapped))
    merged = merge_predictions([p for _, _, p in source_predictions],
                               cfg['merge_iou'], cfg['image_topk'])
    return merged, source_predictions, 1 + len(choices)


def coco_evaluate(annotation_path, rows, output):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO(str(annotation_path))
    if not rows:
        raise RuntimeError('No predictions; preserve failure rather than invent metrics')
    dt = gt.loadRes(rows)
    evaluator = COCOeval(gt, dt, 'bbox')
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.params.imgIds = sorted(gt.imgs)
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    np.savez_compressed(output / 'coco_arrays.npz', precision=evaluator.eval['precision'],
                        recall=evaluator.eval['recall'], scores=evaluator.eval['scores'])
    per_class = {}
    for index, category in enumerate(evaluator.params.catIds):
        p = evaluator.eval['precision'][:, :, index, 1, -1]
        valid = p[p >= 0]
        per_class[gt.cats[category]['name']] = float(valid.mean()) if len(valid) else None
    return {'metrics': dict(zip(METRICS, map(float, evaluator.stats))), 'per_class_APs': per_class,
            'images': len(gt.imgs), 'maxDets': [1, 10, 100], 'full_original_coordinates': True}


def evaluate_dataset(model, cfg, image_root, annotations, output, device, trace=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    dataset = OriginalCocoImages(image_root, annotations)
    model.eval(); post = build_postprocessor().to(device)
    rows, raw, timings, forwards = [], [], [], 0
    if trace:
        if cfg['kind'] != 'queries':
            raise ValueError('Query trace applies to the single-view query candidate')
        model.decoder.trace_enabled = True
        (output / 'query_traces').mkdir()
    for index in range(len(dataset)):
        image, target = dataset[index]
        image_id = int(target['image_id'].item())
        if device.type == 'cuda': torch.cuda.synchronize()
        started = time.perf_counter()
        prediction, sources, count = infer_original(model, image, cfg, post, device)
        if device.type == 'cuda': torch.cuda.synchronize()
        timings.append(time.perf_counter() - started); forwards += count
        rows.extend(prediction_rows(prediction, image_id))
        for kind, window, pred in sources:
            raw.append({'image_id': image_id, 'view': kind, 'window': window,
                        'original_coordinate_predictions': prediction_rows(pred, image_id)})
        if trace:
            np.savez_compressed(output / 'query_traces' / f'{image_id}.npz',
                **{k: v.numpy() for k, v in model.decoder.last_query_trace.items()})
    (output / 'predictions.json').write_text(json.dumps(rows), encoding='utf8')
    (output / 'raw_view_predictions.json').write_text(json.dumps(raw), encoding='utf8')
    result = coco_evaluate(annotations, rows, output)
    result.update({'status': 'candidate_full_val_not_frozen_C_acceptance',
        'config': cfg, 'forward_views': forwards, 'per_image_seconds': timings,
        'timing_scope': 'view construction/model/postprocess/merge; excludes file read and COCO evaluation',
        'FP32_no_deploy': True, 'query_diagnostic_extra_compute': trace,
        'annotation_sha256': __import__('hashlib').sha256(Path(annotations).read_bytes()).hexdigest()})
    (output / 'evaluation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--images', required=True); parser.add_argument('--annotations', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--device', default='cpu')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--allow-query-selection-swap', action='store_true',
                        help='Same-weight diagnostic only: apply a different query selector explicitly')
    args = parser.parse_args(); verify_snapshot(); cfg = load_config(args.config)
    torch.set_num_threads(4)
    model = build_model(cfg)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    bound_cfg = ck.get('bindings', {}).get('config')
    swapped = bound_cfg is not None and bound_cfg != cfg
    if swapped and not (args.allow_query_selection_swap and cfg['kind'] == bound_cfg['kind'] == 'queries'):
        raise ValueError('Checkpoint/config differs; do not silently relabel a trained arm')
    state = ck.get('ema', {}).get('module', ck.get('model', ck))
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device); model.to(device)
    result = evaluate_dataset(model, cfg, args.images, args.annotations, args.output, device, args.trace)
    result.update({'checkpoint_sha256': __import__('hashlib').sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                   'checkpoint_path': str(Path(args.checkpoint).resolve()),
                   'trained_checkpoint_config': bound_cfg, 'explicit_same_weight_query_swap': swapped,
                   'init_or_external_checkpoint_without_candidate_binding': bound_cfg is None})
    (Path(args.output) / 'evaluation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()

