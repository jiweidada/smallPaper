"""FP32 EMA prediction export using the project's loader and postprocessor.

Run from an experiment code directory. Does not change training or evaluator code.
Outputs COCO bbox predictions in ORIGINAL coordinates; translations are undone.
"""
import argparse
import hashlib
import json
import pathlib
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def shift_tensor(images, dx, dy):
    import torch
    height, width = images.shape[-2:]
    if abs(dx) >= width or abs(dy) >= height:
        raise ValueError('Translation must be smaller than the input')
    result = torch.zeros_like(images)
    x0, x1 = max(0, -dx), min(width, width-dx)
    y0, y1 = max(0, -dy), min(height, height-dy)
    result[..., y0+dy:y1+dy, x0+dx:x1+dx] = images[..., y0:y1, x0:x1]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--images')
    parser.add_argument('--annotations')
    parser.add_argument('--shift', nargs=2, type=int, default=[0, 0], metavar=('DX', 'DY'))
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit-images', type=int, default=0, help='Smoke check only; marked incomplete')
    parser.add_argument('--project-eval', action='store_true', help='Evaluate the same zero-shift predictions with the unchanged project evaluator')
    parser.add_argument('--register-soep', action='store_true')
    args = parser.parse_args()
    if args.project_eval and any(args.shift):
        raise ValueError('Translated AP uses the common cropped GT in analyze_preds.py')
    destination = pathlib.Path(args.out)
    destination.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0, str(pathlib.Path.cwd()))
    if args.register_soep:
        import engine.deim.soep_lite_round2  # noqa: F401
    from engine.core import YAMLConfig
    cfg = YAMLConfig(args.config)
    settings = cfg.yaml_cfg['val_dataloader']
    settings.update(total_batch_size=args.batch_size, num_workers=args.workers, shuffle=False, drop_last=False)
    settings.pop('batch_size', None)
    if args.images:
        settings['dataset']['img_folder'] = args.images
    if args.annotations:
        settings['dataset']['ann_file'] = args.annotations
    cfg.yaml_cfg['score_threshold'] = 0.0
    cfg.yaml_cfg['PostProcessor']['num_top_queries'] = 300
    cfg.yaml_cfg['use_amp'] = False
    torch.manual_seed(0)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint.get('ema'), dict) or 'module' not in checkpoint['ema']:
        raise ValueError('Expected an EMA module in the selected best_stg2 checkpoint')
    model = cfg.model
    model.load_state_dict(checkpoint['ema']['module'], strict=True)
    model = model.eval().cuda()
    postprocessor = cfg.postprocessor.eval().cuda()
    postprocessor.score_threshold = 0.0
    postprocessor.num_top_queries = 300
    loader = cfg.val_dataloader
    evaluator = cfg.evaluator if args.project_eval else None
    predictions = []
    seen = set()
    started = time.time()
    dx, dy = args.shift
    with torch.inference_mode():
        for index, (samples, targets) in enumerate(loader):
            if tuple(samples.shape[-2:]) != (640, 640):
                raise ValueError('This protocol requires the final input to be 640x640')
            samples = shift_tensor(samples.cuda().float(), dx, dy)
            sizes = torch.stack([target['orig_size'] for target in targets]).cuda()
            results = postprocessor(model(samples), sizes, for_eval=True)
            evaluation_batch = {}
            for target, result, size in zip(targets, results, sizes):
                image_id = int(target['image_id'].item()) if hasattr(target['image_id'], 'item') else int(target['image_id'])
                if image_id in seen:
                    raise ValueError('Repeated image id in deterministic evaluation loader')
                seen.add(image_id)
                boxes = result['boxes'].detach().cpu().double()
                original_width, original_height = size.detach().cpu().double().tolist()
                boxes[:, [0, 2]] -= dx * original_width / 640.0
                boxes[:, [1, 3]] -= dy * original_height / 640.0
                scores = result['scores'].detach().cpu().tolist()
                labels = result['labels'].detach().cpu().tolist()
                assert len(boxes) == len(scores) == len(labels) == 300
                for box, score, label in zip(boxes.tolist(), scores, labels):
                    x0, y0, x1, y1 = box
                    predictions.append({'image_id': image_id, 'category_id': int(label),
                                        'bbox': [x0, y0, x1-x0, y1-y0], 'score': float(score)})
                evaluation_batch[image_id] = result
            if evaluator is not None:
                evaluator.update(evaluation_batch)
            if index % 20 == 0:
                print(json.dumps({'batches': index+1, 'images': len(seen), 'elapsed': time.time()-started}), flush=True)
            if args.limit_images and len(seen) >= args.limit_images:
                break
    annotation = settings['dataset']['ann_file']
    expected_ids = {int(image['id']) for image in json.loads(pathlib.Path(annotation).read_text())['images']}
    complete = seen == expected_ids
    if not args.limit_images and not complete:
        raise ValueError('Prediction export did not cover the complete annotation image set')
    with (destination/'predictions.json').open('x') as output:
        json.dump(predictions, output, allow_nan=False)
    if evaluator is not None:
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
        metrics = evaluator.coco_eval['bbox'].stats.tolist()
        with (destination/'project_eval.json').open('x') as output:
            json.dump({'metrics_raw_0_to_1': metrics, 'same_predictions_as_export': True}, output, indent=2)
    manifest = {'checkpoint': str(pathlib.Path(args.checkpoint).resolve()), 'checkpoint_sha256': sha256(args.checkpoint),
                'checkpoint_epoch': checkpoint.get('last_epoch', checkpoint.get('epoch')),
                'weight_source': 'ema.module', 'config': str(pathlib.Path(args.config).resolve()),
                'config_sha256': sha256(args.config), 'resolved_config': cfg.yaml_cfg,
                'annotations': str(pathlib.Path(annotation).resolve()), 'annotations_sha256': sha256(annotation),
                'predictions_sha256': sha256(destination/'predictions.json'), 'coordinate_space': 'original_image',
                'shift_input_pixels': [dx, dy], 'shift_reversed_in_export': True,
                'input_size': [640, 640], 'dtype': 'FP32', 'amp': False, 'topk': 300, 'score_threshold': 0,
                'image_count': len(seen), 'prediction_count': len(predictions), 'complete': complete,
                'elapsed_seconds': time.time()-started, 'torch_version': torch.__version__}
    with (destination/'manifest.json').open('x') as output:
        json.dump(manifest, output, indent=2, default=str)
    print(json.dumps({'out': str(destination), 'complete': complete, 'images': len(seen)}))


if __name__ == '__main__':
    main()
