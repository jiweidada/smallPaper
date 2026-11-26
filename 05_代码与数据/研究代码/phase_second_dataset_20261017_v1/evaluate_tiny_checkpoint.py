"""Independent evaluation of actual EMA checkpoints; no resume/output-path override."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time

import torch
from tiny_protocol import evaluate_records, install_native_adapter


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    install_native_adapter()
    from engine.core import YAMLConfig
    from engine.solver.det_engine import _build_resize_pad_metadata
    cfg = YAMLConfig(args.config)
    cfg.yaml_cfg['val_dataloader']['dataset']['img_folder'] = str(args.data / ('test_original' if args.split == 'test' else 'train_original'))
    cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'] = str(args.data / 'annotations' / (args.split + '.json'))
    cfg.yaml_cfg['val_dataloader']['num_workers'] = 2
    cfg.yaml_cfg['val_dataloader']['total_batch_size'] = 4
    model = cfg.model
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert isinstance(checkpoint.get('ema'), dict) and 'module' in checkpoint['ema']
    model.load_state_dict(checkpoint['ema']['module'], strict=True)
    model = model.cuda().eval(); post = cfg.postprocessor.cuda().eval()
    predictions = []; seen = set(); started = time.monotonic()
    loader = cfg.val_dataloader
    with torch.inference_mode():
        for samples, targets in loader:
            samples = samples.cuda(non_blocking=True)
            sizes = torch.stack([target['orig_size'] for target in targets]).cuda()
            resize_pad = _build_resize_pad_metadata(targets, samples)
            results = post(model(samples), sizes, for_eval=True, resize_pad=resize_pad)
            for target, result in zip(targets, results):
                image_id = int(target['image_id'].item()); assert image_id not in seen; seen.add(image_id)
                boxes = result['boxes'].cpu().tolist(); scores = result['scores'].cpu().tolist(); labels = result['labels'].cpu().tolist()
                assert len(scores) <= 300 and all(int(label) == 0 for label in labels)
                for box, score in zip(boxes, scores):
                    predictions.append({'image_id': image_id, 'category_id': 1 if args.split == 'test' else 0,
                                        'bbox': [box[0], box[1], box[2]-box[0], box[3]-box[1]], 'score': float(score)})
    gt_path = args.data / 'annotations' / ('raw_tiny_set_test_all.json' if args.split == 'test' else 'val.json')
    annotation = json.loads(gt_path.read_text())
    assert seen == {int(image['id']) for image in annotation['images']}
    named, ordered, _ = evaluate_records(annotation, predictions)
    with (args.out / 'predictions.json').open('x') as handle: json.dump(predictions, handle)
    result = {'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'dataset': 'TinyPerson_v1', 'split': args.split, 'images': len(seen),
              'checkpoint': str(args.checkpoint), 'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'checkpoint_epoch_zero_based': checkpoint['last_epoch'], 'best_stat': checkpoint.get('best_stat'),
              'selection': 'held-out original-training groups; AP50_tiny-selected best_stg2 EMA',
              'ground_truth': str(gt_path), 'ground_truth_sha256': hashlib.sha256(gt_path.read_bytes()).hexdigest(),
              'metrics_raw_0_to_1': named, 'metrics_percent': {key: value*100 for key, value in named.items()},
              'elapsed_seconds': time.monotonic()-started, 'protocol': {'maxDets': 200, 'input': [640,640],
                  'whole_image_stretch': True, 'tiling': False, 'TTA': False, 'ignore_uncertain': True, 'use_iod_for_ignore': True}}
    with (args.out / 'metrics.json').open('x') as handle: json.dump(result, handle, indent=2)
    print(json.dumps({'split': args.split, 'AP50_tiny': named['AP50_tiny'], 'out': str(args.out)}), flush=True)


if __name__ == '__main__': main()

