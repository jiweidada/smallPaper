"""Compare pre-decoder proposal coverage; GT is used only offline, never in selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torchvision.ops import box_convert, box_iou


def coverage_summary(trace, annotations, image_wh, iou_threshold=.5):
    valid = torch.from_numpy(trace['valid'][0])
    boxes = torch.from_numpy(trace['boxes'][0])
    logits = torch.from_numpy(trace['logits'][0])
    labels = logits.argmax(-1)
    indices = torch.from_numpy(trace['selected_indices'][0]).long()
    if len(indices) != 300 or len(indices.unique()) != 300:
        raise ValueError('Expected 300 unique queries')
    w, h = image_wh
    boxes = box_convert(boxes, 'cxcywh', 'xyxy') * torch.tensor([w, h, w, h])
    target_boxes = torch.tensor([a['bbox'] for a in annotations]).reshape(-1, 4).float()
    target_boxes = box_convert(target_boxes, 'xywh', 'xyxy')
    target_labels = torch.tensor([a['category_id'] for a in annotations])
    if not len(target_boxes):
        return {'all': 0, 'small': 0, 'eligible_covered': 0, 'selected_covered': 0,
                'small_eligible_covered': 0, 'small_selected_covered': 0}
    # Chunked diagnostic avoids an all-image NxGT allocation.
    def covered(index):
        found = torch.zeros(len(target_boxes), dtype=torch.bool)
        for block in index.split(256):
            overlap = box_iou(boxes[block], target_boxes)
            correct_class = labels[block, None] == target_labels[None]
            found |= ((overlap >= iou_threshold) & correct_class).any(0)
        return found
    eligible = covered(valid.nonzero().flatten())
    selected = covered(indices)
    # Original annotation area, COCO-small range. This is a recall proxy, not APs.
    small = torch.tensor([a.get('area', a['bbox'][2] * a['bbox'][3]) < 32 ** 2 for a in annotations])
    return {'all': len(annotations), 'small': int(small.sum()),
            'eligible_covered': int(eligible.sum()), 'selected_covered': int(selected.sum()),
            'small_eligible_covered': int((eligible & small).sum()),
            'small_selected_covered': int((selected & small).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traces', required=True); parser.add_argument('--annotations', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    annotation = json.loads(Path(args.annotations).read_text(encoding='utf8'))
    by_image = {x['id']: [] for x in annotation['images']}
    for ann in annotation['annotations']:
        if not ann.get('iscrowd', 0) and ann['bbox'][2] > 0 and ann['bbox'][3] > 0:
            by_image[ann['image_id']].append(ann)
    rows = []
    for image in annotation['images']:
        path = Path(args.traces) / f'{image["id"]}.npz'
        with np.load(path, allow_pickle=False) as trace:
            rows.append({'image_id': image['id'], 'IoU50': coverage_summary(trace, by_image[image['id']],
                (image['width'], image['height']), .5), 'IoU75': coverage_summary(trace, by_image[image['id']],
                (image['width'], image['height']), .75)})
    result = {'status': 'offline_predecoder_class_correct_box_coverage_diagnostic',
              'not_APs_or_final_recall': True, 'GT_used_by_selector': False, 'images': rows}
    with Path(args.output).open('x', encoding='utf8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
