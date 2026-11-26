"""Whole-image warp, native crops, feature alignment and original-coordinate merge."""
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms, box_convert


def edge_positions(length, tile=640, stride=512):
    if tile <= 0 or not 0 < stride <= tile or length <= 0:
        raise ValueError('Invalid window geometry')
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def windows(width, height, tile=640, stride=512):
    # Width/height smaller than tile are right/bottom padded, never enlarged.
    return [(x, y, x + tile, y + tile)
            for y in edge_positions(height, tile, stride)
            for x in edge_positions(width, tile, stride)]


def make_views(image, window, size=640):
    if image.ndim != 3 or image.shape[0] != 3 or not image.is_floating_point():
        raise ValueError('Expected RGB float32 [3,H,W] in [0,1]')
    height, width = image.shape[-2:]
    x0, y0, x1, y1 = map(int, window)
    if not (0 <= x0 < width and 0 <= y0 < height and
            x1 - x0 == size and y1 - y0 == size):
        raise ValueError('Expected a native-size window anchored inside the image')
    full = F.interpolate(image[None], size=(size, size), mode='bilinear',
                         align_corners=False, antialias=True)[0]
    local = image[:, y0:min(y1, height), x0:min(x1, width)]
    local = F.pad(local, (0, size - local.shape[-1], 0, size - local.shape[-2]))
    # Allow x1/width or y1/height >1 for padding. Invalid aligned sites are masked.
    norm = image.new_tensor([x0 / width, y0 / height, x1 / width, y1 / height])
    return full, local, norm


def alignment_grid(crop_norm, output_hw):
    if crop_norm.ndim != 2 or crop_norm.shape[1] != 4:
        raise ValueError('crop_norm must be [B,4] normalized original xyxy')
    if not torch.isfinite(crop_norm).all() or not (crop_norm[:, 2:] > crop_norm[:, :2]).all():
        raise ValueError('Nonfinite or inverted crop coordinates')
    h, w = output_hw
    # align_corners=False maps normalized pixel centres consistently in both views.
    yy, xx = torch.meshgrid(
        (torch.arange(h, device=crop_norm.device, dtype=crop_norm.dtype) + .5) / h,
        (torch.arange(w, device=crop_norm.device, dtype=crop_norm.dtype) + .5) / w,
        indexing='ij')
    xy = torch.stack((xx, yy), -1)[None]
    start, span = crop_norm[:, :2], crop_norm[:, 2:] - crop_norm[:, :2]
    coords = start[:, None, None] + xy * span[:, None, None]
    valid = ((coords >= 0) & (coords <= 1)).all(-1)[:, None]
    return coords * 2 - 1, valid


def align_global_p3(global_p3, crop_norm, output_hw):
    grid, valid = alignment_grid(crop_norm.float(), output_hw)
    aligned = F.grid_sample(global_p3, grid.to(global_p3.dtype), mode='bilinear',
                            padding_mode='border', align_corners=False)
    return aligned * valid.to(aligned.dtype), valid


def targets_for_window(target, window, image_hw, size=640):
    h, w = image_hw
    boxes = target['boxes'].float()  # original pixel xyxy
    x0, y0, x1, y1 = window
    shift = boxes.new_tensor([x0, y0, x0, y0])
    upper = boxes.new_tensor([min(x1, w), min(y1, h)] * 2)
    lower = boxes.new_tensor([x0, y0] * 2)
    clipped = boxes.maximum(lower).minimum(upper) - shift
    wh = clipped[:, 2:] - clipped[:, :2]
    keep = (wh >= 1).all(-1)
    area = wh[keep].prod(-1)
    return {'image_id': target['image_id'], 'boxes': box_convert(clipped[keep], 'xyxy', 'cxcywh') / size,
            'labels': target['labels'][keep], 'area': area,
            'iscrowd': torch.zeros(len(area), dtype=torch.int64),
            'orig_size': torch.tensor([size, size]),
            'annotation_ids': target['annotation_ids'][keep]}


def targets_for_global(target, image_hw):
    h, w = image_hw
    boxes = box_convert(target['boxes'].float(), 'xyxy', 'cxcywh') / torch.tensor([w, h, w, h])
    return {**target, 'boxes': boxes, 'orig_size': torch.tensor([h, w])}


def map_local_prediction(prediction, window, image_hw):
    h, w = image_hw
    x0, y0, _, _ = window
    boxes = prediction['boxes'] + prediction['boxes'].new_tensor([x0, y0, x0, y0])
    # Ignore boxes centred in padded pixels before clipping.
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2
    keep = (centres[:, 0] >= 0) & (centres[:, 0] < w) & (centres[:, 1] >= 0) & (centres[:, 1] < h)
    boxes = boxes.clamp(min=0).minimum(boxes.new_tensor([w, h, w, h]))
    keep &= (boxes[:, 2:] > boxes[:, :2]).all(-1)
    return {k: (boxes if k == 'boxes' else v)[keep] for k, v in prediction.items()}


def merge_predictions(predictions, iou=.5, topk=300):
    # Both arms use identical class-aware NMS and image-level output cap.
    combined = {k: torch.cat([p[k] for p in predictions]) for k in ('boxes', 'scores', 'labels')}
    valid = torch.isfinite(combined['boxes']).all(-1) & torch.isfinite(combined['scores'])
    valid &= (combined['boxes'][:, 2:] > combined['boxes'][:, :2]).all(-1)
    combined = {k: v[valid] for k, v in combined.items()}
    keep = batched_nms(combined['boxes'], combined['scores'], combined['labels'], iou)[:topk]
    return {k: v[keep] for k, v in combined.items()}

