"""Training-only local feature supervision. No changes to native predictions."""
from __future__ import annotations
import contextlib
import hashlib
import json
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


def module_state_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def native_state(state):
    return {k: v for k, v in state.items() if not k.startswith('aux_projector.')}


def auxiliary_state(state):
    return {k.removeprefix('aux_projector.'): v for k, v in state.items() if k.startswith('aux_projector.')}


class StudentCapture:
    def __init__(self, model):
        self.model = model;self.feature = None;self.handle = None;self.install()

    def install(self):
        assert self.handle is None
        def capture(module, args, output):
            assert self.model.training, 'Student feature hooks are only installed for training'
            self.feature = output
        self.handle = self.model.backbone.backbone.patch_embed.convs[1].register_forward_hook(capture)

    def remove(self):
        if self.handle is not None:
            self.handle.remove();self.handle = None
        self.feature = None


class TeacherCapture:
    def __init__(self, model):
        self.model = model;self.memory = None;self.query = None
        def memory(module, args, output):
            assert not model.training and len(output) == 3
            self.memory = output[1]
        def query(module, args):
            assert not model.training and args[0].shape[1:] == (300, 192)
            self.query = args[0]
        self.handles = [model.encoder.register_forward_hook(memory),
                        model.decoder.dec_score_head[model.decoder.eval_idx].register_forward_pre_hook(query)]

    def clear(self):
        self.memory = None;self.query = None

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.clear()


def make_rois(boxes, batch_indices, feature):
    h, w = feature.shape[-2:]
    xyxy = torch.cat([boxes[:, :2] - boxes[:, 2:] / 2,
                      boxes[:, :2] + boxes[:, 2:] / 2], dim=1).clamp(0, 1)
    xyxy = xyxy * xyxy.new_tensor([w, h, w, h])
    return torch.cat([batch_indices[:, None].to(xyxy.dtype), xyxy], dim=1)


def pool_regions(feature, boxes, batch_indices):
    with torch.autocast(device_type=feature.device.type, enabled=False):
        rois = make_rois(boxes.float(), batch_indices, feature)
        pooled = roi_align(feature.float(), rois, output_size=(3, 3),
                           spatial_scale=1.0, sampling_ratio=2, aligned=True)
        return pooled.mean(dim=(-2, -1))


@torch.no_grad()
def teacher_targets(teacher, matcher, capture, images, targets):
    assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
    capture.clear()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = teacher(images)
    for key in ['pred_logits', 'pred_boxes']:
        if not torch.isfinite(output[key]).all():
            raise FloatingPointError('Nonfinite teacher ' + key)
    assert capture.memory.shape[1:] == (192, 40, 40)
    assert capture.query.shape == (images.shape[0], 300, 192)
    assert not capture.query.requires_grad and not capture.memory.requires_grad
    small_output = {k: output[k].float() for k in ['pred_logits', 'pred_boxes']}
    matches = matcher(small_output, targets)['indices']
    rows, gt_boxes, bi, q_values = [], [], [], []
    for batch_index, ((query_index, gt_index), target) in enumerate(zip(matches, targets)):
        assert len(query_index) == min(300, len(target['boxes']))
        rows.append({'image_id': int(target['image_id']), 'query_indices': query_index.tolist(),
                     'GT_indices': gt_index.tolist(), 'GT_total': len(target['boxes'])})
        if len(query_index):
            qi = query_index.to(images.device);gi = gt_index.to(images.device)
            gt_boxes.append(target['boxes'][gi])
            bi.append(torch.full((len(qi),), batch_index, device=images.device, dtype=torch.long))
            q_values.append(capture.query[batch_index, qi].float())
    if gt_boxes:
        boxes = torch.cat(gt_boxes);batch_indices = torch.cat(bi)
        q = F.normalize(torch.cat(q_values), dim=1, eps=1e-6)
        m = F.normalize(pool_regions(capture.memory, boxes, batch_indices), dim=1, eps=1e-6)
    else:
        boxes = images.new_empty((0, 4), dtype=torch.float32)
        batch_indices = torch.empty(0, device=images.device, dtype=torch.long)
        q = images.new_empty((0, 192), dtype=torch.float32);m = q.clone()
    assert all(torch.isfinite(v).all() for v in [boxes, m, q])
    record = {'instances': rows, 'matched_instances': len(boxes),
              'unmatched_GT_due_to_300_query_limit': sum(max(0, len(t['boxes']) - 300) for t in targets),
              'GT_boxes_sha256': tensor_digest(boxes), 'regional_targets_sha256': tensor_digest(m),
              'query_targets_sha256': tensor_digest(q),
              'teacher_predictions_sha256': tensor_digest(torch.cat([small_output['pred_logits'], small_output['pred_boxes']], dim=-1))}
    capture.clear()
    del output, small_output
    return {'boxes': boxes, 'batch_indices': batch_indices, 'M': m, 'Q': q,
            'record': record}


def auxiliary_loss(model, feature, bundle, arm):
    assert arm in ['M', 'Q'] and feature is not None and feature.shape[1:] == (48, 160, 160)
    assert not bundle[arm].requires_grad
    with torch.autocast(device_type=feature.device.type, enabled=False):
        if len(bundle['boxes']) == 0:
            return model.aux_projector.weight.sum() * 0
        local = pool_regions(feature, bundle['boxes'], bundle['batch_indices'])
        projected = F.normalize(model.aux_projector(local), dim=1, eps=1e-6)
        return (1 - (projected * bundle[arm]).sum(dim=1)).mean()


@contextlib.contextmanager
def native_inference(model, capture=None):
    """Physically remove auxiliary parameters/hooks during native evaluation."""
    auxiliary = getattr(model, 'aux_projector', None)
    if capture is not None:
        capture.remove()
    if auxiliary is not None:
        del model.aux_projector
    try:
        assert not hasattr(model, 'aux_projector')
        yield model
    finally:
        if auxiliary is not None:
            model.aux_projector = auxiliary
        if capture is not None:
            capture.install()


def update_teacher_flow(hasher, bundle):
    encoded = json.dumps(bundle['record'], sort_keys=True, separators=(',', ':')).encode()
    hasher.update(len(encoded).to_bytes(8, 'little'));hasher.update(encoded)
    return hashlib.sha256(encoded).hexdigest()
