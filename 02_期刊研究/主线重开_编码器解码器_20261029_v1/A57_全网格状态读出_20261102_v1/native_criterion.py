import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def box_xyxy_to_cxcywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dim=-1)

def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dim=-1)

def box_iou(boxes1, boxes2):
    area1 = (boxes1[:,2]-boxes1[:,0]) * (boxes1[:,3]-boxes1[:,1])
    area2 = (boxes2[:,2]-boxes2[:,0]) * (boxes2[:,3]-boxes2[:,1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:,:,0] * wh[:,:,1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-7)
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    enclosing = wh[:,:,0] * wh[:,:,1]
    giou = iou - (enclosing - union) / (enclosing + 1e-7)
    return giou

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_bbox=2.0, cost_giou=5.0,
                 num_classes=10, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        B, Nq = outputs['pred_logits'].shape[:2]

        out_prob = outputs['pred_logits'].sigmoid()
        out_bbox = outputs['pred_boxes']

        alpha = self.focal_alpha
        gamma = self.focal_gamma

        indices = []
        for b in range(B):
            tgt_ids = targets[b]['labels']
            tgt_bbox = targets[b]['boxes']

            if len(tgt_ids) == 0:
                indices.append((torch.tensor([], dtype=torch.long),
                                torch.tensor([], dtype=torch.long)))
                continue

            prob = out_prob[b]
            neg_cost = (1 - alpha) * (prob ** gamma) * (-(1 - prob + 1e-8).log())
            pos_cost = alpha * ((1 - prob) ** gamma) * (-(prob + 1e-8).log())
            cost_cls = pos_cost[:, tgt_ids] - neg_cost[:, tgt_ids]

            cost_l1 = torch.cdist(out_bbox[b], tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox[b]),
                box_cxcywh_to_xyxy(tgt_bbox))

            C = (self.cost_class * cost_cls +
                 self.cost_bbox * cost_l1 +
                 self.cost_giou * cost_giou).cpu()

            C[C.isnan() | C.isinf()] = 1e4

            r, c_ = linear_sum_assignment(C.numpy())
            indices.append((torch.as_tensor(r, dtype=torch.long),
                            torch.as_tensor(c_, dtype=torch.long)))
        return indices

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, losses,
                 focal_alpha=0.25, focal_gamma=1.5, class_freq=None):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        if class_freq is not None:
            freq = torch.tensor(class_freq, dtype=torch.float32)
            inv_sqrt = 1.0 / freq.sqrt()
            alpha_per_class = inv_sqrt * focal_alpha / inv_sqrt.mean()
            alpha_per_class = alpha_per_class.clamp(min=0.05, max=0.8)
            self.register_buffer('alpha_per_class', alpha_per_class)
        else:
            self.alpha_per_class = None

    def loss_labels(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        B, Nq, C = pred_logits.shape

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([
            t['labels'][j] for t, (_, j) in zip(targets, indices)
        ])

        target_classes = torch.full(
            (B, Nq), self.num_classes,
            dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_o

        target_onehot = torch.zeros(
            (B, Nq, self.num_classes + 1),
            dtype=pred_logits.dtype, device=pred_logits.device
        )
        target_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_onehot = target_onehot[:, :, :-1]

        gamma = self.focal_gamma
        prob = pred_logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(
            pred_logits, target_onehot, reduction='none'
        )
        p_t = prob * target_onehot + (1 - prob) * (1 - target_onehot)
        focal = ce_loss * ((1 - p_t) ** gamma)

        if self.alpha_per_class is not None:
            pos_alpha = self.alpha_per_class.view(1, 1, -1)
            neg_alpha = 1.0 - self.focal_alpha
            alpha_t = pos_alpha * target_onehot + neg_alpha * (1 - target_onehot)
        else:
            alpha = self.focal_alpha
            alpha_t = alpha * target_onehot + (1 - alpha) * (1 - target_onehot)

        focal = alpha_t * focal

        loss_ce = focal.mean(1).sum() / max(num_boxes, 1)
        return {'loss_ce': loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([
            t['boxes'][j] for t, (_, j) in zip(targets, indices)
        ], dim=0)

        if len(src_boxes) == 0:
            dev = outputs['pred_boxes'].device
            return {'loss_bbox': torch.tensor(0., device=dev),
                    'loss_giou': torch.tensor(0., device=dev)}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / max(num_boxes, 1)

        src_xyxy = box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_cxcywh_to_xyxy(target_boxes)
        giou = torch.diag(generalized_box_iou(src_xyxy, tgt_xyxy))
        loss_giou = (1 - giou).sum() / max(num_boxes, 1)

        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(s, i) for i, (s, _) in enumerate(indices)])
        src_idx = torch.cat([s for (s, _) in indices])
        return batch_idx, src_idx

    def _compute_losses_single(self, outputs, targets, num_boxes):
        indices = self.matcher(outputs, targets)
        losses = {}
        for loss_type in self.losses:
            if loss_type == 'labels':
                losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
            elif loss_type == 'boxes':
                losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        return losses

    def _group_losses(self, outputs, targets, num_boxes, num_groups):
        if num_groups == 1:
            return self._compute_losses_single(outputs, targets, num_boxes)

        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        B, total_Nq, C = pred_logits.shape
        Nq = total_Nq // num_groups

        pred_logits_g = pred_logits.reshape(B, num_groups, Nq, C)
        pred_boxes_g = pred_boxes.reshape(B, num_groups, Nq, 4)

        accumulated = {}
        for g in range(num_groups):
            group_out = {
                'pred_logits': pred_logits_g[:, g],
                'pred_boxes': pred_boxes_g[:, g],
            }
            group_losses = self._compute_losses_single(group_out, targets, num_boxes)
            for k, v in group_losses.items():
                if k not in accumulated:
                    accumulated[k] = v
                else:
                    accumulated[k] = accumulated[k] + v

        for k in accumulated:
            accumulated[k] = accumulated[k] / num_groups

        return accumulated

    def forward(self, outputs, targets):
        num_groups = outputs.get('num_groups', 1)

        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float,
            device=outputs['pred_logits'].device
        ).clamp(min=1).item()

        losses = self._group_losses(outputs, targets, num_boxes, num_groups)

        if 'aux_outputs' in outputs:
            for i, aux_out in enumerate(outputs['aux_outputs']):
                aux_losses = self._group_losses(aux_out, targets, num_boxes, num_groups)
                losses.update({f'{k}_aux{i}': v for k, v in aux_losses.items()})

        return losses
