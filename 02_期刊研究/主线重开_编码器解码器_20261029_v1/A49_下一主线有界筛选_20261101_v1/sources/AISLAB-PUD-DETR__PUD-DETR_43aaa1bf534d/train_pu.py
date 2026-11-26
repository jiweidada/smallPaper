"""
train_pu.py — PUD-DETR training script (nnPU learning).

This script trains a Deformable DETR model with the proposed Positive-Unlabeled
(nnPU) classification loss. Unmatched queries from Hungarian matching are treated
as unlabeled rather than definite negatives, reducing false-negative supervision
caused by missing annotations.

Example usage:
    python train_pu.py \
        --train_json annotations/pascal_train_drop_0.67.json \
        --val_json   annotations/pascal_val.json \
        --train_img_dir /path/to/train/images \
        --val_img_dir   /path/to/val/images \
        --log_dir  logs/pud_detr \
        --log_name drop_0.67_alpha_5.0 \
        --alpha 5.0 \
        --devices 0

References:
    - Deformable DETR: https://arxiv.org/abs/2010.04159
    - nnPU: Kiryo et al., NeurIPS 2017 (https://arxiv.org/abs/1703.00593)
    - HungarianMatcher / SetCriterion adapted from:
      https://github.com/fundamentalvision/Deformable-DETR
"""

import os
import copy
import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import torchvision
from torchvision.ops.boxes import box_area
from scipy.optimize import linear_sum_assignment

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from transformers import (
    DeformableDetrImageProcessor,
    DeformableDetrConfig,
    DeformableDetrForObjectDetection,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PUD-DETR (Deformable DETR with nnPU loss)"
    )

    # --- Paths ---
    parser.add_argument("--hf_checkpoint", type=str, default="SenseTime/deformable-detr",
                        help="HuggingFace model checkpoint identifier")
    parser.add_argument("--train_json", type=str, required=True,
                        help="Path to COCO-format training annotation JSON")
    parser.add_argument("--val_json", type=str, required=True,
                        help="Path to COCO-format validation annotation JSON")
    parser.add_argument("--train_img_dir", type=str, required=True,
                        help="Directory containing training images")
    parser.add_argument("--val_img_dir", type=str, required=True,
                        help="Directory containing validation images")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Root directory for CSVLogger output")
    parser.add_argument("--log_name", type=str, default="pud_detr",
                        help="Experiment name used as a sub-directory in log_dir")

    # --- Training hyperparameters ---
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for transformer modules")
    parser.add_argument("--lr_backbone", type=float, default=1e-5,
                        help="Learning rate for backbone")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="AdamW weight decay")
    parser.add_argument("--gradient_clip_val", type=float, default=0.1,
                        help="Gradient clipping value")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                        help="Number of linear warm-up steps")
    parser.add_argument("--num_workers", type=int, default=6,
                        help="Number of DataLoader worker processes")
    parser.add_argument("--aux_loss", action="store_true",
                        help="Enable auxiliary decoder losses")

    # --- PUD-DETR specific ---
    parser.add_argument("--alpha", type=float, default=5.0,
                        help=(
                            "Class-prior scaling factor α (Eq. 21 in the paper). "
                            "Compensates for underestimation of true positives due to "
                            "missing labels. Default 5.0 (best on VOC 2007 and diatom)."
                        ))

    # --- Hardware ---
    parser.add_argument("--devices", type=int, nargs="+", default=[0],
                        help="GPU device ID(s) to use (e.g. --devices 0 1)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CocoDetection(torchvision.datasets.CocoDetection):
    """COCO-format detection dataset that returns processor-encoded tensors."""

    def __init__(self, img_dir, ann_file, processor):
        super().__init__(img_dir, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        img_id = self.ids[idx]
        ann = {"image_id": img_id, "annotations": target}
        encoding = self.processor(images=img, annotations=ann, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        labels = encoding["labels"][0]
        return pixel_values, labels


def get_id_label_maps(dataset):
    """Return id→label and label→id dicts from a CocoDetection dataset."""
    cats = dataset.coco.cats
    id2label = {k: v["name"] for k, v in cats.items()}
    label2id = {v["name"]: k for k, v in cats.items()}
    return id2label, label2id


def build_collate_fn(processor):
    """Return a collate function that pads pixel values via the processor."""
    def collate_fn(batch):
        pixel_values, labels = zip(*batch)
        encoding = processor.pad(pixel_values, return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": labels,
        }
    return collate_fn


# ---------------------------------------------------------------------------
# Box utilities
# (Modified from Deformable DETR: https://github.com/fundamentalvision/Deformable-DETR)
# ---------------------------------------------------------------------------

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack(
        [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)],
        dim=-1,
    )


def box_xyxy_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=-1)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU (https://giou.stanford.edu/).
    Expects boxes in [x0, y0, x1, y1] format.
    Returns a [N, M] pairwise matrix.
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


# ---------------------------------------------------------------------------
# Hungarian Matcher
# (Modified from Deformable DETR: https://github.com/fundamentalvision/Deformable-DETR)
# ---------------------------------------------------------------------------

class HungarianMatcher(nn.Module):
    """
    Computes a bipartite matching between model predictions and ground-truth
    objects using the Hungarian algorithm.
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 5, cost_giou: float = 2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cannot be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        alpha, gamma = 0.25, 2.0
        neg_cost = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost[:, tgt_ids] - neg_cost[:, tgt_ids]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


# ---------------------------------------------------------------------------
# Loss: nnPU Sigmoid Focal Loss
# ---------------------------------------------------------------------------

def pu_sigmoid_focal_loss(
    inputs,
    targets,
    num_boxes,
    unl_weight,
    alpha_prior,
    alpha: float = 0.25,
    gamma: float = 2,
):
    """
    Non-negative Positive-Unlabeled (nnPU) focal loss for Deformable DETR.

    Replaces the standard negative classification loss for unmatched queries
    with an nnPU risk estimate (Eq. 20 in the paper):

        L_PUD = π_p * R̂⁺_p  +  max(0,  R̂⁻_u  −  π_p * R̂⁻_p)

    where matched queries provide positive supervision (R̂⁺_p and R̂⁻_p) and
    unmatched queries are treated as unlabeled (R̂⁻_u).

    The non-negative correction is applied globally (after aggregation over
    queries and classes) following the ablation results in the paper.

    Args:
        inputs:      Raw logits, shape [B, Q, C].
        targets:     Binary one-hot targets, same shape as inputs.
                     1 for matched (positive) queries, 0 for unmatched (unlabeled).
        num_boxes:   Normalization factor (total ground-truth boxes in the batch).
        unl_weight:  Scaling factor Q / Q_u applied to the unlabeled negative risk
                     (Eq. 20). Compensates for the smaller number of unlabeled queries.
        alpha_prior: Scaled class prior π̂_p = α · Q_p / Q  (Eq. 21).
        alpha:       Focal-loss α for positive/negative balance (default 0.25).
        gamma:       Focal-loss focusing parameter (default 2).

    Returns:
        Scalar loss tensor.
    """
    prob = inputs.sigmoid()
    backgrounds = torch.zeros_like(targets)

    # Per-element focal losses
    ce_pos  = F.binary_cross_entropy_with_logits(inputs, targets,     reduction="none")
    ce_back = F.binary_cross_entropy_with_logits(inputs, backgrounds, reduction="none")

    p_t = prob * targets + (1 - prob) * (1 - targets)
    focal_pos  = ce_pos  * ((1 - p_t) ** gamma)
    focal_back = ce_back * (prob ** gamma)

    if alpha >= 0:
        focal_pos  = (alpha * targets       + (1 - alpha) * (1 - targets))      * focal_pos
        focal_back = (alpha * backgrounds   + (1 - alpha) * (1 - backgrounds))  * focal_back

    # Positive risk term: R̂⁺_p  (matched queries supervised as positive)
    loss_pos = torch.zeros_like(focal_pos)
    loss_pos[targets == 1] = focal_pos[targets == 1] * alpha_prior

    # Negative risk term: R̂⁻_u − π_p · R̂⁻_p
    #   unmatched queries  →  +unl_weight  (contribute to R̂⁻_u)
    #   matched queries    →  −alpha_prior (subtract R̂⁻_p, scaled by π_p)
    loss_neg = focal_back.clone()
    loss_neg[targets == 0] *= unl_weight
    loss_neg[targets == 1] *= -alpha_prior

    # Global non-negative correction: clamp after aggregating over Q and C
    loss = (loss_pos.mean(1).sum() / num_boxes) + (loss_neg.mean(1).sum() / num_boxes).clamp(min=0)
    return loss


# ---------------------------------------------------------------------------
# PUSetCriterion (nnPU)
# ---------------------------------------------------------------------------

class PUSetCriterion(nn.Module):
    """
    Computes PUD-DETR losses:
      1. Hungarian matching between predictions and ground-truth.
      2. nnPU classification loss: unmatched queries are treated as unlabeled
         rather than definite negatives (Eq. 20 in the paper).
      3. Standard box regression losses (L1 + GIoU) for matched pairs.
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, alpha=5.0):
        """
        Args:
            num_classes:  Number of foreground categories.
            matcher:      HungarianMatcher instance.
            weight_dict:  Loss weight dictionary.
            losses:       List of loss names to compute (e.g. ['labels', 'boxes']).
            focal_alpha:  Focal-loss α (default 0.25).
            alpha:        Class-prior scaling factor (Eq. 21, default 5.0).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.alpha = alpha  # class-prior scaling factor

    # --- individual loss functions ---

    def loss_labels(self, outputs, targets, indices, num_boxes, **kwargs):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        num_batches, num_queries = src_logits.shape[:2]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        # Number of matched (positive) queries across the batch
        Q_p = len(target_classes_o)
        Q   = num_batches * num_queries
        Q_u = Q - Q_p

        # Scaling factors for nnPU (Eq. 20 & 21)
        unl_weight  = Q / Q_u if Q_u > 0 else 1.0
        alpha_prior = self.alpha * (Q_p / Q)

        target_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device,
        )
        target_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_onehot = target_onehot[:, :, :-1]

        loss_ce = (
            pu_sigmoid_focal_loss(
                src_logits, target_onehot, num_boxes,
                unl_weight=unl_weight,
                alpha_prior=alpha_prior,
                alpha=self.focal_alpha,
                gamma=2,
            )
            * src_logits.shape[1]
        )
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes, **kwargs):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        return {
            "loss_bbox": loss_bbox.sum() / num_boxes,
            "loss_giou": loss_giou.sum() / num_boxes,
        }

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {"labels": self.loss_labels, "boxes": self.loss_boxes}
        assert loss in loss_map, f"Unknown loss: {loss}"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        outputs_main = {k: v for k, v in outputs.items() if k not in ("aux_outputs", "enc_outputs")}
        indices = self.matcher(outputs_main, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, aux_indices, num_boxes)
                    losses.update({k + f"_{i}": v for k, v in l_dict.items()})

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])
            enc_indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, enc_indices, num_boxes)
                losses.update({k + "_enc": v for k, v in l_dict.items()})

        return losses


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class PUDDeformableDETR(pl.LightningModule):
    """
    PyTorch Lightning wrapper for PUD-DETR (Deformable DETR with nnPU loss).
    """

    def __init__(
        self,
        hf_checkpoint,
        processor,
        id2label,
        label2id,
        train_dataset,
        val_dataset,
        collate_fn,
        val_json,
        lr=1e-4,
        lr_backbone=1e-5,
        weight_decay=1e-4,
        aux_loss=False,
        warmup_steps=2000,
        batch_size=4,
        num_workers=6,
        alpha=5.0,
    ):
        super().__init__()
        self.processor = processor
        self.id2label = id2label
        self.label2id = label2id
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.collate_fn = collate_fn
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.aux_loss = aux_loss
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.alpha = alpha

        self.model = self._build_model(hf_checkpoint)
        self.weight_dict = self._build_weight_dict()
        self.criterion = self._build_criterion()

        self._total_training_steps = None
        self.coco_preds = []
        self.coco_gt = COCO(val_json)

    # --- model / criterion builders ---

    def _build_model(self, hf_checkpoint):
        cfg = DeformableDetrConfig.from_pretrained(
            hf_checkpoint,
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id,
        )
        return DeformableDetrForObjectDetection.from_pretrained(
            hf_checkpoint, config=cfg, ignore_mismatched_sizes=True
        )

    def _build_weight_dict(self):
        weight_dict = {
            "loss_ce": 1,
            "loss_bbox": self.model.config.bbox_loss_coefficient,
            "loss_giou": self.model.config.giou_loss_coefficient,
        }
        if self.aux_loss:
            aux = {}
            for i in range(self.model.config.decoder_layers - 1):
                aux.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux)
        return weight_dict

    def _build_criterion(self):
        matcher = HungarianMatcher(
            cost_class=self.model.config.class_cost,
            cost_bbox=self.model.config.bbox_cost,
            cost_giou=self.model.config.giou_cost,
        ).to(self.device)
        return PUSetCriterion(
            num_classes=len(self.id2label),
            matcher=matcher,
            weight_dict=self.weight_dict,
            losses=["labels", "boxes"],
            alpha=self.alpha,
        ).to(self.device)

    # --- forward & steps ---

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def _common_step(self, batch, validation=False):
        device = self.device
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        targets = [
            {"labels": t["class_labels"].to(device), "boxes": t["boxes"].to(device)}
            for t in batch["labels"]
        ]

        detr_out = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            output_hidden_states=True,
        )

        outputs = self._decode_outputs(detr_out)
        loss_dict = self.criterion(outputs, targets)
        loss = sum(self.weight_dict[k] * v for k, v in loss_dict.items() if k in self.weight_dict)

        if validation:
            labels = [{k: v.to(device) for k, v in lbl.items()} for lbl in batch["labels"]]
            return loss, loss_dict, detr_out, labels
        return loss, loss_dict

    def _decode_outputs(self, detr_out):
        """Decode intermediate hidden states into pred_logits / pred_boxes."""
        hidden = detr_out.intermediate_hidden_states
        init_ref = detr_out.init_reference_points
        inter_ref = detr_out.intermediate_reference_points

        classes, coords = [], []
        for lvl in range(hidden.shape[1]):
            ref = init_ref if lvl == 0 else inter_ref[:, lvl - 1]
            ref = inverse_sigmoid(ref)
            cls = self.model.class_embed[lvl](hidden[:, lvl])
            delta = self.model.bbox_embed[lvl](hidden[:, lvl])
            if ref.shape[-1] == 4:
                coord = (delta + ref).sigmoid()
            else:
                delta[..., :2] += ref
                coord = delta.sigmoid()
            classes.append(cls)
            coords.append(coord)

        out_class = torch.stack(classes, dim=1)
        out_coord = torch.stack(coords, dim=1)
        outputs = {
            "pred_logits": out_class[:, -1],
            "pred_boxes": out_coord[:, -1],
        }
        if self.aux_loss:
            outputs["aux_outputs"] = [
                {"pred_logits": pl, "pred_boxes": pb}
                for pl, pb in zip(out_class[:, :-1].unbind(1), out_coord[:, :-1].unbind(1))
            ]
        return outputs

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self._common_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict, detr_out, labels = self._common_step(batch, validation=True)
        self.log("validation_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f"validation_{k}", v)

        orig_sizes = torch.stack([lbl["orig_size"] for lbl in labels], dim=0)
        preds = self.processor.post_process_object_detection(detr_out, threshold=0, target_sizes=orig_sizes)

        for lbl, pred in zip(labels, preds):
            img_id = lbl["image_id"]
            boxes = box_xyxy_to_xywh(pred["boxes"]).tolist()
            self.coco_preds.extend([
                {"image_id": img_id.item(), "category_id": cat, "bbox": box, "score": score}
                for cat, score, box in zip(pred["labels"].tolist(), pred["scores"].tolist(), boxes)
            ])
        return loss

    def on_validation_epoch_end(self):
        if not self.coco_preds:
            return
        coco_dt = self.coco_gt.loadRes(self.coco_preds)
        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metric_names = [
            "validation_mAP@[IoU=0.50:0.95]", "validation_mAP@50", "validation_mAP@75",
            "validation_AP(small)", "validation_AP(medium)", "validation_AP(large)",
            "validation_AR@1", "validation_AR@10", "validation_AR@100",
            "validation_AR(small)", "validation_AR(medium)", "validation_AR(large)",
        ]
        for name, val in zip(metric_names, coco_eval.stats):
            self.log(name, val, prog_bar=(name == "validation_mAP@[IoU=0.50:0.95]"))
        self.coco_preds.clear()

    # --- dataloaders & optimizer ---

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, collate_fn=self.collate_fn,
            batch_size=self.batch_size, num_workers=self.num_workers,
            prefetch_factor=1, shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, collate_fn=self.collate_fn,
            batch_size=self.batch_size, num_workers=self.num_workers,
            prefetch_factor=1,
        )

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
             "lr": self.lr_backbone},
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        total_steps = self._total_training_steps
        warmup_steps = min(self.warmup_steps, max(1, total_steps - 1))
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=0.0),
            ],
            milestones=[warmup_steps],
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def setup(self, stage):
        if stage == "fit":
            dl = self.train_dataloader()
            self._total_training_steps = len(dl) * self.trainer.max_epochs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 60)
    print("PUD-DETR — nnPU Training")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k:<22}: {v}")
    print("=" * 60)

    # Processor & datasets
    processor = DeformableDetrImageProcessor.from_pretrained(args.hf_checkpoint)

    trainset = CocoDetection(args.train_img_dir, args.train_json, processor)
    valset   = CocoDetection(args.val_img_dir,   args.val_json,   processor)

    id2label, label2id = get_id_label_maps(trainset)
    collate_fn = build_collate_fn(processor)

    # Model
    model = PUDDeformableDETR(
        hf_checkpoint=args.hf_checkpoint,
        processor=processor,
        id2label=id2label,
        label2id=label2id,
        train_dataset=trainset,
        val_dataset=valset,
        collate_fn=collate_fn,
        val_json=args.val_json,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        aux_loss=args.aux_loss,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        alpha=args.alpha,
    )

    # Trainer
    logger = CSVLogger(save_dir=args.log_dir, name=args.log_name)
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_mAP@[IoU=0.50:0.95]",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="best_validation_mAP_epoch{epoch}",
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        gradient_clip_val=args.gradient_clip_val,
        accelerator="gpu",
        devices=args.devices,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
