"""GT-free label mapping of the native actual selected score occurrences.

This is an isolated inference adapter, not a change to native TopK tie policy.
The carrier budget is before local clipping, class-aware NMS and COCO evaluation.
"""
import torch
from torch import nn
from engine.deim.postprocessor import scale_boxes_to_original_image


@torch.no_grad()
def select_carrier(native,semantic,k=300):
    if native.ndim!=3 or native.shape[-2:]!=(300,10) or native.shape!=semantic.shape:
        raise ValueError('Requires [B,300,10] native and semantic logits.')
    if k!=300 or native.device!=semantic.device or native.dtype!=semantic.dtype:
        raise ValueError('Fixed Top300 and equal dtype/device required.')
    if not native.is_floating_point() or not torch.isfinite(native).all() or not torch.isfinite(semantic).all():
        raise ValueError('Finite floating logits required.')
    scores,flat=torch.topk(native.sigmoid().flatten(1),k,dim=-1)
    query=flat//10;old_label=flat%10
    order=torch.argsort(native,dim=-1,descending=True,stable=True)
    inverse=torch.empty_like(order).scatter(-1,order,torch.arange(10,device=native.device).expand_as(order))
    old_rank=inverse.flatten(1).gather(1,flat)
    semantic_order=torch.argsort(semantic,dim=-1,descending=True,stable=True)
    labels=semantic_order.flatten(1).gather(1,query*10+old_rank)
    return {'scores':scores,'query':query,'labels':labels,'native_labels':old_label,
            'native_flat_indices':flat,'native_class_rank':old_rank}


class BudgetPostProcessor(nn.Module):
    def __init__(self,native_postprocessor,trace=False):
        super().__init__();self.native=native_postprocessor
        if (not self.native.use_focal_loss or self.native.num_classes!=10 or
            self.native.num_top_queries!=300 or self.native.remap_mscoco_category):
            raise ValueError('Requires native focal/10-class/Top300/non-remapped postprocessor.')
        self.trace=trace;self.last_carrier=None;self.deploy_mode=False

    @torch.no_grad()
    def forward(self,outputs,orig_target_sizes,for_eval=False,resize_pad=None):
        if 'semantic_logits' not in outputs:
            if self.deploy_mode:
                raise ValueError('Deploy adapter requires semantic metadata.')
            return self.native(outputs,orig_target_sizes,for_eval,resize_pad)
        native,boxes=outputs['pred_logits'],outputs['pred_boxes']
        if boxes.shape!=(*native.shape[:2],4) or not torch.isfinite(boxes).all():
            raise ValueError('Finite paired native boxes required.')
        carrier=select_carrier(native,outputs['semantic_logits'])
        scaled=scale_boxes_to_original_image(boxes,orig_target_sizes,resize_pad)
        selected=scaled.gather(1,carrier['query'][...,None].repeat(1,1,4))
        if self.trace:self.last_carrier={k:v.detach().clone() for k,v in carrier.items()}
        labels,scores=carrier['labels'],carrier['scores']
        if self.deploy_mode:return labels,selected,scores
        return [{'labels':lab[sco>self.native.score_threshold],
                 'boxes':box[sco>self.native.score_threshold],
                 'scores':sco[sco>self.native.score_threshold]}
                for lab,box,sco in zip(labels,selected,scores)]

    def deploy(self):
        self.eval();self.deploy_mode=True;return self
