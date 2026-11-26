"""Controlled R50-FPN/DynamicHead endpoint detector; one inference head pass."""
from types import SimpleNamespace as NS
import torch
from torch import nn
from torch_detector_parts_v1 import FPN
from head_query_time_A60_v1 import DynamicHead
from box_progress_conditioning_v1 import xywh_to_xyxy

def configuration():
    dd=NS(NUM_CLASSES=10,NUM_PROPOSALS=1024,NHEADS=8,DROPOUT=0.,DIM_FEEDFORWARD=2048,
        ACTIVATION='relu',HIDDEN_DIM=256,NUM_CLS=1,NUM_REG=3,NUM_HEADS=6,NUM_DYNAMIC=2,
        DIM_DYNAMIC=64,DEEP_SUPERVISION=True,USE_FOCAL=True,USE_FED_LOSS=False,
        PRIOR_PROB=.01,CLASS_WEIGHT=2.,GIOU_WEIGHT=2.,L1_WEIGHT=5.,NO_OBJECT_WEIGHT=.1,
        ALPHA=.25,GAMMA=2.,OTA_K=5)
    return NS(MODEL=NS(DiffusionDet=dd,ROI_HEADS=NS(IN_FEATURES=['p2','p3','p4','p5']),
        ROI_BOX_HEAD=NS(POOLER_TYPE='ROIAlignV2',POOLER_RESOLUTION=7,POOLER_SAMPLING_RATIO=2)))

class EndpointDetector(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=FPN()
        self.head=DynamicHead(configuration(),{k:NS(channels=256,stride=s) for k,s in zip(['p2','p3','p4','p5'],[4,8,16,32])})
        self.register_buffer('pixel_mean',torch.tensor([123.675,116.280,103.530]).view(1,3,1,1),persistent=False)
        self.register_buffer('pixel_std',torch.tensor([58.395,57.120,57.375]).view(1,3,1,1),persistent=False)
    def forward(self,images,noisy_cxcywh,time_labels):
        assert images.ndim==4 and images.shape[-2:]==(640,640)
        assert noisy_cxcywh.shape[:2]==time_labels.shape and noisy_cxcywh.shape[0]==images.shape[0]
        x=(images.float()-self.pixel_mean)/self.pixel_std
        with torch.autocast(device_type=images.device.type,dtype=torch.bfloat16,enabled=images.is_cuda):
            pyramid=self.backbone(x)
        with torch.autocast(device_type=images.device.type,enabled=False):
            features=[pyramid[k].float() for k in ['p2','p3','p4','p5']]
            boxes=xywh_to_xyxy(noisy_cxcywh.float())*640.
            scores,bboxes=self.head(features,boxes,time_labels.float(),None)
        return dict(pred_logits=scores[-1],pred_boxes=bboxes[-1],
            aux_outputs=[dict(pred_logits=a,pred_boxes=b) for a,b in zip(scores[:-1],bboxes[:-1])])
