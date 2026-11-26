"""A54: same pointwise scale head; Q projects its readout to a shared affine field.

M is an ordinary dense auxiliary scale predictor. Q has exactly the same learned
parameters and uses a regularized, fixed rank-three scene readout instead.
This does not invent least squares, scale priors, or monocular GSD estimation.
"""
import math,types
import torch
from torch import nn
from torch.nn import functional as F

HIDDEN=32
RIDGE=.05
AUX_WEIGHT=1.0
NEW_INIT_SEED=20261103

class SceneScalePrior(nn.Module):
    def __init__(self,channels,arm,class_offsets,initial_log_scale):
        super().__init__();assert arm in ['M','Q'];self.arm=arm
        self.map=nn.Sequential(nn.Conv2d(channels,HIDDEN,1),nn.GELU(),nn.Conv2d(HIDDEN,1,1))
        nn.init.zeros_(self.map[-1].weight);nn.init.constant_(self.map[-1].bias,float(initial_log_scale))
        self.gain=nn.Parameter(torch.zeros(()))
        self.register_buffer('class_offsets',torch.as_tensor(class_offsets,dtype=torch.float32).clone())
        self.ridge=RIDGE;self.latest={}
    @staticmethod
    def basis(fmap):
        h,w=fmap.shape[-2:]
        x=(torch.arange(w,device=fmap.device,dtype=torch.float32)+.5)*2/w-1
        y=(torch.arange(h,device=fmap.device,dtype=torch.float32)+.5)*2/h-1
        return x,y
    def coefficients(self,fmap):
        f=fmap.float();x,y=self.basis(f)
        a0=f.mean((-2,-1)).squeeze(1)
        ax=(f*x[None,None,None,:]).mean((-2,-1)).squeeze(1)/(x.square().mean()+self.ridge)
        ay=(f*y[None,None,:,None]).mean((-2,-1)).squeeze(1)/(y.square().mean()+self.ridge)
        return torch.stack([a0,ax,ay],-1)
    def read(self,fmap,points):
        if self.arm=='Q':
            a=self.coefficients(fmap);p=points.float()*2-1
            return a[:,None,0]+a[:,None,1]*p[...,0]+a[:,None,2]*p[...,1]
        grid=points.float().mul(2).sub(1).unsqueeze(2)
        return F.grid_sample(fmap.float(),grid,mode='bilinear',padding_mode='border',align_corners=False).squeeze(1).squeeze(-1)
    def correct(self,fmap,ordinary_ref_unact,class_logits):
        raw=ordinary_ref_unact.float();ref=raw.sigmoid();wh=ref[...,2:].clamp(1e-6,1-1e-6)
        size_log=wh.log().mean(-1)
        class_log=(class_logits.detach().float().softmax(-1)*self.class_offsets).sum(-1)
        scale_log=self.read(fmap,ref[...,:2])+class_log
        desired=(scale_log-size_log).clamp(-math.log(2),math.log(2))
        shift=self.gain.tanh()*desired
        new_wh=(wh*shift.exp().unsqueeze(-1)).clamp(1e-6,1-1e-6)
        # This residual is exactly zero at zero gain; native logits remain bit-identical.
        logit_delta=shift.unsqueeze(-1)+torch.log1p(-wh)-torch.log1p(-new_wh)
        result=torch.cat([raw[...,:2],raw[...,2:]+logit_delta],-1)
        self.latest={'gain':float(self.gain.detach().tanh()),'scale_map_std':float(fmap.detach().float().std()),
            'mean_abs_log_size_shift':float(shift.detach().abs().mean()),
            'width_or_height_clip_fraction':float(((wh*shift.exp().unsqueeze(-1))>=1-1e-6).detach().float().mean())}
        return result.to(ordinary_ref_unact.dtype)
    def auxiliary_loss(self,fmap,targets):
        loss=fmap.float().sum()*0.;count=0
        for i,t in enumerate(targets):
            boxes=t['boxes'].float();labels=t['labels'].long()
            if not len(boxes):continue
            assert boxes.shape[-1]==4 and labels.min()>=0 and labels.max()<10
            predicted=self.read(fmap[i:i+1],boxes[None,:,:2]).squeeze(0)+self.class_offsets[labels]
            target_log=boxes[:,2:].clamp_min(1e-6).log().mean(-1)
            loss=loss+F.smooth_l1_loss(predicted,target_log,reduction='sum',beta=.1);count+=len(boxes)
        return loss/max(count,1)

def install(model,criterion,arm,class_offsets,initial_log_scale):
    assert arm in ['M','Q'];decoder=model.decoder
    assert not hasattr(decoder,'scene_scale_prior')
    previous=torch.random.get_rng_state();torch.manual_seed(NEW_INIT_SEED)
    module=SceneScalePrior(decoder.hidden_dim,arm,class_offsets,initial_log_scale)
    torch.random.set_rng_state(previous);decoder.add_module('scene_scale_prior',module)
    original_forward=decoder.forward;original_select=decoder._select_topk;original_input=decoder._get_decoder_input
    def select(this,memory,outputs_logits,outputs_anchors_unact,topk):
        selected=original_select(memory,outputs_logits,outputs_anchors_unact,topk)
        # Evaluation's native _select_topk intentionally omits logits, but the same linear head is available.
        this._a54_class_logits=(selected[1] if selected[1] is not None else this.enc_score_head(selected[0])).detach()
        return selected
    def inputs(this,memory,spatial_shapes,denoising_logits=None,denoising_bbox_unact=None):
        content,refs,enc_boxes,enc_logits=original_input(memory,spatial_shapes,denoising_logits,denoising_bbox_unact)
        n=this.num_queries;assert refs.shape[1]>=n and this._a54_class_logits.shape[1]==n
        corrected=module.correct(this._a54_map,refs[:,-n:],this._a54_class_logits)
        # DN references and native encoder auxiliary predictions remain unchanged.
        refs=torch.cat([refs[:,:-n],corrected],dim=1)
        return content,refs,enc_boxes,enc_logits
    def forward(this,feats,targets=None,spatial_feat=None):
        assert feats[0].shape[1]==module.map[0].in_channels
        this._a54_map=module.map(feats[0]);this._a54_class_logits=None
        try:
            out=original_forward(feats,targets,spatial_feat)
            if this.training:
                assert targets is not None
                with torch.autocast(device_type=feats[0].device.type,enabled=False):out['a54_scale_loss']=module.auxiliary_loss(this._a54_map,targets)
            return out
        finally:
            this._a54_map=None;this._a54_class_logits=None
    decoder._select_topk=types.MethodType(select,decoder)
    decoder._get_decoder_input=types.MethodType(inputs,decoder)
    decoder.forward=types.MethodType(forward,decoder)
    original_criterion=criterion.forward
    def loss_forward(this,outputs,targets,**kwargs):
        clean={k:v for k,v in outputs.items() if k!='a54_scale_loss'}
        losses=original_criterion(clean,targets,**kwargs)
        if 'a54_scale_loss' in outputs:losses['loss_a54_scene_scale']=AUX_WEIGHT*outputs['a54_scale_loss']
        return losses
    criterion.forward=types.MethodType(loss_forward,criterion)
    return module
