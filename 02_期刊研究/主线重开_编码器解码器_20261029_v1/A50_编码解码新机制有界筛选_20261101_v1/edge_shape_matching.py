"""A50: native categorical-edge expected absolute GT error in training assignment.

No distribution calibration, joint-box posterior, or new Wasserstein principle is claimed.
M keeps native marginal mean/variance but replaces shape by a Gaussian; Q integrates
the actual native categorical marginal. Both use the same deterministic bounded cost.
"""
import math
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment


def physical_support(outputs, reg_max, up, reg_scale):
    from engine.edgecrafter.utils import weighting_function
    raw=outputs['pred_corners'].float()
    ref=outputs['ref_points'].float()
    prob=raw.reshape(*raw.shape[:2],4,reg_max+1).softmax(-1)
    scale=abs(float(reg_scale))
    assert scale>0
    project=weighting_function(reg_max,ref.new_tensor([up]),ref.new_tensor([reg_scale]))
    # Decode every native bin with exactly the affine map in distance2bbox.
    centers=ref[...,[0,1,0,1],None]
    widths=ref[...,[2,3,2,3],None]
    sign=ref.new_tensor([-1,-1,1,1])[None,None,:,None]
    supports=centers+sign*(.5*scale+project[None,None,None,:])*(widths/scale)
    assert torch.isfinite(supports).all() and torch.isfinite(prob).all()
    return prob,supports


def expected_error(prob,supports,gt_xyxy,mode):
    """[Q,4,K] probabilities/supports, [G,4] GT -> [Q,G] mean relative edge error."""
    assert mode in ('M','Q')
    if len(gt_xyxy)==0:
        return prob.new_empty((len(prob),0))
    wh=(gt_xyxy[:,2:]-gt_xyxy[:,:2]).clamp_min(1e-6)
    denom=wh[:,[0,1,0,1]]
    mean=(prob*supports).sum(-1)
    if mode=='M':
        var=(prob*(supports-mean[...,None]).square()).sum(-1).clamp_min(0)
        sigma=var.sqrt()
    chunks=[]
    for gt,normalizer in zip(gt_xyxy.split(64),denom.split(64)):
        delta=(mean[:,None,:]-gt[None,:,:]).abs()
        if mode=='M':
            s=sigma[:,None,:]
            z=delta/s.clamp_min(1e-12)
            error=s*math.sqrt(2/math.pi)*torch.exp(-.5*z.square())+delta*torch.erf(z/math.sqrt(2))
            error=torch.where(s>0,error,delta)
        else:
            error=(prob[:,None,:,:]*(supports[:,None,:,:]-gt[None,:,:,None]).abs()).sum(-1)
        chunks.append((error/normalizer[None,:,:]).mean(-1))
    result=torch.cat(chunks,dim=1)
    assert torch.isfinite(result).all() and (result>=0).all()
    return result


def native_cost(matcher,outputs,target):
    from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy,generalized_box_iou
    logits=outputs['pred_logits'].float()
    boxes=outputs['pred_boxes'].float()
    ids=target['labels']
    tgt=target['boxes'].float()
    if matcher.use_focal_loss:
        p=logits.sigmoid()[:,ids]
        neg=(1-matcher.alpha)*(p**matcher.gamma)*(-(1-p+1e-8).log())
        pos=matcher.alpha*((1-p)**matcher.gamma)*(-(p+1e-8).log())
        cls=pos-neg
    else:
        cls=-logits.softmax(-1)[:,ids]
    return (matcher.cost_class*cls+matcher.cost_bbox*torch.cdist(boxes,tgt,p=1)
        -matcher.cost_giou*generalized_box_iou(box_cxcywh_to_xyxy(boxes),box_cxcywh_to_xyxy(tgt)))


class EdgeShapeMatcher(nn.Module):
    def __init__(self,native_matcher,reg_max,up,reg_scale,arm):
        super().__init__()
        assert arm in ('M','Q') and not native_matcher.mask_point_sample_ratio
        self.native=native_matcher
        self.reg_max=int(reg_max);self.up=float(up);self.reg_scale=float(reg_scale);self.arm=arm
        self.calls=0;self.shape_calls=0;self.fallback_calls=0

    @torch.no_grad()
    def forward(self,outputs,targets,return_topk=False):
        from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy
        self.calls+=1
        assert not return_topk, 'No silent change to o2m interface'
        if 'pred_corners' not in outputs:
            self.fallback_calls+=1
            return self.native(outputs,targets,return_topk=return_topk)
        self.shape_calls+=1
        prob,supports=physical_support(outputs,self.reg_max,self.up,self.reg_scale)
        indices=[]
        for i,target in enumerate(targets):
            current={k:v[i] for k,v in outputs.items() if k in ('pred_logits','pred_boxes')}
            c=native_cost(self.native,current,target)
            d=expected_error(prob[i],supports[i],box_cxcywh_to_xyxy(target['boxes'].float()),self.arm)
            c=c+self.native.cost_giou*(-torch.expm1(-d))
            assert torch.isfinite(c).all()
            rows,cols=linear_sum_assignment(c.cpu().numpy())
            indices.append((torch.as_tensor(rows,dtype=torch.int64),torch.as_tensor(cols,dtype=torch.int64)))
        return {'indices':indices}


def install(criterion,model,arm):
    assert arm in ('B','M','Q')
    if arm=='B':return criterion.matcher
    decoder=model.decoder
    criterion.matcher=EdgeShapeMatcher(criterion.matcher,decoder.reg_max,decoder.up.detach().item(),
                                      decoder.reg_scale.detach().item(),arm)
    return criterion.matcher
