"""Explicit borrowed/simple controls for protocol63, no optimizer or training loop."""
import torch
from torch import nn

class PoolFeatureControl(nn.Module):
    def __init__(self,mode):
        super().__init__()
        if mode not in ('maxlogit','mean'):raise ValueError(mode)
        self.mode=mode
    def forward(self,content,dense,logits,boxes,indices,affinity,mask):
        batch=torch.arange(indices.shape[0],device=indices.device)[:,None,None]
        pooled=dense[batch,indices.clamp_min(0)]
        if self.mode=='maxlogit':
            scores=logits[batch,indices.clamp_min(0)].max(-1).values.masked_fill(~mask,-torch.inf)
            chosen=scores.argmax(-1);attention=torch.nn.functional.one_hot(chosen,indices.shape[-1]).to(content.dtype)
        else:attention=mask.to(content.dtype)/mask.sum(-1,keepdim=True)
        result=(attention[...,None]*pooled).sum(-2)
        return result,{'attention':attention,'residual':result-content}

class DenseSupervisionDetector(nn.Module):
    """Training-only same-head dense predictions; native normal/DN outputs unchanged.

    Captures live encoder output features, not detached candidate-pool features.
    Dense box values at selected positions use actual native300 head results.
    No new trainable parameters and no extra decoder queries.
    """
    def __init__(self,base):
        super().__init__();self.base=base;self._native=base.decoder._select_topk;self._captured=[];self.dense_head_calls=0
        if base.decoder.num_queries!=300 or base.decoder.mgsa_groups or base.decoder.o2m_t or base.decoder.query_select_method!='default':raise ValueError('Only native default300 supports this control.')
        base.decoder._select_topk=self._capture
    def _capture(self,memory,logits,anchors,k):
        result=self._native(memory,logits,anchors,k)
        if self.base.decoder.training:
            top=logits.detach().max(-1).values.topk(k,dim=-1).indices
            boxes=(self.base.decoder.enc_bbox_head(memory)+anchors).sigmoid();self.dense_head_calls+=1
            valid=torch.isfinite(anchors).all(-1).expand(memory.shape[0],-1)
            self._captured.append((logits,boxes,valid,top))
        return result
    def forward(self,samples,targets=None):
        self._captured=[]
        try:
            out=self.base(samples,targets)
            if self.base.decoder.training:
                if len(self._captured)!=2:raise RuntimeError('Expected global and local decoder calls.')
                for view,(logits,boxes,valid,top) in zip(('global','local'),self._captured):
                    native=out[view]['enc_aux_outputs'][0]['pred_boxes']
                    boxes=boxes.scatter(1,top[...,None].expand(-1,-1,4),native)
                    if not torch.equal(valid,valid[:1].expand_as(valid)):raise ValueError('Different valid token masks within batch.')
                    token_ids=torch.nonzero(valid[0]).flatten()
                    out[view]['dense_aux']={'pred_logits':logits[:,token_ids],'pred_boxes':boxes[:,token_ids],'token_ids':token_ids,'native_top':top}
            return out
        finally:self._captured=[]
    def cache_global(self,*args,**kwargs):return self.base.cache_global(*args,**kwargs)
    def predict_local(self,*args,**kwargs):return self.base.predict_local(*args,**kwargs)
    def restore_native_interfaces(self):self.base.decoder._select_topk=self._native;self._captured=[]

def dense_auxiliary_loss(criterion,outputs,targets,repeat=3,scale=.25):
    """O2M Hungarian with repeated GT, same native MAL+L1+GIoU weights.

    This adapts a dense-supervision idea. It does not reproduce DDQ/Co-DETR/DHQ.
    Geometry/classification are both supervised; targets are training-only.
    """
    if repeat!=3 or scale!=.25:raise ValueError('Protocol63 fixed repeat3,scale.25.')
    repeated=[{'labels':t['labels'].repeat(repeat),'boxes':t['boxes'].repeat(repeat,1)} for t in targets]
    aux=outputs['dense_aux'];indices=criterion.matcher(aux,repeated,epoch=0,num_queries_list=None)['indices']
    normalizer=torch.tensor([sum(len(t['labels']) for t in repeated)],dtype=torch.float32,device=aux['pred_logits'].device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(normalizer);normalizer/=torch.distributed.get_world_size()
    normalizer=float(normalizer.clamp_min(1))
    losses=criterion.loss_labels_mal(aux,repeated,indices,normalizer)
    losses.update(criterion.loss_boxes(aux,repeated,indices,normalizer))
    if set(losses)!=set(('loss_mal','loss_bbox','loss_giou')):raise ValueError('Expected native MAL+L1+GIoU.')
    return {k+'_dense_o2m':v*criterion.weight_dict[k]*scale for k,v in losses.items()}

def paired_dense_auxiliary_loss(criterion,outputs,targets):
    a=dense_auxiliary_loss(criterion,outputs['global'],targets['global'])
    b=dense_auxiliary_loss(criterion,outputs['local'],targets['local'])
    return {k:.5*(a[k]+b[k]) for k in a}
