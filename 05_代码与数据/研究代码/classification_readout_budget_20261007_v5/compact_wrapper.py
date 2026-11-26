"""Class-only encoder context, fixed native decoder300, only300 extra bbox rows."""
import torch
from torch import nn
from semantic_branch import ClassificationReadout
from compact_candidates import native_identity,candidate_table,compact_pool

class CompactClassificationDetector(nn.Module):
    def __init__(self,base,branch=None,trace=False,readout='direct'):
        super().__init__();self.base=base;self.semantic_branch=ClassificationReadout() if branch is None else branch
        if readout not in ['direct','carrier_metadata']:raise ValueError(readout)
        self.readout=readout;self.trace=trace;self.traces=[];self.semantic_scores=[];self.requires_pool=getattr(self.semantic_branch,'requires_pool',True)
        dec=base.decoder
        if dec.query_select_method!='default' or dec.learn_query_content or dec.mgsa_groups or dec.o2m_t or dec.num_queries!=300 or dec.eval_idx!=2:raise ValueError('Native default300/3-layer configuration required.')
        self._select=dec._select_topk;self._input=dec._get_decoder_input;self._pending=self._pool=self._hidden=None
        self.extra_head_calls=0;self.extra_bbox_rows=0
        dec._select_topk=self._capture;dec._get_decoder_input=self._capture_input
        self._handles=[dec.decoder.layers[2].register_forward_hook(self._capture_hidden),dec.register_forward_hook(self._readout)]

    def _capture(self,memory,logits,anchors,k):
        result=self._select(memory,logits,anchors,k)
        if not self.requires_pool:return result
        with torch.no_grad():
            top=native_identity(anchors,result[2]);valid=torch.isfinite(anchors).all(-1)
            if not torch.equal(memory.gather(1,top[...,None].expand(-1,-1,256)),result[0]):raise ValueError('Native content identity mismatch.')
            table=candidate_table(logits.detach(),top,valid)
            extra=table[:,300:];extra_memory=memory.detach().gather(1,extra[...,None].expand(-1,-1,256))
            extra_anchors=anchors.gather(1,extra[...,None].expand(-1,-1,4))
            boxes=(self.base.decoder.enc_bbox_head(extra_memory)+extra_anchors).sigmoid()
            self.extra_head_calls+=1;self.extra_bbox_rows+=extra.numel()
        self._pending=(memory.detach(),logits.detach(),top,table,boxes)
        return result

    def _capture_input(self,*args,**kwargs):
        bundle=self._input(*args,**kwargs)
        if not self.requires_pool:return bundle
        if self._pending is None:raise RuntimeError('Missing native selection.')
        dense,L,top,table,extra_boxes=self._pending;self._pending=None
        boxes=torch.cat([bundle[1][:,-300:].sigmoid(),extra_boxes],1)
        table_logits=L.gather(1,table[...,None].expand(-1,-1,10))
        pi,aff,mask=compact_pool(boxes,table_logits,table)
        self._pool=(dense,L,top,table,boxes,pi,aff,mask)
        return bundle

    def _capture_hidden(self,module,args,out):self._hidden=out[:,-300:].detach()

    def _readout(self,module,args,out):
        if self._hidden is None or (self.requires_pool and self._pool is None):raise RuntimeError('Missing class readout features.')
        hidden=self._hidden
        if self.requires_pool:dense,L,top,table,boxes,pi,aff,mask=self._pool
        else:dense=L=top=table=boxes=pi=aff=mask=None
        self._pool=self._hidden=None
        semantic,info=self.semantic_branch(hidden,dense,out['pred_logits'],pi,mask)
        self.semantic_scores.append(semantic)
        if self.trace:
            payload={'hidden':hidden,'native_logits':out['pred_logits'],'native_boxes':out['pred_boxes'],'semantic':semantic,**info}
            if self.requires_pool:payload.update({'dense':dense,'all_enc_logits':L,'top':top,'candidate_table':table,'candidate_boxes':boxes,'pool_indices':pi,'pool_affinity':aff,'pool_mask':mask})
            self.traces.append({k:v.detach().clone() for k,v in payload.items()})
        if module.training:return out
        return {**out,'pred_logits':semantic if self.readout=='direct' else out['pred_logits'],'native_logits':out['pred_logits'],'semantic_logits':semantic}

    def _clear(self):self.semantic_scores=[];self.traces=[]
    def forward(self,*args,**kwargs):self._clear();return self.base(*args,**kwargs)
    def predict_local(self,*args,**kwargs):self._clear();return self.base.predict_local(*args,**kwargs)
    def cache_global(self,*args,**kwargs):self._clear();return self.base.cache_global(*args,**kwargs)
    def restore_native_interfaces(self):
        self.base.decoder._select_topk=self._select;self.base.decoder._get_decoder_input=self._input
        for h in self._handles:h.remove()
        self._handles=[];self._pending=self._pool=self._hidden=None
