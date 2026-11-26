"""Registered research wrapper for the frozen native shared dual-view detector.

Single-process forward ownership only. Native source is not edited. Extra
content-adapter parameters are registered under content_adapter for a future
optimizer; no optimizer is constructed here.
"""
import torch
from torch import nn
from content_pool import ContentReassociation,prediction_pool,normal_content_only

class ReassociatedDetector(nn.Module):
    def __init__(self,base,adapter=None,trace=False):
        super().__init__();self.base=base;self.content_adapter=ContentReassociation() if adapter is None else adapter
        dec=base.decoder
        if dec.learn_query_content or dec.mgsa_groups or dec.o2m_t or dec.num_queries!=300:
            raise ValueError('Only declared native300 content-gather configuration is supported.')
        self.trace=trace;self.traces=[];self.dense_head_calls=0;self._pending=None
        self._native_select=dec._select_topk;self._native_input=dec._get_decoder_input
        dec._select_topk=self._capture_selection;dec._get_decoder_input=self._adapt_decoder_input

    def _capture_selection(self,memory,logits,anchors,k):
        with torch.no_grad():
            top=torch.topk(logits.detach().max(-1).values,k,dim=-1).indices
            boxes=torch.sigmoid(self.base.decoder.enc_bbox_head(memory.detach())+anchors)
            valid=torch.isfinite(anchors).all(-1).expand(memory.shape[0],-1)
            self.dense_head_calls+=1
        self._pending=(memory.detach(),logits.detach(),top,boxes,valid)
        return self._native_select(memory,logits,anchors,k)

    def _adapt_decoder_input(self,*args,**kwargs):
        bundle=self._native_input(*args,**kwargs);content,ref=bundle[:2]
        if self._pending is None:raise RuntimeError('Selection was not captured.')
        dense,logits,top,boxes,valid=self._pending;self._pending=None
        boxes=boxes.clone();boxes.scatter_(1,top[...,None].expand(-1,-1,4),ref[:,-300:].detach().sigmoid())
        indices,affinity,mask=prediction_pool(boxes,valid,logits.max(-1).values,top)
        adjusted,info=normal_content_only(self.content_adapter,content,dense,logits,boxes,indices,affinity,mask)
        if self.trace:
            self.traces.append({k:v.detach().clone() for k,v in {'top':top,'reference':ref,'original_content':content,'adjusted_content':adjusted,'pool_indices':indices,'pool_affinity':affinity,'pool_mask':mask,'attention':info['attention']}.items()})
        return (adjusted,ref,*bundle[2:])

    def forward(self,samples,targets=None):return self.base(samples,targets)
    def predict_local(self,*args,**kwargs):return self.base.predict_local(*args,**kwargs)
    def cache_global(self,*args,**kwargs):return self.base.cache_global(*args,**kwargs)
    def restore_native_interfaces(self):
        self.base.decoder._select_topk=self._native_select;self.base.decoder._get_decoder_input=self._native_input;self._pending=None
