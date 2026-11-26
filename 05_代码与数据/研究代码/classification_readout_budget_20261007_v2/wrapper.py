"""Native training outputs untouched; single-owner late readout capture."""
import sys
from pathlib import Path
import torch
from torch import nn
from semantic_branch import ClassificationReadout,preserve_score_multiset
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'geometry_content_reassociation_20261007_v1'))
from content_pool import prediction_pool

class ClassificationDetector(nn.Module):
    def __init__(self,base,branch=None,trace=False):
        super().__init__();self.base=base;self.semantic_branch=ClassificationReadout() if branch is None else branch
        dec=base.decoder
        if dec.learn_query_content or dec.mgsa_groups or dec.o2m_t or dec.num_queries!=300 or dec.eval_idx!=2:raise ValueError('Requires fixed native 300/3-layer configuration.')
        self._select=dec._select_topk;self._input=dec._get_decoder_input;self._pending=None;self._pool=None;self._hidden=None
        self.trace=trace;self.traces=[];self.semantic_scores=[];self.dense_head_calls=0
        dec._select_topk=self._capture;dec._get_decoder_input=self._capture_input
        self._handles=[dec.decoder.layers[2].register_forward_hook(self._capture_hidden),dec.register_forward_hook(self._readout)]

    def _capture(self,memory,logits,anchors,k):
        with torch.no_grad():
            top=logits.max(-1).values.topk(k,dim=-1).indices
            boxes=(self.base.decoder.enc_bbox_head(memory.detach())+anchors).sigmoid()
            valid=torch.isfinite(anchors).all(-1).expand(len(memory),-1);self.dense_head_calls+=1
        self._pending=(memory.detach(),logits.detach(),top,boxes,valid)
        return self._select(memory,logits,anchors,k)

    def _capture_input(self,*args,**kwargs):
        bundle=self._input(*args,**kwargs)
        if self._pending is None:raise RuntimeError('Native selection absent.')
        dense,L,top,boxes,valid=self._pending;self._pending=None
        boxes=boxes.clone();boxes.scatter_(1,top[...,None].expand(-1,-1,4),bundle[1][:,-300:].sigmoid())
        pi,affinity,mask=prediction_pool(boxes,valid,L.max(-1).values,top)
        self._pool=(dense,top,boxes,pi,affinity,mask)
        return bundle

    def _capture_hidden(self,module,args,out):self._hidden=out[:,-300:].detach()

    def _readout(self,module,args,out):
        if self._pool is None or self._hidden is None:raise RuntimeError('Incomplete native readout.')
        dense,top,boxes,pi,affinity,mask=self._pool;hidden=self._hidden;self._pool=self._hidden=None
        semantic,info=self.semantic_branch(hidden,dense,out['pred_logits'],pi,mask)
        self.semantic_scores.append(semantic)
        if self.trace:
            self.traces.append({k:v.detach().clone() for k,v in {'hidden':hidden,'dense':dense,'native_logits':out['pred_logits'],
                'native_boxes':out['pred_boxes'],'top':top,'all_enc_boxes':boxes,'pool_indices':pi,'pool_affinity':affinity,'pool_mask':mask,'semantic':semantic,**info}.items()})
        if module.training:return out
        return {**out,'pred_logits':preserve_score_multiset(out['pred_logits'],semantic)}

    def _clear(self):self.semantic_scores=[];self.traces=[]
    def forward(self,*args,**kwargs):self._clear();return self.base(*args,**kwargs)
    def predict_local(self,*args,**kwargs):self._clear();return self.base.predict_local(*args,**kwargs)
    def cache_global(self,*args,**kwargs):self._clear();return self.base.cache_global(*args,**kwargs)
    def restore_native_interfaces(self):
        self.base.decoder._select_topk=self._select;self.base.decoder._get_decoder_input=self._input
        for h in self._handles:h.remove()
        self._handles=[];self._pending=self._pool=self._hidden=None
