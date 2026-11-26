"""Passive actual-training snapshots plus same-output native matching control."""
import json
from pathlib import Path
import types
import numpy as np
import torch


class LiveMatchingAudit:
    def __init__(self,model,criterion,binding,output):
        self.criterion,self.binding,self.output=criterion,binding,Path(output)
        self.epoch=-1;self.attempt=0;self.selected=False;self.saved=None
        self.original_matcher_forward=criterion.matcher.forward
        self.original_go=criterion._get_go_indices
        self.native_indices=[];self.candidate_indices=[]
        self.handles=[model.register_forward_pre_hook(self.model_pre,with_kwargs=True),
            criterion.register_forward_pre_hook(self.criterion_pre,with_kwargs=True),
            criterion.register_forward_hook(self.criterion_post,with_kwargs=True)]
        audit=self
        def forward(module,*args,**kwargs):
            result=audit.original_matcher_forward(*args,**kwargs)
            if audit.selected:
                with torch.no_grad():native=binding.original_matcher_forward(*args,**kwargs)
                audit.native_indices.append(native['indices']);audit.candidate_indices.append(result['indices'])
            return result
        criterion.matcher.forward=types.MethodType(forward,criterion.matcher)

    def enter_epoch(self,epoch):self.epoch=epoch;self.attempt=0

    def model_pre(self,module,args,kwargs):
        if not module.training:return
        self.attempt+=1
        self.selected=self.epoch in (0,4,8,11) and self.attempt in (1,101,501,1001,1501)
        if self.selected:self.shape=tuple(args[0].shape)

    def criterion_pre(self,module,args,kwargs):
        if not self.selected:return
        self.native_indices=[];self.candidate_indices=[];self.saved=(args[0],args[1])

    def criterion_post(self,module,args,kwargs,result):
        if not self.selected:return
        try:
            assert len(self.native_indices)==len(self.candidate_indices)==5
            outputs,targets=self.saved
            refs=outputs['pre_outputs']['pred_boxes'].detach().float()
            bound=(2*outputs['up'].detach().float().abs()*outputs['reg_scale'].detach().float().abs()).reshape(())
            scale=outputs['reg_scale'].detach().float().abs().reshape(())
            original=self.original_go(self.native_indices[0],self.native_indices[1:])
            candidate=self.original_go(self.candidate_indices[0],self.candidate_indices[1:])
            rows=[];payload={}
            for slot,target in enumerate(targets):
                gt=target['boxes'].detach().float();h,w=self.shape[-2:]
                short=(gt[:,2:]*gt.new_tensor([w,h])).amin(-1)
                for arm,pairs in [('native_cost_control',original),('active_candidate',candidate)]:
                    q,g=pairs[slot];rr=refs[slot,q];gg=gt[g]
                    lo,hi=gg[:,:2]-gg[:,2:]/2,gg[:,:2]+gg[:,2:]/2
                    dist=torch.cat(((rr[:,:2]-lo)/(rr[:,2:]/scale+1e-16)-scale/2,
                        (hi-rr[:,:2])/(rr[:,2:]/scale+1e-16)-scale/2),-1)
                    ok=dist.abs().amax(-1)<=bound
                    for label,mask in [('under4',short[g]<4),('4to8',(short[g]>=4)&(short[g]<8)),
                        ('8to16',(short[g]>=8)&(short[g]<16)),('ge16',short[g]>=16)]:
                        rows.append(dict(slot=slot,arm=arm,band=label,pairs=int(mask.sum()),outside=int((~ok[mask]).sum()),
                            GT_rows=int(torch.unique(g[mask.cpu()]).numel())))
                    payload[f'{arm}.{slot}.query']=q.cpu().numpy();payload[f'{arm}.{slot}.GT']=g.cpu().numpy()
                payload[f'target.{slot}.boxes']=gt.cpu().numpy();payload[f'target.{slot}.labels']=target['labels'].detach().cpu().numpy()
                payload[f'pre.{slot}']=refs[slot].cpu().numpy()
                payload[f'final.{slot}']=outputs['pred_boxes'][slot].detach().float().cpu().numpy()
                payload[f'logits.{slot}']=outputs['pred_logits'][slot].detach().float().cpu().numpy()
            folder=self.output/'matching_snapshots';folder.mkdir(exist_ok=True)
            stem=f'epoch{self.epoch:02d}_attempt{self.attempt:04d}'
            assert not (folder/(stem+'.npz')).exists()
            np.savez_compressed(folder/(stem+'.npz'),**payload)
            with (folder/(stem+'.json')).open('x') as f:json.dump(dict(epoch_zero_based=self.epoch,attempt=self.attempt,
                shape=self.shape,rows=rows,scope='Actual candidate training output before this optimizer update; native matching on the same tensors is a responsibility counterfactual, not a BASE forward or actual BASE history.'),f)
        finally:
            self.saved=None;self.selected=False;self.native_indices=[];self.candidate_indices=[]

    def close(self):
        for h in self.handles:h.remove()
        self.criterion.matcher.forward=self.original_matcher_forward
