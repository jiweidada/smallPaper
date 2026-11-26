"""Counterfactual losses on identical frozen predictions. No optimizer/model graph."""
import copy,time,types
from collections import Counter
import numpy as np
import torch
from probe_runtime import build_criterion
from engine.deim.box_ops import box_cxcywh_to_xyxy,box_iou,generalized_box_iou
from rules import frequencies,solve_cover,anchor_fill

def branches(outputs):
    result=[('final',outputs)]
    result += [(f'aux_{i}',v) for i,v in enumerate(outputs.get('aux_outputs',[]))]
    if 'pre_outputs' in outputs:result.append(('pre',outputs['pre_outputs']))
    result += [(f'enc_{i}',v) for i,v in enumerate(outputs.get('enc_aux_outputs',[]))]
    return result
def asmap(pair):return {int(q):int(g) for q,g in zip(pair[0].tolist(),pair[1].tolist())}
def clone_structure(x,corner_grad=False,key=''):
    if isinstance(x,torch.Tensor):
        v=x.detach().clone()
        if corner_grad and key=='pred_corners':v.requires_grad_(True)
        return v
    if isinstance(x,dict):return {k:clone_structure(v,corner_grad,k) for k,v in x.items()}
    if isinstance(x,list):return [clone_structure(v,corner_grad,key) for v in x]
    if isinstance(x,tuple):return tuple(clone_structure(v,corner_grad,key) for v in x)
    return x
def map_pair(mapping,original):
    q=original[0].clone();return (q,torch.tensor([mapping[int(i)] for i in q],dtype=torch.int64,device=q.device))
def quality(output,target,mapping,own):
    if not mapping:return {'pairs':0,'GT':len(target['labels']),'covered':0,'mean_iou':None,'q25_iou':None,'loss_bbox':0.,'loss_giou':0.}
    qs=sorted(mapping);gs=[mapping[q] for q in qs];pred=output['pred_boxes'][qs];gt=target['boxes'][gs]
    iou=box_iou(box_cxcywh_to_xyxy(pred),box_cxcywh_to_xyxy(gt))[0].diag().numpy()
    giou=generalized_box_iou(box_cxcywh_to_xyxy(pred),box_cxcywh_to_xyxy(gt)).diag().numpy()
    l1=(pred-gt).abs().sum(-1).numpy();counts=Counter(gs)
    gt_best={g:max(float(iou[i]) for i,v in enumerate(gs) if v==g) for g in counts}
    return {'pairs':len(qs),'GT':len(target['labels']),'covered':len(counts),'mean_iou':float(iou.mean()),'q25_iou':float(np.quantile(iou,.25)),'loss_bbox':float(l1.mean()),'loss_giou':float((1-giou).mean()),'pair_query':qs,'pair_GT':gs,'pair_iou':iou.tolist(),'pair_l1':l1.tolist(),'pair_giou':giou.tolist(),'GT_best_pair_iou':gt_best,'GT_repetitions':[counts.get(g,0) for g in range(len(target['labels']))],'own_positive_identity_disagreements':sum(q in own and own[q]!=mapping[q] for q in qs),'own_positive_queries':sum(q in own for q in qs),'classification_background_queries':sum(q not in own for q in qs)}
def slice_outputs(outputs,bi):
    b=outputs['pred_boxes'].shape[0]
    def one(v):
        if isinstance(v,torch.Tensor):return v[bi:bi+1].clone() if v.ndim and v.shape[0]==b else v.clone()
        if isinstance(v,dict):return {k:one(x) for k,x in v.items()}
        if isinstance(v,list):return [one(x) for x in v]
        return v
    return {k:one(v) for k,v in outputs.items() if not k.startswith('dn_')}

def analyze(outputs,targets,y,check_grad=False):
    assert outputs['pred_boxes'].device.type=='cpu'
    native=build_criterion(y);bs=branches(outputs)
    matches=[native.matcher(out,targets,epoch=24)['indices'] for name,out in bs]
    u0=native._get_go_indices(matches[0],matches[1:])
    maps={name:[] for name in ['native_union','coverage_constrained','last_anchor_fill']}
    records=[]
    for bi,t in enumerate(targets):
        n=len(t['labels']);v=frequencies([list(asmap(m[bi]).items()) for m in matches]);original=asmap(u0[bi]);last=asmap(matches[0][bi])
        full=any(len(asmap(m[bi]))==n and len(set(asmap(m[bi]).values()))==n for m in matches)
        started=time.perf_counter()
        if full:fixed,meta=solve_cover(v,original,n,outputs['pred_boxes'].shape[1])
        else:fixed,meta=dict(original),{'fallback':True,'reason':'no_complete_branch'}
        solver_seconds=time.perf_counter()-started
        anchor,anchor_meta=anchor_fill(original,last,v,n,outputs['pred_boxes'].shape[1])
        allmap={'native_union':original,'coverage_constrained':fixed,'last_anchor_fill':anchor}
        v0=sum(v[q,g] for q,g in original.items());vf=sum(v[q,g] for q,g in fixed.items())
        r={'input_within_batch':bi,'GT':n,'K':len(original),'native_covered':len(set(original.values())),'candidate_covered':len(set(fixed.values())),'capacity_eligible':bool(n and n<=300 and full),'candidate_differs_anchor':fixed!=anchor,'native_missing_GT':sorted(set(range(n))-set(original.values())),'votes':[[q,g,c] for (q,g),c in sorted(v.items())],'vote_loss_ratio':(v0-vf)/v0 if v0 else 0.,'solver_seconds':solver_seconds,'solver':meta,'anchor':anchor_meta,'branches':{}}
        for name,mapping in allmap.items():
            assert set(mapping)==set(original);maps[name].append(map_pair(mapping,u0[bi]))
        for j,(name,out) in enumerate(bs):
            single={k:val[bi] for k,val in out.items() if k in ['pred_boxes','pred_logits']}
            own=asmap(matches[j][bi]);q={rule:quality(single,t,mapping,own) for rule,mapping in allmap.items()}
            q['own_layer_hungarian']=quality(single,t,own,own);q['last_hungarian']=quality(single,t,last,own)
            r['branches'][name]=q
        records.append(r)
    # Native and candidate criteria have independent caches. Only ordinary direct
    # bbox/GIoU/FGL are replaced. MAL, DN and ALL DDF are explicitly native-U0.
    x=clone_structure(outputs,corner_grad=check_grad);nt=clone_structure(targets)
    base=build_criterion(y)(clone_structure_dict_keep_tensors(x),nt,epoch=24)
    result={};ddfkeys=[k for k in base if 'loss_ddf' in k]
    for rule,indices in maps.items():
        c=build_criterion(y)
        c._get_go_indices=types.MethodType(lambda self,indices0,aux:indices,c)
        # Capture the fixed list with a separate object, rather than loop binding.
        fixed_indices=indices
        c._get_go_indices=lambda indices0,aux,fixed_indices=fixed_indices:fixed_indices
        changed=c(clone_structure_dict_keep_tensors(x),clone_structure(targets),epoch=24)
        isolated=dict(base)
        direct=[k for k in changed if ('loss_bbox' in k or 'loss_giou' in k or 'loss_fgl' in k) and 'dn' not in k]
        for k in direct:isolated[k]=changed[k]
        assert all(torch.isfinite(v).all() for v in isolated.values())
        ddf_identical=all(torch.equal(isolated[k],base[k]) for k in ddfkeys)
        grad_error=0.
        if check_grad and ddfkeys:
            leaves=[out['pred_corners'] for name,out in branches(x) if 'pred_corners' in out and out['pred_corners'].requires_grad]
            aa=torch.autograd.grad(sum(base[k] for k in ddfkeys),leaves,retain_graph=True,allow_unused=True)
            bb=torch.autograd.grad(sum(isolated[k] for k in ddfkeys),leaves,retain_graph=True,allow_unused=True)
            grad_error=max([float((a-b).abs().max()) for a,b in zip(aa,bb) if a is not None and b is not None] or [0.])
        assert ddf_identical and grad_error==0.
        result[rule]={'losses':{k:float(v.detach()) for k,v in isolated.items()},'unisolated_DDF_changes':{k:float((changed[k]-base[k]).detach()) for k in ddfkeys},'DDF_native_components_reused':True,'DDF_value_identical':ddf_identical,'DDF_gradient_max_abs_difference':grad_error,'gradient_checked':check_grad,'ordinary_direct_loss_keys':direct}
    return {'inputs':records,'batch_losses':result,'branch_names':[name for name,out in bs],'branch_matches':[{name:[m[0].tolist(),m[1].tolist()] for (name,out),layer in zip(bs,matches) for m in [layer[bi]]} for bi in range(len(targets))]}

def clone_structure_dict_keep_tensors(x):
    """Independent containers preserve the same frozen prediction leaves for checks."""
    if isinstance(x,dict):return {k:clone_structure_dict_keep_tensors(v) for k,v in x.items()}
    if isinstance(x,list):return [clone_structure_dict_keep_tensors(v) for v in x]
    if isinstance(x,tuple):return tuple(clone_structure_dict_keep_tensors(v) for v in x)
    return x
