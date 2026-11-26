"""Training-only budget-pair supervision: borrowed ranking idea, not a new loss family."""
import torch
import torch.nn.functional as F
from torchvision.ops import box_convert,box_iou

def _stable_extreme(values,ids,maximize=False):
    # Stable token-ID order first, then stable value sort, fixes exact ties.
    a=torch.argsort(ids,stable=True)
    b=torch.argsort(values[a],descending=maximize,stable=True)
    return a[b[0]]

def budget_pairwise_loss(criterion,outputs,targets,pointwise=False,return_metadata=False):
    aux=outputs['dense_aux'];L=aux['pred_logits'];B=aux['pred_boxes'];ids=aux['token_ids'];tops=aux['native_top']
    if L.ndim!=3 or L.shape[-1]!=10 or B.shape!=(*L.shape[:2],4) or ids.shape!=(L.shape[1],):raise ValueError('Expected native10-class dense auxiliary format.')
    losses=[];metadata=[]
    for b,t in enumerate(targets):
        n=len(t['labels']);row={'batch_index':b,'GT_n':n,'pairs':[]}
        if n==0:metadata.append(row);continue
        with torch.no_grad():
            repeated={'labels':t['labels'].repeat(3),'boxes':t['boxes'].repeat(3,1)}
            qi,ji=criterion.matcher({'pred_logits':L[b:b+1].detach(),'pred_boxes':B[b:b+1].detach()},[repeated])['indices'][0]
            gt_index=ji%n
            I=box_iou(box_convert(B[b].detach(),'cxcywh','xyxy'),box_convert(t['boxes'].detach(),'cxcywh','xyxy'))
            selected=torch.isin(ids,tops[b]);selidx=torch.nonzero(selected).flatten()
            if len(selidx)!=300:raise ValueError('Require300 distinct valid native tokens.')
            cls=L[b].detach().argmax(-1);scores=L[b].detach().max(-1).values
            supported=(I[selidx]>=.5)&(cls[selidx,None]==t['labels'][None,:])
            protected=supported.any(-1);represented=supported.any(0);bad=selidx[~protected]
            row['protected_selected_n']=int(protected.sum());row['represented_GT_n']=int(represented.sum());row['bad_selected_n']=len(bad)
            if len(bad)==0:metadata.append(row);continue
            neg=bad[_stable_extreme(scores[bad],ids[bad])];nc=int(cls[neg])
            for j in range(n):
                if bool(represented[j]):continue
                pos=qi[(gt_index==j)&(~selected[qi])&(I[qi,j]>=.5)]
                if len(pos)==0:continue
                p=pos[_stable_extreme(I[pos,j],ids[pos],maximize=True)];truth=int(t['labels'][j]);weight=I[p,j].detach()
                row['pairs'].append({'positive_dense_index':int(p),'positive_token':int(ids[p]),'negative_dense_index':int(neg),'negative_token':int(ids[neg]),'truth':truth,'negative_class':nc,'GT_local_index':j,'IoU':float(weight),'negative_has_correct_class_geometry_support':False})
        for pair in row['pairs']:
            p=pair['positive_dense_index'];q=pair['negative_dense_index'];c=pair['truth'];k=pair['negative_class']
            ps=L[b,p,c];ns=L[b,q,k]
            value=F.softplus(-ps)+F.softplus(ns) if pointwise else F.softplus(ns-ps)
            losses.append(value*pair['IoU'])
        metadata.append(row)
    count=torch.tensor([len(losses)],dtype=torch.float32,device=L.device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(count);count/=torch.distributed.get_world_size()
    normalizer=float(count.clamp_min(1));loss=.25*torch.stack(losses).sum()/normalizer if losses else L.sum()*0.
    key='loss_budget_pointwise' if pointwise else 'loss_budget_rank'
    result={key:loss}
    return (result,{'views':metadata,'normalizer':normalizer,'pair_n':len(losses),'new_trainable_parameters':0,'training_only':True}) if return_metadata else result

def paired_budget_loss(criterion,outputs,targets,pointwise=False):
    a=budget_pairwise_loss(criterion,outputs['global'],targets['global'],pointwise)
    b=budget_pairwise_loss(criterion,outputs['local'],targets['local'],pointwise)
    return {k:.5*(a[k]+b[k]) for k in a}
