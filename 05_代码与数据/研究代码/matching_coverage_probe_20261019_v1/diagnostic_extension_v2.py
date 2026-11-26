"""Pre-forward protocol addendum: DN, vote ties, overflow, split losses.

Append-only instrumentation; frozen native model/criterion/rules are unchanged.
No model forward and no optimizer in this module.
"""
from collections import defaultdict
import torch
from evaluate_rules import analyze as original_analyze, branches, asmap
from probe_runtime import build_criterion
from engine.deim.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from engine.deim.dfine_utils import bbox2distance


def split_losses(part):
    return {term:sum(part['losses'][k] for k in part['ordinary_direct_loss_keys']
                     if k.startswith('loss_'+term)) for term in ['bbox','giou','fgl']}


def tie_diagnostics(votes, original, n):
    byq=defaultdict(dict)
    for q,g,f in votes:byq[q][g]=f
    tied={};stable={}
    for q,gs in byq.items():
        maximum=max(gs.values());best=sorted(g for g,f in gs.items() if f==maximum)
        stable[q]=best[0]
        if len(best)>1:tied[q]={'GT':best,'votes':maximum,'production_U0_GT':original[q]}
    return {'top_vote_tied_queries':len(tied),'query_count':len(byq),
            'top_vote_ties':tied,'deterministic_min_GT_union':[[q,g] for q,g in sorted(stable.items())],
            'native_differs_min_GT_queries':sum(original[q]!=g for q,g in stable.items()),
            'min_GT_union_missing_GT':sorted(set(range(n))-set(stable.values())),
            'tie_rule':'U0 left untouched; record all top-frequency ties and smallest-GT sensitivity',
            'native_indices_device':'CPU: native matcher returns CPU int64 indices even for CUDA costs'}


def dn_diagnostics(outputs,targets,y,result):
    """Per-GT additive DN geometry, checked against full native criterion totals."""
    if 'dn_outputs' not in outputs:
        return {'present':False,'reason':'No DN outputs (synthetic preflight or DN disabled)'}
    crit=build_criterion(y);assert crit.boxes_weight_format is None
    indices=crit.get_cdn_matched_indices(outputs['dn_meta'],targets)
    groups=int(outputs['dn_meta']['dn_num_group'])
    denom=max(sum(len(t['labels']) for t in targets),1)*groups
    pair_b=torch.cat([torch.full_like(q,bi) for bi,(q,g) in enumerate(indices)])
    pair_q=torch.cat([q for q,g in indices]);pair_g=torch.cat([g for q,g in indices])
    gt=torch.cat([t['boxes'][g] for t,(q,g) in zip(targets,indices)])
    layers=[(f'dn_{i}',v) for i,v in enumerate(outputs['dn_outputs'])]
    if 'dn_pre_outputs' in outputs:layers.append(('dn_pre',outputs['dn_pre_outputs']))
    cache=None;records={};weights=crit.weight_dict
    for name,out in layers:
        pred=out['pred_boxes'][pair_b,pair_q]
        iou=box_iou(box_cxcywh_to_xyxy(pred),box_cxcywh_to_xyxy(gt))[0].diag()
        giou=generalized_box_iou(box_cxcywh_to_xyxy(pred),box_cxcywh_to_xyxy(gt)).diag()
        l1=(pred-gt).abs().sum(-1)
        raw={'bbox':l1,'giou':1-giou}
        if 'pred_corners' in out:
            if cache is None:
                cache=bbox2distance(out['ref_points'][pair_b,pair_q].detach(),
                      box_cxcywh_to_xyxy(gt),crit.reg_max,outputs['reg_scale'],outputs['up'])
            label,wr,wl=cache
            raw['fgl']=crit.unimodal_distribution_focal_loss(
                out['pred_corners'][pair_b,pair_q].reshape(-1,crit.reg_max+1),
                label,wr,wl,weight=iou.repeat_interleave(4),reduction='none').reshape(-1,4).sum(-1)
        weighted={term:v*weights['loss_'+term]/denom for term,v in raw.items()}
        references=result['batch_losses']['native_union']['losses'];checks={}
        for term,v in weighted.items():
            key='loss_'+term+'_'+name
            reference=references[key];difference=abs(float(v.sum())-reference)
            assert difference<=1e-5*max(1.,abs(reference)),(key,difference,reference)
            checks[key]={'native_total':reference,'sum_per_pair':float(v.sum()),'abs_error':difference}
        images=[]
        for bi,t in enumerate(targets):
            missing=set(result['inputs'][bi]['native_missing_GT']);pergt=[]
            for g in range(len(t['labels'])):
                mask=(pair_b==bi)&(pair_g==g);vals=iou[mask]
                pergt.append({'GT':g,'omitted_by_U0':g in missing,'positive_DN_slots':int(mask.sum()),
                    'mean_iou':float(vals.mean()) if len(vals) else None,
                    'q25_iou':float(torch.quantile(vals,.25)) if len(vals) else None,
                    'raw_pair_mean':{term:float(v[mask].mean()) if mask.any() else None for term,v in raw.items()},
                    'weighted_batch_contribution':{term:float(v[mask].sum()) for term,v in weighted.items()}})
            images.append({'input_within_batch':bi,'GT':pergt})
        records[name]={'full_native_sum_checks':checks,'images':images}
    return {'present':True,'dn_num_group':groups,'native_DN_normalizer':denom,
            'per_GT_contribution_definition':'sum native-weighted positive-DN pair losses / full-batch (max(total_GT,1)*DN_groups); not a new normalization',
            'FGL_cache':'first DN layer references, reused across DN layers exactly as native criterion',
            'DDF_excluded_from_geometry_decomposition':True,'branches':records}


def analyze(outputs,targets,y,check_grad=False):
    result=original_analyze(outputs,targets,y,check_grad=check_grad)
    qcount=int(outputs['pred_boxes'].shape[1])
    for r in result['inputs']:
        mapping=dict(zip(r['branches']['final']['native_union'].get('pair_query',[]),
                         r['branches']['final']['native_union'].get('pair_GT',[])))
        r['tie_sensitivity']=tie_diagnostics(r['votes'],mapping,r['GT'])
        r['Q']=qcount;r['overflow']=r['GT']>qcount
        r['unavoidable_missing_with_K_queries']=max(0,r['GT']-r['K'])
        r['excess_missing_beyond_K_capacity']=len(r['native_missing_GT'])-r['unavoidable_missing_with_K_queries']
        r['changed_queries']={rule:sum(g!=mapping[qq] for qq,g in zip(
            r['branches']['final'][rule].get('pair_query',[]),r['branches']['final'][rule].get('pair_GT',[])))
            for rule in ['native_union','coverage_constrained','last_anchor_fill']}
        r['missing_GT_own_layer_IoU']={name:[{'GT':g,'IoU':data['own_layer_hungarian'].get('GT_best_pair_iou',{}).get(g)}
             for g in r['native_missing_GT']] for name,data in r['branches'].items()}
    for part in result['batch_losses'].values():part['ordinary_direct_components']=split_losses(part)
    result['DN_geometry']=dn_diagnostics(outputs,targets,y,result)
    result['diagnostic_extension']='v2: DN/ties/overflow/components; no change to original mappings or frozen outputs'
    return result
