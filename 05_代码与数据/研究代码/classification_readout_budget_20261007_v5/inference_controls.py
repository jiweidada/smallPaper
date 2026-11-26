"""Same-head inference controls; borrowed product operation, no annotation inputs."""
import torch

def output_for_native_postprocessor(outputs,rule='direct'):
    native=outputs.get('native_logits',outputs['pred_logits'])
    semantic=outputs['semantic_logits']
    if rule=='direct':scores=semantic
    elif rule=='native':scores=native
    elif rule=='product':
        prob=native.sigmoid()*semantic.softmax(-1)
        if not ((prob>0)&(prob<1)).all():raise ValueError('Product outside finite logit range; no silent floor.')
        scores=prob.log()-torch.log1p(-prob)
    else:raise ValueError('Direct/native/product only; carrier uses BudgetPostProcessor separately.')
    return {**outputs,'pred_logits':scores}

def output_for_legacy_carrier(outputs):
    if 'native_logits' not in outputs:raise ValueError('Actual native logits required.')
    return {**outputs,'pred_logits':outputs['native_logits']}

def collapse_to_query_argmax(outputs,native_postprocessor,orig_target_sizes):
    # Ordinary argmax-collapse control: same semantic checkpoint, then per-query dedup.
    raw=output_for_native_postprocessor(outputs,'direct')
    result=native_postprocessor(raw,orig_target_sizes)
    logits=raw['pred_logits'];flat=logits.sigmoid().flatten(1).topk(300,dim=-1).indices
    if native_postprocessor.score_threshold!=0:raise ValueError('Control requires original zero threshold for occurrence alignment.')
    collapsed=[]
    for bi,(row,index) in enumerate(zip(result,flat)):
        query=index//10;seen=set();keep=[]
        for j,q in enumerate(query.tolist()):
            if q not in seen:seen.add(q);keep.append(j)
        positions=torch.tensor(keep,dtype=torch.long,device=query.device)
        unique_queries=query[positions]
        output={k:v[positions] for k,v in row.items()}
        maximum,label=logits[bi,unique_queries].max(-1)
        output['labels']=label;output['scores']=maximum.sigmoid()
        collapsed.append(output)
    return collapsed
