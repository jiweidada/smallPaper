"""Interpret native top-k slot movement using isolated frozen virtual endpoints.

Two seeds, old development batches00/01, original loss/Adam recipe only.
No optimizer.step, checkpoint write, remote or GPU. Old code imported unchanged.
"""
import datetime, gc, hashlib, json, os, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent; WORK=ROOT.parents[1]
sys.path.insert(0,str(WORK/'英文小论文/code/source_allocation_frozen_20261020_v1'))
import diagnostic as d
torch,np=d.torch,d.np
OLD=WORK/'英文小论文/data/10_来源分配冻结诊断_20261020_v1'
OUT=ROOT/'topk_membership_frozen'

def save(p,x):d.native.save_json(p,x)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def snapshot(selector,outputs):
    return dict(indices=selector.observed.detach().clone(),
                final=outputs['pred_boxes'].detach().clone(),
                logits=outputs['pred_logits'].detach().clone(),
                encoder=outputs['enc_aux_outputs'][0]['pred_boxes'].detach().clone())
def evaluate_base_sites(before,after,targets):
    rows=[]
    for image,target in enumerate(targets):
        base_ids=before['indices'][image].tolist();new_ids=set(after['indices'][image].tolist())
        assert len(set(base_ids))==len(base_ids)==300
        shape=target['boxes'];tiny=(shape[:,2:].min(-1).values*640<4)
        for stage in ['encoder','final']:
            matrix=d.box_iou(d.box_cxcywh_to_xyxy(before[stage][image]),d.box_cxcywh_to_xyxy(shape))[0]
            maximum=matrix.max(0).values
            # Fixed deterministic tie rule: lowest spatial tokenID among maxima.
            order=torch.tensor(sorted(range(300),key=lambda j:base_ids[j]))
            best=order[matrix[order].argmax(0)]
            kept=torch.tensor([base_ids[q] in new_ids for q in best.tolist()])
            for label,mask in [('shortside_lt4',tiny),('shortside_ge4',~tiny)]:
                selected=mask&(maximum>=.5)
                rows.append(dict(image=image,stage=stage,size=label,GT=int(mask.sum()),
                    baseline_IoU_ge_0_5_GT=int(selected.sum()),
                    best_source_site_retained=int(kept[selected].sum()),
                    best_source_site_lost=int((~kept[selected]).sum())))
    return rows
def main():
    limits=d.native.resource_limits();assert limits['available_RAM_GiB']>12
    base_protocol=json.loads((OLD/'01_冻结协议与资产.json').read_text(encoding='utf8'))
    for row in base_protocol['sources']:assert sha(row['path'])==row['sha256']
    OUT.mkdir(exist_ok=False)
    protocol=dict(utc=d.now(),cohort='BASE24online seed1/2, saved development00/01; same8 augmentedimages across seeds, not independent validation',
        objective='Audit whether previously reported ordered top-k movement means physical spatial-token replacement. Do not equate ordering change with candidate loss or training harm.',
        perturbation='Only original complete native loss, all originally trainable params, global0.1clip and copied historical AdamW moments/currentLR. Independent virtual endpoints, never accumulated/writtenback.',
        measurements=['ordered slot differences','old members missing from new set','retained members with changed rank','exact decomposition ordered_changed=members_lost+retained_rank_changed','best baseline selected-encoder/final geometry source-site retention for GT with maxIoU>=.5, tiny shortside<4 versus others','same-set reverse-order control at unchanged parameters, inverse-permute ordinary outputs before comparing'],
        reference='Reproduce old data10 base topkindices, old gradient/update statistics and native/fixed endpoint metrics before interpreting new counts.',
        decision='Descriptive falsification, no posthoc gate: report ordering-versus-membership decomposition in every endpoint. A large ordered count alone cannot support instability. No detection/AP or training priority follows from membership loss or size association; near-neighbor literature and stable detection benefit still needed.',
        tie_rule='For baseline bestIoU ties select lowest spatial tokenID; only base maxIoU>=.5 targets qualify for retention. Denominator0 is NA, not0.',
        limitations='Two checkpoints, two repeated augmentation batches; one historical-Adam virtual perturbation is not actual longitudinal training. Source site loss does not imply GT loss because other queries can recover it. Sizes/targets only used for diagnosis, never deployment selection.',
        resources=limits,new_training_steps=0,optimizer_step_calls=0,remote_operations=False,
        code_sha256=sha(__file__),imported_diagnostic_sha256=sha(d.__file__),old_protocol_sha256=sha(OLD/'01_冻结协议与资产.json'))
    save(OUT/'01_protocol.json',protocol)
    d.OUT=OUT;d.ARMS=['original']
    recipes=json.loads((d.OLD/'baseline_recipes.json').read_text(encoding='utf-8-sig'))
    result=[];started=time.monotonic()
    for seed in [1,2]:
        folder=OUT/f'seed{seed}';folder.mkdir()
        cp=base_protocol['checkpoints'][str(seed)];assert sha(cp['path'])==cp['sha256']
        raw=torch.load(cp['path'],map_location='cpu',weights_only=False)
        model=d.native.build_model(recipes[str(seed)]);model.load_state_dict(raw['model'],strict=True);model.train()
        assert model.decoder.learn_query_content is False
        criterion=d.native.build_criterion(recipes[str(seed)]);criterion.load_state_dict(raw['criterion'],strict=True)
        named,binding=d.full_optimizer_binding(model,raw,recipes[str(seed)])
        assert len(named)==488
        buffers={n:b.detach().clone() for n,b in model.named_buffers()}
        original_hash=d.native.model_hash(model)
        del raw;gc.collect()
        selector=d.QuerySelection(model.decoder);history=[]
        hook=model.register_forward_hook(lambda m,args,outputs: history.append(snapshot(selector,outputs)))
        cases={};base={}
        for batch in [0,1]:
            assert d.psutil.virtual_memory().available>8*2**30
            history.clear()
            cases[batch]=d.capture(model,criterion,selector,named,binding,buffers,base_protocol['batches'][batch],seed,folder)
            base[batch]=history[0]
            old=json.loads((OLD/f'seed{seed}/batch{batch:02d}_capture.json').read_text(encoding='utf8'))
            new=json.loads((folder/f'batch{batch:02d}_capture.json').read_text(encoding='utf8'))
            assert old['topk_indices']==new['topk_indices']
            for k in ['global_gradient_norm_native_FP32','global_clip_factor','parameter_update_L2_norm']:
                assert old['updates']['original'][k]==new['updates']['original'][k],k
            # Zero-parameter-change, exactly same membership, reversed order.
            d.restore_buffers(model,buffers);d.reset_rng(cases[batch]['rng'])
            selector.mode='fixed';selector.fixed=cases[batch]['topk'].flip(1)
            with torch.no_grad():
                rev=torch.func.functional_call(model,({n:p.detach() for n,p in model.named_parameters()},d.native.cpu_tree(buffers)),(cases[batch]['payload']['images'],cases[batch]['payload']['targets']),strict=True)
            box_error=float((rev['pred_boxes'].flip(1)-base[batch]['final']).abs().max())
            logit_error=float((rev['pred_logits'].flip(1)-base[batch]['logits']).abs().max())
            assert box_error<1e-5 and logit_error<1e-4,(box_error,logit_error)
            save(folder/f'batch{batch:02d}_permutation_control.json',dict(box_error=box_error,logit_error=logit_error,ordered_slots_changed=1200,members_lost=0,learn_query_content=False))
            del rev;history.clear()
        for source,target in [(0,1),(1,0)]:
            old=json.loads((OLD/f'seed{seed}/cross_{source:02d}_to_{target:02d}.json').read_text(encoding='utf8'))
            records={};native_output=None
            for mode in ['fixed','native']:
                selector.mode=mode;selector.fixed=cases[target]['topk'];d.reset_rng(cases[target]['rng']);history.clear()
                with torch.no_grad():
                    output=torch.func.functional_call(model,(cases[source]['endpoints']['original'],d.native.cpu_tree(buffers)),(cases[target]['payload']['images'],cases[target]['payload']['targets']),strict=True)
                metric=d.metric(output,cases[target]['payload']['targets'],cases[target]['union'],cases[target]['denominator'],criterion,cases[target]['own'])
                expected=old['arms']['original'][mode]['metrics']['weighted_geometry']
                assert abs(metric['weighted_geometry']-expected)<1e-12
                if mode=='native':native_output=history[-1]
                records[mode]=metric
                del output
            oldids=cases[target]['topk'];newids=native_output['indices']
            assert int((newids!=oldids).sum())==old['arms']['original']['native']['native_topk_ordered_slots_changed_vs_base']
            images=[]
            for i in range(4):
                aa=oldids[i].tolist();bb=newids[i].tolist();aset,bset=set(aa),set(bb);lost=len(aset-bset)
                ranknew={site:j for j,site in enumerate(bb)}
                retained_rank_changed=sum(site in bset and ranknew[site]!=j for j,site in enumerate(aa))
                changed=sum(x!=y for x,y in zip(aa,bb))
                assert changed==lost+retained_rank_changed
                images.append(dict(image=i,ordered_slots_changed=changed,members_lost=lost,retained_rank_changed=retained_rank_changed,intersection=len(aset&bset),old_topk=aa,new_topk=bb))
            row=dict(seed=seed,source_batch=source,evaluation_batch=target,old_metrics_reproduced=True,images=images,
                counts=dict(ordered_slots_changed=sum(x['ordered_slots_changed'] for x in images),members_lost=sum(x['members_lost'] for x in images),retained_rank_changed=sum(x['retained_rank_changed'] for x in images),slots=1200),
                GT_best_site_retention=evaluate_base_sites(base[target],native_output,cases[target]['payload']['targets']),metrics=records)
            save(folder/f'cross_{source:02d}_to_{target:02d}_membership.json',row);result.append(row)
            print(json.dumps({k:v for k,v in row.items() if k in ['seed','source_batch','evaluation_batch','counts']}),flush=True)
            del native_output;history.clear()
        hook.remove();selector.decoder._select_topk=selector.native;d.restore_buffers(model,buffers)
        assert d.native.model_hash(model)==original_hash and not any(p.grad is not None for p in model.parameters())
        assert sha(cp['path'])==cp['sha256']
        save(folder/'complete.json',dict(original_model_hash_unchanged=original_hash,checkpoint_sha256_unchanged=cp['sha256'],training_steps=0,old_replay_pass=True))
        del model,criterion,named,binding,buffers,cases,base,selector,history;gc.collect()
    save(OUT/'02_results.json',dict(protocol=protocol,rows=result,elapsed_seconds=time.monotonic()-started,training_steps=0,model_forwards=20))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save(OUT/('failure_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),dict(error=repr(e),traceback=traceback.format_exc(),blind_retry=False))
        raise
