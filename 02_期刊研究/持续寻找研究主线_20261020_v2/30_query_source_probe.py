"""Fixed-300 real-source admission probe. Frozen model, no backward or update."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import datetime,gc,json,random,sys,time,traceback,types,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1]
sys.path.insert(0,str(WORK/'英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1'))
import common as n
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
torch,np=n.torch,n.np
DATA=WORK/'英文小论文/data/08_匹配聚合覆盖探针_20261019_v1'
OLD=WORK/'英文小论文/data/09_DN初始化冻结诊断_20261020_v1'
OUT=ROOT/'31_query_source_probe'
ARMS=['original','GTcenter_source','rank301','DDQ_initial_NMS','fixed_random_source']


def save(name,obj):n.save_json(OUT/name,obj)


def log(**kw):
    row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),**kw}
    print(json.dumps(row,ensure_ascii=False),flush=True)
    with (OUT/'progress.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')


def matching_count(matrix):
    return int((maximum_bipartite_matching(csr_matrix(matrix),perm_type='column')>=0).sum())


def matrix_iou(boxes,targets):
    return n.box_iou(n.box_cxcywh_to_xyxy(boxes),n.box_cxcywh_to_xyxy(targets))[0]


def image_metrics(output,targets,image):
    boxes=output['pred_boxes'][image]; logits=output['pred_logits'][image]
    target=targets[image]; pair=matrix_iou(boxes,target['boxes'])
    maximum=pair.max(0).values
    scores=logits.sigmoid();C=logits.shape[-1]
    values,indices=torch.topk(scores.flatten(),300)
    q,classes=indices//C,indices%C
    classified=pair[q]*(classes[:,None]==target['labels'][None,:])
    classified_max=classified.max(0).values
    thresholded=(classified*(values[:,None]>=.3)).max(0).values
    return {'GT':len(maximum),'Q':len(boxes),'C05':matching_count((pair>=.5).numpy()),
            'C075':matching_count((pair>=.75).numpy()),'perGT_maxIoU':maximum.tolist(),
            'flat300_trueclass_maxIoU':classified_max.tolist(),
            'flat300_score_ge03_trueclass_maxIoU':thresholded.tolist(),
            'flat300_score_ge0001_trueclass_maxIoU':(classified*(values[:,None]>=.001)).max(0).values.tolist()}


def nms_select(boxes,scores,valid,limit=300):
    ids=np.flatnonzero(valid.numpy())
    score=scores.numpy(); xy=n.box_cxcywh_to_xyxy(boxes).numpy()
    ordered=ids[np.lexsort((ids,-score[ids]))]
    remaining=ordered.copy();keep=[]
    while len(remaining) and len(keep)<limit:
        first=int(remaining[0]);keep.append(first);remaining=remaining[1:]
        if not len(remaining):break
        a=xy[first];b=xy[remaining]
        inter=np.maximum(0,np.minimum(a[2:],b[:,2:])-np.maximum(a[:2],b[:,:2])).prod(-1)
        union=np.prod(a[2:]-a[:2])+np.prod(b[:,2:]-b[:,:2],axis=-1)-inter
        overlap=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
        remaining=remaining[overlap<=.8]
    fill=limit-len(keep)
    if fill:keep+=list(int(v) for v in ordered if int(v) not in set(keep))[:fill]
    assert len(keep)==limit and len(set(keep))==limit
    return torch.tensor(keep,dtype=torch.long),fill


class SelectionSpy:
    def __init__(self,decoder):
        self.original=decoder._select_topk; self.forced=None;self.latest=None
        spy=self
        def wrapper(module,memory,logits,anchors,topk):
            assert module.query_select_method=='default' and not module.learn_query_content
            selected=torch.topk(logits.max(-1).values,topk,dim=-1).indices
            native=self.original(memory,logits,anchors,topk)
            ids=selected if spy.forced is None else spy.forced
            made=(memory.gather(1,ids[...,None].expand(-1,-1,memory.shape[-1])),
                  logits.gather(1,ids[...,None].expand(-1,-1,logits.shape[-1])),
                  anchors.gather(1,ids[...,None].expand(-1,-1,4)))
            if spy.forced is None:
                assert all(torch.equal(a,b) for a,b in zip(native,made))
            spy.latest={'memory':memory.detach().clone(),'logits':logits.detach().clone(),
                        'anchors':anchors.detach().clone(),'selected':selected.clone()}
            return made
        decoder._select_topk=types.MethodType(wrapper,decoder)


def reseed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)


def restore(model,buffers):
    for name,v in model.named_buffers():v.copy_(buffers[name])


def synthetic():
    a=np.array([[1,1],[0,0]],dtype=bool)
    assert matching_count(a)==1 and matching_count(a[::-1])==1
    assert matching_count(np.eye(3,dtype=bool))==3
    # Same query may legitimately occupy multiple class slots in native ranking.
    scores=torch.tensor([[.9,.8],[.7,.1]])
    ids=torch.topk(scores.flatten(),3).indices
    assert (ids//2).tolist()==[0,0,1]
    boxes=torch.tensor([[.5,.5,.2,.2],[.5,.5,.2,.2],[.8,.8,.1,.1]])
    # The general NMS routine supports the same deterministic limit in this check.
    selected,fill=nms_select(boxes,torch.tensor([.9,.8,.7]),torch.ones(3,dtype=torch.bool),limit=2)
    assert selected.tolist()==[0,2] and fill==0
    return {'maximum_matching_and_permutation':True,'native_multiclass_flatten_preserved':True,'NMS_prediction_only':True}


def main():
    OUT.mkdir(exist_ok=False);started=time.time();resource=n.resource_limits()
    torch.set_grad_enabled(False)
    save('00_implementation.json',{'protocol_sha256':n.sha(ROOT/'29_真实source替换冻结首检事前协议.txt'),
                                 'script_sha256':n.sha(__file__),'resource':resource,'synthetic':synthetic(),
                                 'model_backwards':0,'optimizer_steps':0})
    protocol=json.loads((OLD/'01_锁定资产与诊断口径.json').read_text(encoding='utf8'))
    recipes=json.loads((DATA/'baseline_recipes.json').read_text(encoding='utf-8-sig'))
    models={};buffers={};spies={};base={};payloads={};sources=[];hashes={}
    for item in protocol['batches']:
        path=DATA/'shared_inputs/inputs'/item['file'];assert n.sha(path)==item['sha256']
        payloads[item['batch']]=torch.load(path,map_location='cpu',weights_only=False)
    for seed in [1,2]:
        cp=Path(protocol['checkpoints'][str(seed)]['path'])
        digest=n.sha(cp);assert digest==protocol['checkpoints'][str(seed)]['sha256']
        raw=torch.load(cp,map_location='cpu',weights_only=False)
        model=n.build_model(recipes[str(seed)]);model.load_state_dict(raw['model'],strict=True);model.train()
        assert not model.decoder.mgsa_groups and not model.decoder.o2m_t and model.o2m_c is None
        assert model.decoder.num_queries==300
        for p in model.parameters():p.requires_grad_(False)
        models[seed]=model;buffers[seed]={k:v.clone() for k,v in model.named_buffers()};hashes[seed]=n.model_hash(model)
        spies[seed]=SelectionSpy(model.decoder);base[seed]={};del raw;gc.collect()
        for batch,payload in payloads.items():
            restore(model,buffers[seed]);dnseed=payload['spec']['forward_DN_seed'];reseed(dnseed)
            feats=model.encoder(model.backbone(payload['images']))
            rng=(random.getstate(),np.random.get_state(),torch.get_rng_state())
            out=model.decoder(feats,payload['targets'])
            captured=spies[seed].latest
            metrics=[image_metrics(out,payload['targets'],i) for i in range(4)]
            # The exact same cached-encoder pipeline must reproduce all ordinary tensors.
            restore(model,buffers[seed]);random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
            replay=model.decoder(feats,payload['targets'])
            errors={key:float((out[key]-replay[key]).abs().max()) for key in ['pred_boxes','pred_logits','pred_corners','ref_points']}
            assert all(v==0 for v in errors.values()),errors
            base[seed][batch]={'features':feats,'rng':rng,'metrics':metrics,'capture':captured}
            torch.save({'pred_boxes':out['pred_boxes'],'pred_logits':out['pred_logits'],'metrics':metrics,'selected':captured['selected']},OUT/f'seed{seed}_batch{batch:02d}_baseline.pt')
            sources.append({'seed':seed,'batch':batch,'checkpoint':str(cp),'checkpoint_sha256':digest,
                            'payload':str(DATA/'shared_inputs/inputs'/protocol['batches'][batch]['file']),
                            'payload_sha256':protocol['batches'][batch]['sha256'],'baseline_replay_errors':errors})
            log(seed=seed,batch=batch,stage='baseline_and_exact_replay_complete',seconds=time.time()-started)
            del out,replay;gc.collect()
    chosen={}
    for batch,payload in payloads.items():
        chosen[batch]=[]
        for image,t in enumerate(payload['targets']):
            small=(t['boxes'][:,2:].min(-1).values*640<4).numpy()
            missed=(np.asarray(base[1][batch]['metrics'][image]['perGT_maxIoU'])<.5)&(np.asarray(base[2][batch]['metrics'][image]['perGT_maxIoU'])<.5)
            eligible=np.flatnonzero(small&missed)
            chosen[batch].append(int(eligible[0]) if len(eligible) else None)
    save('01_baseline_only_target_selection.json',{'chosen':chosen,'rule':'first common baseline-missed shortside<4 GT per image; no candidate outcomes accessed'})
    rows=[]
    for seed in [1,2]:
        model=models[seed];spy=spies[seed]
        for batch,payload in payloads.items():
            b=base[seed][batch];cap=b['capture'];old=cap['selected']
            assert old.shape==(4,300)
            dense_boxes=(model.decoder.enc_bbox_head(cap['memory'])+cap['anchors']).sigmoid()
            valid=torch.isfinite(cap['anchors']).all(-1)
            score=cap['logits'].max(-1).values
            ids_by_arm={a:old.clone() for a in ARMS}
            metadata=[]
            for image,g in enumerate(chosen[batch]):
                if g is None:metadata.append(None);continue
                selected=set(old[image].tolist())
                valid_ids=np.flatnonzero(valid[image].numpy())
                ordered=valid_ids[np.lexsort((valid_ids,-score[image,valid_ids].numpy()))]
                unselected=[int(v) for v in ordered if int(v) not in selected]
                natural_order=old[image].tolist()+unselected
                p3=torch.tensor(sorted(v for v in unselected if v<80*80),dtype=torch.long)
                assert len(p3)>0
                site=torch.stack([(p3%80+.5)/80,(p3//80+.5)/80],-1)
                center=payload['targets'][image]['boxes'][g,:2]
                distance=(site-center).square().sum(-1)
                position=int(torch.argmin(distance));admit=int(p3[position]);rank301=unselected[0]
                random_id=int(np.random.default_rng(20261021+batch*4+image).choice(p3.numpy()))
                ids_by_arm['GTcenter_source'][image,-1]=admit
                ids_by_arm['rank301'][image,-1]=rank301
                ids_by_arm['fixed_random_source'][image,-1]=random_id
                keep,fill=nms_select(dense_boxes[image],score[image],valid[image])
                ids_by_arm['DDQ_initial_NMS'][image]=keep
                target=payload['targets'][image]['boxes'][g:g+1]
                encoder_iou=float(matrix_iou(dense_boxes[image,admit:admit+1],target)[0,0])
                rank=natural_order.index(admit)+1
                metadata.append({'GT':g,'label':int(payload['targets'][image]['labels'][g]),
                                 'GT_count':len(payload['targets'][image]['labels']),'N_over_Q':len(payload['targets'][image]['labels'])>300,
                                 'admitted_source':admit,'victim_source':int(old[image,-1]),'rank301_source':rank301,
                                 'random_source':random_id,'admitted_original_rank':rank,'in_rank301_364':301<=rank<=364,
                                 'source_center_distance_px':float(distance[position].sqrt()*640),
                                 'admitted_encoder_IoU':encoder_iou,'NMS_filled_sources':fill})
            arm_metrics={'original':b['metrics']}
            for arm in ARMS[1:]:
                restore(model,buffers[seed]);random.setstate(b['rng'][0]);np.random.set_state(b['rng'][1]);torch.set_rng_state(b['rng'][2])
                spy.forced=ids_by_arm[arm]
                out=model.decoder(b['features'],payload['targets'])
                arm_metrics[arm]=[image_metrics(out,payload['targets'],i) for i in range(4)]
                torch.save({'pred_boxes':out['pred_boxes'],'pred_logits':out['pred_logits'],'selected':ids_by_arm[arm],
                            'metrics':arm_metrics[arm]},OUT/f'seed{seed}_batch{batch:02d}_{arm}.pt')
                log(seed=seed,batch=batch,arm=arm,stage='frozen_decoder_complete',seconds=time.time()-started)
                del out;gc.collect()
            spy.forced=None
            for image,meta in enumerate(metadata):
                if meta is None:continue
                g=meta['GT'];before=b['metrics'][image]
                cells={}
                for arm in ARMS:
                    after=arm_metrics[arm][image]
                    old_i=np.asarray(before['perGT_maxIoU']);new_i=np.asarray(after['perGT_maxIoU'])
                    cells[arm]={'C05':after['C05'],'C075':after['C075'],'target_IoU':after['perGT_maxIoU'][g],
                                'target_recovered':after['perGT_maxIoU'][g]>=.5,
                                'flat300_trueclass_target_recovered':after['flat300_trueclass_maxIoU'][g]>=.5,
                                'flat300_score_ge0001_trueclass_target_recovered':after['flat300_score_ge0001_trueclass_maxIoU'][g]>=.5,
                                'flat300_score_ge03_trueclass_target_recovered':after['flat300_score_ge03_trueclass_maxIoU'][g]>=.5,
                                'perGT_new05':int(((old_i<.5)&(new_i>=.5)).sum()),
                                'perGT_lost05':int(((old_i>=.5)&(new_i<.5)).sum()),
                                'perGT_new075':int(((old_i<.75)&(new_i>=.75)).sum()),
                                'perGT_lost075':int(((old_i>=.75)&(new_i<.75)).sum())}
                rows.append({'seed':seed,'batch':batch,'image':image,'metadata':meta,'arms':cells})
        restore(model,buffers[seed]);assert n.model_hash(model)==hashes[seed]
        assert n.sha(Path(protocol['checkpoints'][str(seed)]['path']))==protocol['checkpoints'][str(seed)]['sha256']
    pooled={};decision={}
    for seed in [1,2]:
        chosen_rows=[r for r in rows if r['seed']==seed];attempts=len(chosen_rows)
        pooled[seed]={}
        for arm in ARMS:
            pooled[seed][arm]={key:sum(r['arms'][arm][key] for r in chosen_rows) for key in chosen_rows[0]['arms'][arm] if key!='target_IoU'} if attempts else {}
        if not attempts:decision[seed]={'attempts':0,'passed':False,'reason':'NA no eligible target'};continue
        p=pooled[seed];recovered=p['GTcenter_source']['target_recovered']
        decision[seed]={'attempts':attempts,'recovered':recovered,'recovery_at_least10percent':10*recovered>=attempts,
                        'C05_not_lower':p['GTcenter_source']['C05']>=p['original']['C05'],
                        'strictly_more_recovery_than_rank301':recovered>p['rank301']['target_recovered'],
                        'successful_sources_inside_rank301_364':sum(r['arms']['GTcenter_source']['target_recovered'] and r['metadata']['in_rank301_364'] for r in chosen_rows)}
        decision[seed]['passed']=decision[seed]['recovery_at_least10percent'] and decision[seed]['C05_not_lower'] and decision[seed]['strictly_more_recovery_than_rank301']
    result={'state':'complete','sources':sources,'rows':rows,'pooled':pooled,'decision':decision,
            'pilot_opportunity_passed_both_seeds':all(r['passed'] for r in decision.values()),
            'protocol_sha256':n.sha(ROOT/'29_真实source替换冻结首检事前协议.txt'),
            'model_state_unchanged':True,'model_backwards':0,'optimizer_steps':0,'new_training':False,'heldout_reads':False,
            'seconds':time.time()-started,'limits':'GT chooses intervention source only; own predicted reference and real content, same300. Not deployable GT selector, AP or proof of utility learning. Initial NMS is not full DDQ reproduction.'}
    save('02_results.json',result)
    save('03_manifest.json',{p.name:n.sha(p) for p in OUT.iterdir() if p.is_file()})
    log(stage='complete',decision=decision,passed=result['pilot_opportunity_passed_both_seeds'],seconds=time.time()-started)


if __name__=='__main__':
    def hard_timeout():
        try:
            if OUT.exists():save('timeout.json',{'seconds':900,'only_own_process':True,'pid':os.getpid(),'model_updates':0})
        finally:os._exit(124)
    watchdog=threading.Timer(900,hard_timeout);watchdog.daemon=True;watchdog.start()
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'automatic_identical_retry':False})
        raise
    finally:watchdog.cancel()
