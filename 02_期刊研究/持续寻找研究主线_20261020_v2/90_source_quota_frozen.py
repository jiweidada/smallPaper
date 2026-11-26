"""Frozen source demand allocation compared to native, minimum-floor, and uniform."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import pathlib,sys,json,importlib.util,time,copy,gc,types
ROOT=pathlib.Path(__file__).resolve().parent;WORK=ROOT.parents[1]
sys.path.insert(0,str(WORK/'英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1'))
import common as n
torch,np=n.torch,n.np
OUT=ROOT/'91_source_quota_frozen'

def module(name,file):
 spec=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def demand(counts,Q=300):
 if sum(counts)<=Q:return counts.copy()
 numerator=[Q*c for c in counts];den=sum(counts);r=[v//den for v in numerator]
 for i in sorted(range(len(r)),key=lambda i:(-(numerator[i]%den),i))[:Q-sum(r)]:r[i]+=1
 return r

def main():
 start=time.monotonic();OUT.mkdir(exist_ok=False);resource=n.resource_limits();torch.set_grad_enabled(False);torch.backends.mha.set_fastpath_enabled(False)
 assert demand([19,60,174,27])==[19,60,174,27] and demand([53,200,108,57])==[38,144,77,41]
 helper=module('s84','84_source_self_attention.py');metric=module('m63q','63_frozen_layout_detection.py')
 data=torch.load(ROOT/'77_all_count_frozen/00_all_count_inputs.pt',map_location='cpu',weights_only=False)
 sel=json.loads((ROOT/'85_source_self_attention/00_selected.json').read_text(encoding='utf8'))
 cases=[next(c for c in data['cases'] if c['event']==r['event']) for r in sel]
 src=torch.load(ROOT/'85_source_self_attention/00_source_maps.pt',map_location='cpu',weights_only=False)['source_maps']
 images=torch.stack([data['images'][c['input_sha256']] for c in cases]);del data;gc.collect()
 tags=json.loads((ROOT/'49_augmentation_replay_audit_v4/05_tagged_source_identity.json').read_text(encoding='utf8'))
 ids={(r['seed'],r['position']):r for r in tags};counts=[]
 for c in cases:
  gg=ids[c['event']['seed'],c['event']['position']]['instances'];counts.append([sum(g['source_slot']==i for g in gg) for i in range(4)])
 protocol=json.loads((WORK/'英文小论文/data/09_DN初始化冻结诊断_20261020_v1/01_锁定资产与诊断口径.json').read_text(encoding='utf8'))
 recipes=json.loads((WORK/'英文小论文/data/08_匹配聚合覆盖探针_20261019_v1/baseline_recipes.json').read_text(encoding='utf-8-sig'))
 rows=[];selections=[];totals={};decisions={};parity={};unchanged={};forwards=0
 for seed in [1,2]:
  cp=pathlib.Path(protocol['checkpoints'][str(seed)]['path']);cp_sha=n.sha(cp);assert cp_sha==protocol['checkpoints'][str(seed)]['sha256']
  raw=torch.load(cp,map_location='cpu',weights_only=False);recipe=copy.deepcopy(recipes[str(seed)]);recipe['HGNetv2']['pretrained']=False
  model=n.build_model(recipe);model.load_state_dict(raw['model'],strict=True);model.eval();del raw;gc.collect();before=n.model_hash(model)
  dec=model.decoder;assert dec.query_select_method=='default' and dec.num_queries==300
  original=dec._select_topk;context={'arm':'original'}
  def choose(self,memory,logits,anchors,topk):
   score=logits.max(-1).values;orig=torch.topk(score,topk,dim=1).indices
   source=helper.lookup(src,anchors.sigmoid()[...,:2]);idx=[]
   for b,cc in enumerate(counts):
    arm=context['arm'];reserve=[0]*4 if arm=='original' else [min(v,75) for v in cc] if arm=='floor75' else demand(cc) if arm=='demand' else [75]*4
    chosen=[];actual=[]
    for s,r in enumerate(reserve,1):
     pool=torch.where((source[b]==s)&torch.isfinite(anchors[b]).all(-1))[0];k=min(r,len(pool))
     take=pool[torch.topk(score[b,pool],k).indices] if k else pool[:0]
     chosen.extend(take.tolist());actual.append(k)
    assert len(chosen)==len(set(chosen)) and len(chosen)<=topk
    excluded=torch.zeros(len(score[b]),dtype=torch.bool);excluded[chosen]=True
    remain=topk-len(chosen);residual=score[b].clone();residual[excluded]=-torch.inf
    extra=torch.topk(residual,remain).indices.tolist() if remain else []
    taken=torch.tensor(chosen+extra,dtype=torch.long);assert len(taken)==topk and len(taken.unique())==topk
    equal=torch.equal(taken.sort().values,orig[b].sort().values)
    if equal:take=orig[b]
    else:
     taken=taken.sort().values;take=taken[torch.argsort(score[b,taken],descending=True,stable=True)]
    if arm=='original':assert torch.equal(take,orig[b])
    idx.append(take);picked=source[b,take]
    selections.append({'model_seed':seed,'arm':arm,'event':cases[b]['event'],'GT_per_source':cc,'requested_reserve':reserve,'actual_reserve':actual,
      'selected_source_counts_including_padding':[int((picked==s).sum()) for s in range(5)],'same_set_as_native':equal,
      'new_members':int((~torch.isin(take,orig[b])).sum()),'selected_indices':take.tolist(),
      'adjacent_selected_score_ties':int((score[b,take][:-1]==score[b,take][1:]).sum())})
   index=torch.stack(idx)
   return memory.gather(1,index[:,:,None].expand(-1,-1,memory.shape[-1])),None,anchors.gather(1,index[:,:,None].expand(-1,-1,4))
  dec._select_topk=types.MethodType(choose,dec)
  cache=torch.load(ROOT/f'77_all_count_frozen/seed{seed}_all_count_predictions.pt',map_location='cpu',weights_only=False)
  preds={};armtot={};rr_seed=[]
  for arm in ['original','floor75','demand','uniform75']:
   assert time.monotonic()-start<1200 and n.psutil.virtual_memory().available/2**30>8
   context['arm']=arm;out=model(images);forwards+=1;assert forwards<=12
   assert all(torch.isfinite(out[f]).all() for f in ['pred_boxes','pred_logits'])
   preds[arm]={f:out[f].clone() for f in ['pred_boxes','pred_logits']};rr=[]
   for b,c in enumerate(cases):
    r=metric.metrics({f:out[f][b] for f in ['pred_boxes','pred_logits']},c['target'],c['original_sub4']);rr.append(r)
    rr_seed.append({'model_seed':seed,'arm':arm,'event':c['event'],'metrics':r})
   t={'cases':len(rr)}
   for k in rr[0]:
    if not isinstance(rr[0][k],list):t[k]=sum(r[k] for r in rr)
   armtot[arm]=t
  o=preds['original'];parity[str(seed)]={'max_box_error':max(float((o['pred_boxes'][b]-cache[c['input_sha256']]['pred_boxes']).abs().max()) for b,c in enumerate(cases)),
   'max_logit_error':max(float((o['pred_logits'][b]-cache[c['input_sha256']]['pred_logits']).abs().max()) for b,c in enumerate(cases))}
  assert parity[str(seed)]['max_box_error']==0 and parity[str(seed)]['max_logit_error']==0
  for item in rr_seed:
   old=next(r['metrics'] for r in rr_seed if r['arm']=='original' and r['event']==item['event'])
   for name in ['class05','class075','geo05','geo075']:
    a=np.array(old[f'perGT_{name}']);b=np.array(item['metrics'][f'perGT_{name}']);item['metrics'][f'new_{name}']=int((~a&b).sum());item['metrics'][f'lost_{name}']=int((a&~b).sum())
  rows+=rr_seed;totals[str(seed)]=armtot
  d=armtot['demand'];decision={a:d['sub4_class05']>armtot[a]['sub4_class05'] for a in ['original','floor75','uniform75']}
  decision['all_class05_not_lower']=d['all_class05']>=armtot['original']['all_class05'];decision['passed']=all(decision.values());decisions[str(seed)]=decision
  torch.save(preds,OUT/f'seed{seed}_predictions.pt');dec._select_topk=original;unchanged[str(seed)]=n.model_hash(model)==before;assert unchanged[str(seed)] and n.sha(cp)==cp_sha
  del model,cache,preds;gc.collect()
 n.save_json(OUT/'00_selections.json',selections);n.save_json(OUT/'01_rows.json',rows)
 n.save_json(OUT/'02_results.json',{'state':'complete','resources':resource,'totals':totals,'decision':decisions,'both_passed':all(d['passed'] for d in decisions.values()),'native_backend_parity':parity,
  'model_state_unchanged':unchanged,'actual_batch_forwards':forwards,'model_backwards':0,'optimizer_steps':0,'new_training':False,'heldout_reads':False,
  'seconds':time.monotonic()-start,'script_sha256':n.sha(__file__),'protocol_sha256':n.sha(ROOT/'89_需求配额选点冻结首检_事前协议.txt'),
  'source_sha256':{f:n.sha(ROOT/f) for f in ['85_source_self_attention/00_selected.json','85_source_self_attention/00_source_maps.pt','77_all_count_frozen/00_all_count_inputs.pt']},
  'scope':'Same-budget GT-demand selection premise only; not new density principle, trained AP, or deployable access to GT counts.'})
 n.save_json(OUT/'03_SHA256.json',{p.name:n.sha(p) for p in OUT.iterdir() if p.is_file()})
 print(json.dumps({'state':'complete','totals':totals,'decision':decisions,'forwards':forwards,'seconds':time.monotonic()-start},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
