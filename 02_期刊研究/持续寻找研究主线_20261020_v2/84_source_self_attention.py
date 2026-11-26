"""Frozen source-specific versus equal-size random query self-attention grouping."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import pathlib,sys,json,importlib.util,time,copy,gc,hashlib,types
ROOT=pathlib.Path(__file__).resolve().parent;WORK=ROOT.parents[1]
sys.path.insert(0,str(WORK/'英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1'))
import common as n
from PIL import Image
from torch.nn import functional as TF
torch,np=n.torch,n.np
OUT=ROOT/'85_source_self_attention'

def module(name,file):
 spec=importlib.util.spec_from_file_location(name,ROOT/file);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

def source_map(renderer,event,row):
 rgb,mask=renderer.restore(event);W,H=rgb.size;ids=np.array(mask)
 rr={'pieces':row['pieces'],'arms':{'all_count':{'layout':row['layout'],'scales':row['scales']}}}
 scale,shift,cell=renderer.mapping('all_count',rr,None)
 if row['layout']=='original':result=mask
 else:
  arr=np.zeros((H,W),np.uint8);x=(np.arange(W)+.5)*640/W;y=(np.arange(H)+.5)*640/H
  for i in range(4):
   layer=Image.fromarray((ids==i+1).astype(np.uint8),'L');s=float(scale[i]);tx,ty=shift[i]*np.array([W/640,H/640])
   layer=np.array(layer.transform((W,H),Image.Transform.AFFINE,(1/s,0,-tx/s,0,1/s,-ty/s),resample=Image.Resampling.NEAREST,fillcolor=0)).astype(bool)
   origin,size=cell;left,top=origin[i];right,bottom=origin[i]+size[i]
   layer&=(x[None,:]>=left)&(x[None,:]<right)&(y[:,None]>=top)&(y[:,None]<bottom)
   assert not (layer&(arr>0)).any();arr[layer]=i+1
  result=Image.fromarray(arr,'L')
 return torch.from_numpy(np.array(result.resize((640,640),Image.Resampling.NEAREST)).copy()).long()

def lookup(src,xy):
 xy=xy.detach();ix=(xy[...,0]*640).floor().long();iy=(xy[...,1]*640).floor().long();valid=(ix>=0)&(ix<640)&(iy>=0)&(iy<640)
 b=torch.arange(src.shape[0]).view(-1,*([1]*(xy.ndim-2))).expand_as(ix)
 val=src[b,iy.clamp(0,639),ix.clamp(0,639)];return torch.where(valid,val,torch.zeros_like(val))

def main():
 start=time.monotonic();OUT.mkdir(exist_ok=False);resource=n.resource_limits();torch.set_grad_enabled(False);torch.backends.mha.set_fastpath_enabled(False)
 renderer=module('r61','61_prepare_layout_rgb.py');metric=module('m63','63_frozen_layout_detection.py')
 data=torch.load(ROOT/'77_all_count_frozen/00_all_count_inputs.pt',map_location='cpu',weights_only=False)
 selected=[]
 for seed in [1,2]:selected += sorted([c for c in data['cases'] if c['event']['seed']==seed],key=lambda c:c['event']['position'])[:2]
 assert len(selected)==4
 traces=json.loads((ROOT/'49_augmentation_replay_audit_v4/01_transform_traces.json').read_text(encoding='utf8'));events={(e['seed'],e['position']):e for e in traces}
 georows=json.loads((ROOT/'74_layout_counterexamples/01_all_count_geometry.json').read_text(encoding='utf8'))['rows'];geo={(g['seed'],g['position']):g for g in georows}
 ids=json.loads((ROOT/'49_augmentation_replay_audit_v4/05_tagged_source_identity.json').read_text(encoding='utf8'));tags={(r['seed'],r['position']):r for r in ids}
 maps=[];gt_sources=[]
 for c in selected:
  key=c['event']['seed'],c['event']['position'];maps.append(source_map(renderer,events[key],geo[key]));gt_sources.append(torch.tensor([g['source_slot']+1 for g in tags[key]['instances']]))
 src=torch.stack(maps);images=torch.stack([data['images'][c['input_sha256']] for c in selected]);del data;gc.collect()
 n.save_json(OUT/'00_selected.json',[{'event':c['event'],'input_sha256':c['input_sha256'],'source_mask_sha256':renderer.tensor_hash(m)} for c,m in zip(selected,maps)])
 torch.save({'events':[c['event'] for c in selected],'source_maps':src},OUT/'00_source_maps.pt')
 protocol=json.loads((WORK/'英文小论文/data/09_DN初始化冻结诊断_20261020_v1/01_锁定资产与诊断口径.json').read_text(encoding='utf8'))
 recipes=json.loads((WORK/'英文小论文/data/08_匹配聚合覆盖探针_20261019_v1/baseline_recipes.json').read_text(encoding='utf-8-sig'))
 rows=[];totals={};decisions={};parity={};state_equal={};fwd=0;attention=[]
 for seed in [1,2]:
  cp=pathlib.Path(protocol['checkpoints'][str(seed)]['path']);cp_sha=n.sha(cp);assert cp_sha==protocol['checkpoints'][str(seed)]['sha256']
  raw=torch.load(cp,map_location='cpu',weights_only=False);recipe=copy.deepcopy(recipes[str(seed)]);recipe['HGNetv2']['pretrained']=False
  model=n.build_model(recipe);model.load_state_dict(raw['model'],strict=True);model.eval();del raw;gc.collect();before=n.model_hash(model)
  dec=model.decoder;original_select=dec._select_topk;context={'mode':'original','source_ids':None,'layer':0}
  def select_hook(self,memory,outputs_logits,anchors,topk):
   result=original_select(memory,outputs_logits,anchors,topk);centers=result[2].sigmoid()[...,:2]
   sid=lookup(src,centers);assert sid.shape==(4,300);context['source_ids']=sid
   random=[]
   for j,c in enumerate(selected):
    gen=torch.Generator().manual_seed(int(hashlib.sha256(('source-group:'+c['input_sha256']).encode()).hexdigest()[:15],16))
    random.append(sid[j,torch.randperm(300,generator=gen)])
   context['random_ids']=torch.stack(random);context['layer']=0
   return result
  dec._select_topk=types.MethodType(select_hook,dec)
  handles=[]
  def mha_pre(mod,args,kwargs):
   sid=context['source_ids'];assert sid is not None
   group=context['random_ids'] if context['mode']=='random_source_sizes' else sid
   if context['mode']!='original':
    mask=group[:,:,None]!=group[:,None,:];assert not torch.diagonal(mask,dim1=1,dim2=2).any()
    kwargs=dict(kwargs);assert kwargs.get('attn_mask') is None
    kwargs['attn_mask']=mask.repeat_interleave(mod.num_heads,dim=0)
   if context['mode']=='original':
    q,k=args[:2];d=mod.embed_dim;h=mod.num_heads;hd=d//h
    qp=TF.linear(q,mod.in_proj_weight[:d],mod.in_proj_bias[:d]).view(4,300,h,hd).transpose(1,2)
    kp=TF.linear(k,mod.in_proj_weight[d:2*d],mod.in_proj_bias[d:2*d]).view(4,300,h,hd).transpose(1,2)
    weights=torch.softmax(torch.matmul(qp,kp.transpose(-1,-2))/(hd**.5),dim=-1)
    foreign=sid[:,:,None]!=sid[:,None,:]
    attention.append({'model_seed':seed,'layer':context['layer']-1,'foreign_attention_mass_per_query':(weights*foreign[:,None]).sum(-1).mean(1).tolist(),
       'source_counts':[[int((s==i).sum()) for i in range(5)] for s in sid]})
   return args,kwargs
  def layer_pre(mod,args,kwargs):
   reference=args[1] if len(args)>1 else kwargs['reference_points'];xy=reference[..., :2]
   if xy.ndim==4:xy=xy[:,:,0]
   now=lookup(src,xy);sid=context['source_ids']
   if context['mode']=='original':
    attention.append({'model_seed':seed,'reference_layer':context['layer'],'reference_source_changes_per_case':(now!=sid).sum(1).tolist(),
      'reference_source_counts':[[int((s==i).sum()) for i in range(5)] for s in now]})
   context['layer']+=1
  for layer in dec.decoder.layers:
   handles.append(layer.register_forward_pre_hook(layer_pre,with_kwargs=True));handles.append(layer.self_attn.register_forward_pre_hook(mha_pre,with_kwargs=True))
  cache=torch.load(ROOT/f'77_all_count_frozen/seed{seed}_all_count_predictions.pt',map_location='cpu',weights_only=False)
  allpred={};seed_rows=[];arm_totals={}
  for arm in ['original','true_source','random_source_sizes']:
   assert time.monotonic()-start<1200 and n.psutil.virtual_memory().available/2**30>8
   context['mode']=arm;out=model(images);fwd+=1;assert fwd<=12
   assert torch.isfinite(out['pred_boxes']).all() and torch.isfinite(out['pred_logits']).all()
   allpred[arm]={f:out[f].clone() for f in ['pred_boxes','pred_logits']}
   rr=[]
   for j,c in enumerate(selected):
    prediction={f:out[f][j] for f in ['pred_boxes','pred_logits']};r=metric.metrics(prediction,c['target'],c['original_sub4'])
    pair=n.box_iou(n.box_cxcywh_to_xyxy(prediction['pred_boxes']),n.box_cxcywh_to_xyxy(c['target']['boxes']))[0]
    value,index=torch.topk(prediction['pred_logits'].sigmoid().flatten(),300);query,cl=index//10,index%10
    hit=(pair[query]>=.5)&(cl[:,None]==c['target']['labels'][None,:]);origin=context['source_ids'][j,query]
    same=origin[:,None]==gt_sources[j][None,:];foreign=(origin[:,None]>0)&~same
    for lab,sel in [('same_source',same),('other_source',foreign),('padding_source',origin[:,None]==0)]:
     present=(hit&sel).any(0);r[f'perGT_{lab}_class05']=present.tolist();r[f'{lab}_class05_presence']=int(present.sum())
    rr.append(r);seed_rows.append({'model_seed':seed,'arm':arm,'event':c['event'],'metrics':r})
   t={'cases':len(rr)}
   for k in rr[0]:
    if not isinstance(rr[0][k],list):t[k]=sum(r[k] for r in rr)
   arm_totals[arm]=t
  original=allpred['original'];parity[str(seed)]={'max_box_error':max(float((original['pred_boxes'][i]-cache[c['input_sha256']]['pred_boxes']).abs().max()) for i,c in enumerate(selected)),
    'max_logit_error':max(float((original['pred_logits'][i]-cache[c['input_sha256']]['pred_logits']).abs().max()) for i,c in enumerate(selected))}
  parity[str(seed)]['coverage_equal_to_cached']=all(
    metric.metrics(cache[c['input_sha256']],c['target'],c['original_sub4'])[field]
    == next(r['metrics'][field] for r in seed_rows if r['arm']=='original' and r['event']==c['event'])
    for c in selected for field in ['all_class05','all_class075','sub4_class05','sub4_class075','all_geo05','all_geo075','sub4_geo05','sub4_geo075'])
  for arm in ['true_source','random_source_sizes']:
   for j,c in enumerate(selected):
    old=metric.metrics({f:original[f][j] for f in original},c['target'],c['original_sub4'])
    r=next(r['metrics'] for r in seed_rows if r['arm']==arm and r['event']==c['event'])
    for name in ['class05','class075','geo05','geo075']:
     a=np.array(old[f'perGT_{name}']);b=np.array(r[f'perGT_{name}']);r[f'new_{name}']=int((~a&b).sum());r[f'lost_{name}']=int((a&~b).sum())
  totals[str(seed)]=arm_totals
  a,b,c=arm_totals['true_source'],arm_totals['original'],arm_totals['random_source_sizes']
  decisions[str(seed)]={'all_class05_over_original':a['all_class05']>b['all_class05'],'all_class05_over_random':a['all_class05']>c['all_class05'],'sub4_class05_not_lower':a['sub4_class05']>=b['sub4_class05']}
  decisions[str(seed)]['passed']=all(decisions[str(seed)].values())
  rows+=seed_rows;torch.save(allpred,OUT/f'seed{seed}_predictions.pt')
  for h in handles:h.remove()
  dec._select_topk=original_select;state_equal[str(seed)]=n.model_hash(model)==before;assert state_equal[str(seed)] and n.sha(cp)==cp_sha
  del model,cache,allpred;gc.collect()
 n.save_json(OUT/'01_rows.json',rows);n.save_json(OUT/'02_attention.json',attention)
 n.save_json(OUT/'03_results.json',{'state':'complete','resources':resource,'totals':totals,'decision':decisions,'both_passed':all(d['passed'] for d in decisions.values()),
  'backend_parity':parity,'model_state_unchanged':state_equal,'actual_batch_forwards':fwd,'model_backwards':0,'optimizer_steps':0,'new_training':False,'heldout_reads':False,
  'seconds':time.monotonic()-start,'script_sha256':n.sha(__file__),'protocol_sha256':n.sha(ROOT/'83_来源自注意力冻结诊断_事前协议.txt'),
  'source_sha256':{f:n.sha(ROOT/f) for f in ['77_all_count_frozen/00_all_count_inputs.pt','74_layout_counterexamples/01_all_count_geometry.json']},
  'scope':'Source-specific grouping diagnostic, not source isolation novelty, AP, or training benefit.'})
 n.save_json(OUT/'04_SHA256.json',{p.name:n.sha(p) for p in OUT.iterdir() if p.is_file()})
 print(json.dumps({'state':'complete','totals':totals,'decision':decisions,'parity':parity,'forwards':fwd,'seconds':time.monotonic()-start},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
