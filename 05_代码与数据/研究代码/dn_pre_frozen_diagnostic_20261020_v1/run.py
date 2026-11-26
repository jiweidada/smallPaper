"""Four frozen CPU forwards; native losses/gradients; cloned one-step comparisons."""
import argparse,datetime,gc,random,time,traceback,math
from common import *

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def progress(**kw):
 row={'utc':stamp(),'rss_GiB':psutil.Process().memory_info().rss/2**30,**kw}
 print(json.dumps(row,ensure_ascii=False),flush=True)
 with (OUT/'progress.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')

def synthetic_checks():
 torch.manual_seed(8091020)
 w=torch.randn(4,3,dtype=torch.float64,requires_grad=True)
 b=torch.randn(4,dtype=torch.float64,requires_grad=True)
 h=torch.randn(2,3,dtype=torch.float64,requires_grad=True)
 original=F.linear(h,w,b).square().sum()
 gh=torch.autograd.grad(original,h,retain_graph=True)[0]
 isolated=F.linear(h,w.detach(),b.detach()).square().sum()
 assert torch.equal(torch.autograd.grad(isolated,h)[0],gh)
 assert w.grad is None and b.grad is None
 theta=torch.tensor([.2,-.3],dtype=torch.float64)
 grad=torch.tensor([.4,-.7],dtype=torch.float64)
 state={'step':torch.tensor(17.),'exp_avg':torch.tensor([.03,-.02],dtype=torch.float64),'exp_avg_sq':torch.tensor([.04,.06],dtype=torch.float64)}
 group={'amsgrad':False,'betas':[.9,.999],'lr':.0004,'weight_decay':.0001,'eps':1e-8,'maximize':False}
 bind={'test':{'state':state,'group':group}}
 got=adam_virtual([theta],[grad],bind,.37)[0]
 m=.9*state['exp_avg']+.1*grad*.37
 v=.999*state['exp_avg_sq']+.001*(grad*.37).square()
 expected=theta*(1-.0004*.0001)-.0004*(m/(1-.9**18))/(torch.sqrt(v/(1-.999**18))+1e-8)
 err=float((got-expected).abs().max());assert err<1e-14,err
 assert state['step'].item()==17 and torch.equal(theta,torch.tensor([.2,-.3],dtype=torch.float64))
 return {'DN_upstream_gradient_preserved':True,'functional_AdamW_matches_explicit_formula_max_abs':err,'original_state_and_parameters_unmodified':True}

def capture_seed(seed,protocol):
 folder=OUT/f'seed{seed}';folder.mkdir(exist_ok=False)
 y=json.loads((OLD/'baseline_recipes.json').read_text(encoding='utf-8-sig'))[str(seed)]
 cp=pathlib.Path(protocol['checkpoints'][str(seed)]['path'])
 assert sha(cp)==protocol['checkpoints'][str(seed)]['sha256']
 raw=torch.load(cp,map_location='cpu',weights_only=False);assert raw['last_epoch']==23
 model=build_model(y);model.load_state_dict(raw['model'],strict=True);model.train()
 criterion=build_criterion(y);criterion.load_state_dict(raw['criterion'],strict=True)
 names,bindings=optimizer_binding(model,raw,y)
 named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
 positions=[next(i for i,(n,p) in enumerate(named) if n==name) for name in names]
 head_params=[named[i][1] for i in positions]
 weights=[p.detach().clone() for p in head_params]
 initial_hash=model_hash(model);buffers={n:v.detach().clone() for n,v in model.named_buffers()}
 del raw;gc.collect()
 captures={};hooks={}
 def before_decoder(module,args):
  assert 'anchors' not in hooks
  hooks['anchors']=args[1].detach().sigmoid().clone()
  hooks['anchor_logits']=inverse_sigmoid(hooks['anchors']).clone()
 def before_last(module,args):
  assert 'hidden' not in hooks
  hooks['hidden']=args[0].detach().clone()
 handles=[model.decoder.decoder.register_forward_pre_hook(before_decoder),model.decoder.pre_bbox_head.layers[-1].register_forward_pre_hook(before_last)]
 def restore_buffers():
  with torch.no_grad():
   for n,v in model.named_buffers():v.copy_(buffers[n])
 for row in protocol['batches']:
  batch=row['batch'];progress(seed=seed,batch=batch,stage='native_forward_start')
  payload_path=OLD/'shared_inputs/inputs'/row['file'];assert sha(payload_path)==row['sha256']
  payload=torch.load(payload_path,map_location='cpu',weights_only=False)
  assert __import__('hashlib').sha256(payload['images'].numpy().tobytes()).hexdigest()==row['float_input_sha256']
  dn_seed=payload['spec']['forward_DN_seed'];rng=(random.getstate(),np.random.get_state(),torch.get_rng_state())
  random.seed(dn_seed);np.random.seed(dn_seed);torch.manual_seed(dn_seed)
  restore_buffers();hooks.clear();started=time.perf_counter()
  outputs=model(payload['images'],payload['targets'])
  progress(seed=seed,batch=batch,stage='native_losses',forward_seconds=time.perf_counter()-started)
  assert 'dn_pre_outputs' in outputs
  criterion.mark(outputs);losses=criterion(outputs,payload['targets'],epoch=24)
  lo=losses['loss_bbox_pre']+losses['loss_giou_pre']
  ld=losses['loss_bbox_dn_pre']+losses['loss_giou_dn_pre']
  assert all(torch.isfinite(v).all() for v in losses.values())
  go=torch.autograd.grad(lo,head_params,retain_graph=True)
  gd=torch.autograd.grad(ld,head_params,retain_graph=True)
  progress(seed=seed,batch=batch,stage='full_native_gradient_norm_start')
  grads=torch.autograd.grad(sum(losses.values()),[p for n,p in named],allow_unused=True)
  assert all(g is None or torch.isfinite(g).all() for g in grads)
  total_norm_sq=sum(float(g.double().square().sum()) for g in grads if g is not None)
  total_head=[grads[i].detach().clone() for i in positions]
  residual=[g-o-d for g,o,d in zip(total_head,go,gd)]
  assert norm(residual)<2e-5*max(1.,norm(total_head)),norm(residual)
  no_dn=[g-d for g,d in zip(total_head,gd)]
  norm_sq_without=total_norm_sq-dot(total_head,total_head)+dot(no_dn,no_dn)
  assert norm_sq_without>=0
  split=outputs['dn_meta']['dn_num_split'][0]
  cache={**cpu_tree(hooks),'targets':cpu_tree(payload['targets']),'ordinary':criterion.record['ordinary'],'dn':criterion.record['dn'],'dn_split':split,'seed':seed,'batch':batch,'DN_seed':dn_seed,'weights':weights,'optimizer_bindings':bindings,'gradient_ordinary':cpu_tree(go),'gradient_DN':cpu_tree(gd),'gradient_total_last':total_head,'gradient_total_without_DN_last':no_dn,'global_grad_norm_include_DN':total_norm_sq**.5,'global_grad_norm_exclude_DN_last':norm_sq_without**.5,'native_ordinary_pre_loss':float(lo),'native_DN_pre_loss':float(ld),'native_loss_components':{k:float(v.detach()) for k,v in losses.items()},'native_pre_boxes':cpu_tree(outputs['pre_outputs']['pred_boxes']),'GT_counts':[len(t['labels']) for t in payload['targets']]}
  assert cache['ordinary']['denominator']==sum(len(x[0]) for x in cache['ordinary']['indices'])
  assert cache['dn']['denominator']==sum(cache['GT_counts'])*outputs['dn_meta']['dn_num_group']
  native_indices=criterion._get_go_indices
  # Verify the cached-head reconstruction against native predictions and gradients.
  reconstructed=head_boxes(cache,weights,dtype=torch.float32)
  box_error=float((reconstructed-cache['native_pre_boxes']).abs().max());assert box_error<3e-7,box_error
  w_check=[p.detach().clone().requires_grad_() for p in weights]
  test_loss=head_loss(cache,w_check,criterion,dtype=torch.float32)[0]
  test_grads=torch.autograd.grad(test_loss,w_check)
  loss_error=abs(float(test_loss)-cache['native_ordinary_pre_loss']);assert loss_error<3e-5,loss_error
  grad_error=max(float((a-b).abs().max()) for a,b in zip(test_grads,go));assert grad_error<3e-5,grad_error
  dn_check=head_loss(cache,weights,criterion,dtype=torch.float32,branch='dn')[0]
  dn_error=abs(float(dn_check)-cache['native_DN_pre_loss']);assert dn_error<3e-5,dn_error
  cache['verification']={'cached_pre_boxes_max_abs':box_error,'ordinary_loss_abs':loss_error,'ordinary_gradient_max_abs':grad_error,'DN_loss_abs':dn_error,'total_last_minus_ordinary_minus_DN_norm':norm(residual)}
  with (folder/f'batch{batch:02d}_frozen_head.pt').open('xb') as f:torch.save(cache,f)
  captures[batch]=cache
  restore_buffers();random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
  assert not any(p.grad is not None for p in model.parameters())
  assert model_hash(model)==initial_hash
  progress(seed=seed,batch=batch,stage='batch_captured',seconds=time.perf_counter()-started,ordinary_pairs=sum(len(q) for q,g in cache['ordinary']['indices']),DN_pairs=sum(len(q) for q,g in cache['dn']['indices']),global_gradient_norm=cache['global_grad_norm_include_DN'],cache_checks=cache['verification'])
  del outputs,losses,lo,ld,grads,payload,go,gd,test_loss,test_grads,w_check,dn_check;gc.collect()
 for h in handles:h.remove()
 assert sha(cp)==protocol['checkpoints'][str(seed)]['sha256']
 assert model_hash(model)==initial_hash
 save_json(folder/'capture_complete.json',{'seed':seed,'checkpoint_sha256':sha(cp),'initial_and_final_model_state_sha256':initial_hash,'model_parameters_unchanged':True,'BN_buffers_restored':True,'optimizer_step_calls':0,'checkpoint_writes':0,'capture_files':[{ 'file':f'batch{b:02d}_frozen_head.pt','sha256':sha(folder/f'batch{b:02d}_frozen_head.pt')} for b in captures],'head_optimizer_binding':[{ 'name':n,'id':bindings[n]['id'],'lr':bindings[n]['group']['lr'],'state_step':float(bindings[n]['state']['step'])} for n in names]})
 del model,criterion;gc.collect()
 return captures,y

def analyze_seed(seed,caches,y):
 criterion=build_criterion(y);rows=[]
 for generator,evaluation in [(0,1),(1,0),(0,0),(1,1)]:
  source=caches[generator];target=caches[evaluation];weights=source['weights']
  assert all(torch.equal(a,b) for a,b in zip(weights,target['weights']))
  virtual={}
  for kind,key in [('include_DN','gradient_total_last'),('exclude_DN_last','gradient_total_without_DN_last')]:
   global_norm=source['global_grad_norm_include_DN' if kind=='include_DN' else 'global_grad_norm_exclude_DN_last']
   virtual[kind]=adam_virtual(weights,source[key],source['optimizer_bindings'],clip_factor(global_norm))
  row={'seed':seed,'generator_batch':generator,'evaluation_batch':evaluation,'cross_batch_primary':generator!=evaluation,'clip_include_DN':clip_factor(source['global_grad_norm_include_DN']),'clip_exclude_DN_last':clip_factor(source['global_grad_norm_exclude_DN_last']),'update_difference_norm':norm([a-b for a,b in zip(virtual['include_DN'],virtual['exclude_DN_last'])]),'gradient_DN_norm':norm(source['gradient_DN']),'gradient_ordinary_norm':norm(source['gradient_ordinary']),'groups':{}}
  for hard in [False,True]:
   group='input_anchor_IoU_lt_0_5' if hard else 'all_ordinary'
   before=metrics(target,weights,criterion,hard)
   including=metrics(target,virtual['include_DN'],criterion,hard)
   excluding=metrics(target,virtual['exclude_DN_last'],criterion,hard)
   if before['pairs']==0:
    row['groups'][group]={'pairs':0,'DN_effect':None};continue
   eval_weights=[p.double().detach().requires_grad_() for p in weights]
   eval_loss=head_loss(target,eval_weights,criterion,hard)[0]
   eval_gradient=torch.autograd.grad(eval_loss,eval_weights)
   dot_dn=dot(eval_gradient,source['gradient_DN']);dot_own=dot(eval_gradient,source['gradient_ordinary'])
   dn_effect=including['loss_contribution']-excluding['loss_contribution']
   no_dn_decrease=before['loss_contribution']-excluding['loss_contribution']
   one={'before':before,'include_DN':including,'exclude_DN_last':excluding,'DN_effect':dn_effect,'positive_means_DN_locally_harmful':True,'decrease_excluding_DN_with_historical_Adam_momentum':no_dn_decrease,'DN_effect_over_excluding_decrease':dn_effect/no_dn_decrease if no_dn_decrease>0 else None,'eval_ordinary_dot_generator_DN':dot_dn,'eval_ordinary_dot_generator_ordinary':dot_own,'SGD_DN_direction_harmful_if_dot_negative':dot_dn<0,'SGD_harm_to_ordinary_self_descent_ratio':-dot_dn/dot_own if dot_own>0 else None}
   row['groups'][group]=one
  rows.append(row)
 return rows

def main():
 limits=resource_limits();protocol_path=OUT/'01_锁定资产与诊断口径.json'
 protocol=json.loads(protocol_path.read_text(encoding='utf8'))
 for item in protocol['matching_and_native_source_identity']:assert sha(CODE/item['relative'])==item['sha256']
 tests=synthetic_checks()
 save_json(OUT/'02_实现与运行前核验.json',{'utc':stamp(),'resources':limits,'synthetic_checks':tests,'protocol_sha256':sha(protocol_path),'scripts':{p.name:sha(p) for p in sorted(CODE.glob('*.py'))},'profiling_import_sentinel':'calflops.calculate_flops raises if called; unused optional profiler, no tensor computation replaced','frozen_model_forward_started':False})
 progress(stage='preflight_passed',resources=limits)
 all_rows=[]
 for seed in [1,2]:
  caches,y=capture_seed(seed,protocol)
  rows=analyze_seed(seed,caches,y);all_rows+=rows
  save_json(OUT/f'seed{seed}/counterfactual_results.json',{'seed':seed,'rows':rows})
  progress(seed=seed,stage='seed_complete',cross_batch_rows=2)
  del caches;gc.collect()
 result={'state':'complete','completed_utc':stamp(),'protocol_sha256':sha(protocol_path),'rows':all_rows,'scope':'Four local CPU training-mode frozen forwards; raw online BASE24, original criterion; only copied last-layer tensors virtually updated. Other parameters fixed during loss evaluation. Not AP, not full-model training counterfactual, not long-term harm proof.','resources':limits,'new_training_steps':0,'optimizer_step_calls':0,'remote_operations':False,'heldout_prediction_reads':False}
 save_json(OUT/'03_全部局部反事实原始结果.json',result)
 progress(stage='complete',frozen_forwards=4,training_steps=0)

if __name__=='__main__':
 try:main()
 except BaseException as e:
  failure=OUT/('failure_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
  save_json(failure,{'utc':stamp(),'error':type(e).__name__+': '+str(e),'traceback':traceback.format_exc(),'training_steps':0,'automatic_identical_retry':False})
  raise
