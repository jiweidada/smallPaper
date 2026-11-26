"""Fixed-last CPU validation, 317 section7. No optimizer or global forward."""
import os,sys,importlib,json,time,copy,math
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
HERE=Path(__file__).resolve().parent;WORK=HERE.parent
sys.path.insert(0,str(WORK/'348_cpu_training_20261023_v1'))
C=importlib.import_module('348_common')
sys.path.insert(0,str(C.S347));T347=importlib.import_module('train347');T=T347.T
sys.path.insert(0,str(WORK/'343_codex_stage4_3b_20261022_v1'))
M=importlib.import_module('343_formal_counts_v3');F=M.F;G=M.G
E=importlib.import_module('343_evaluator');BN=importlib.import_module('343_bootstrap_natural')
B=BN.B;U=BN.U;D=importlib.import_module('343_direction_rule')
GE=importlib.import_module('g_effect_342')
import torch,numpy as np
torch.set_num_threads(2)
def write(name,obj):C.write(HERE/name,obj)
def say(s):print(s,flush=True)
def main():
 start=time.perf_counter();C.plan();seal=C.verify_seal(C.HERE/'348_交付封存SHA256.json')
 training=C.read(C.HERE/'348_训练_result.json');assert len(training['runs'])==8
 sources={str(p):C.sha(p) for p in [Path(__file__),*list((WORK/'343_codex_stage4_3b_20261022_v1').glob('*.py')),WORK/'342_codex_pretrain_20261022_v1/g_effect_342.py']}
 checked=[]
 for r in training['runs']:
  C.verify_seal(Path(r['run_dir']).parent/'348_run_封存SHA256.json')
  assert r['step']==1218 and r['full_budget_completed']
  rec=C.read(r['final_checkpoint']['path']);C.verify_record(rec['checkpoint_path'],rec['checkpoint_sha256'])
  cp=torch.load(rec['checkpoint_path'],map_location='cpu',weights_only=False)
  assert (cp['step'],cp['epoch'],cp['batch_index'])==(1218,6,0) and len(cp['step_logs'])==1218
  assert T347.state_sha(cp['model'])==r['final_model_state_sha256']
  checked.append({'run':r['run_id'],'checkpoint':rec,'model_sha256':r['final_model_state_sha256']})
 write('349_八run固定末轮复核.json',{'status':'已验证','seal':seal,'runs':checked,'sources':sources})
 say('八run封存、1218步及固定末轮权重复核通过')
 ann=Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val\annotations\val.json')
 dataset=C.read(ann);assert len(dataset['images'])==548
 by={}
 for a in dataset['annotations']:by.setdefault(a['image_id'],[]).append(a)
 manifest=C.read(WORK/'342_codex_stage5_review_20261022_v1/342_活动缓存7566_逐条SHA总清单.json')
 val={(r['seed'],r['image_id']):r for r in manifest['entries'] if r['split']=='val'}
 if not val:val={(r['seed'],r['image_id']):r for r in manifest['entries'] if r['split']!='development'}
 assert len(val)==1096
 allmetrics={};allcounts={};geffects={};prepared={};bcounts={};proof=[];geomerrors=[];pngs=[]
 for seed in (1,2):
  models={}
  for arm in 'NSGR':
   r=next(r for r in training['runs'] if r['run_id']==f'{arm}_seed{seed}')
   cp=torch.load(r['checkpoint_path'],map_location='cpu',weights_only=False)
   model=T.build_arm(arm,seed).eval();model.load_state_dict(cp['model'],strict=True);models[arm]=model
  predictions={a:[] for a in ['baseline',*'NSGR']};counts={a:[] for a in predictions};gd=[];gr=[]
  with torch.inference_mode():
   for iid in range(548):
    rec=val[seed,iid];C.verify_record(rec['path'],rec['file_sha256'])
    e=torch.load(rec['path'],map_location='cpu',weights_only=True);assert e['annotation_sha256']==C.sha(ann)
    e['fullp_nonpositive_mask']=T.geom.fullp_nonpositive_mask(e['final300_boxes'],e['fullp_query_index'])
    assert not e['fullp_nonpositive_mask'].any()
    batch=T.collate([e]);outs={};pres={};outs['baseline'],pres['baseline']=F.fuse(e)
    for arm,model in models.items():
     out=model(batch['rgb'],batch['P_boxes'],batch['P_cls'],batch['P_scores'],batch['P_valid'])
     local=out['pred_boxes'][0];mapped=G.local_cxcywh_to_orig_xyxy(local,e['window_xyxy_640'],e['orig_wh'])
     # Independent NumPy center/extent map, with no shared conversion helper.
     q=local.numpy().astype(np.float64);w=e['window_xyxy_640'].numpy().astype(np.float64);wh=np.asarray(e['orig_wh'],dtype=float)
     xy=np.column_stack((q[:,0]-q[:,2]/2,q[:,1]-q[:,3]/2,q[:,0]+q[:,2]/2,q[:,1]+q[:,3]/2))
     ref=(xy*np.array([w[2]-w[0],w[3]-w[1]]*2)+np.array([w[0],w[1]]*2))*np.tile(wh/640,2)
     err=float(np.max(np.abs(mapped.numpy()-ref)));assert err<.002;geomerrors.append(err)
     back=G.orig_xyxy_to_local_cxcywh(mapped,e['window_xyxy_640'],e['orig_wh']);assert torch.allclose(back,local,atol=2e-5,rtol=0)
     outs[arm],pres[arm]=F.fuse(e,local,out['pred_logits'][0])
     if arm=='G':
      _,safe,classes,_=T.head342.core.candidate_inputs(batch['P_boxes'],batch['P_cls'],batch['P_scores'],batch['P_valid'])
      measure=GE.measure(out['mass'],safe,classes,batch['P_valid'],model.class_null)
      gd.extend(measure['per_window_delta']);gr.extend(measure['residual_over_mass'])
     if arm=='R' and iid in (0,1,5):
      from PIL import Image,ImageDraw
      img=Image.open(Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val\images')/e['file_name']).convert('RGB')
      draw=ImageDraw.Draw(img)
      for box in mapped.tolist():draw.rectangle(box,outline='red',width=2)
      name=f'349_R_seed{seed}_{iid:06d}.png'
      with open(HERE/name,'xb') as f:img.save(f,format='PNG')
      pngs.append(str(HERE/name))
    items=by.get(iid,[]);aids=[a['id'] for a in items];assert aids==e['gt_annotation_ids']
    gt=G.orig_xywh_to_orig_xyxy(torch.tensor([a['bbox'] for a in items],dtype=torch.float64).reshape(-1,4))
    mx=G.pairwise_iou(G.raw_query_cxcywh_to_640_xyxy(e['final300_boxes']),e['gt_boxes_640'],allow_raw_a=True).max(0).values.numpy()
    gw=e['G_W'].numpy();tiny=e['gt_short_side_640_all'].numpy()<4
    groups={'tiny_complete_Geo05':(mx<.5)&gw&tiny,'all_image_Geo05':mx<.5,'all_sizes_complete_Geo05':(mx<.5)&gw,'tiny_complete_Geo075':(mx<.75)&gw&tiny}
    records={a:{} for a in outs}
    for th in (.5,.75):
     owners={a:M.match(o,gt,e['gt_labels'],aids,th) for a,o in outs.items()}
     for a,o in outs.items():
      assert np.array_equal(owners[a],M.independent_match(o,gt,e['gt_labels'],aids,th))
      records[a][str(th)]=M.metrics(o,owners[a],outs['baseline'],owners['baseline'],groups,aids,pres[a],num_new_instances=0 if a=='baseline' else 40)
    for a,o in outs.items():
     predictions[a].extend(F.coco_rows(iid,o));counts[a].append({'image_id':iid,'counts':records[a]})
    if (iid+1)%50==0:say(f'val seed{seed} {iid+1}/548: 四臂前向、映射、独立匹配通过')
  delta=float(np.mean(gd));geffects[str(seed)]={'delta':delta,'verdict':GE.classify(delta),'per_window_delta':gd,'residual_over_mass':gr,'kappa_final':models['G'].class_null.tolist(),'kappa_displacement':models['G'].class_null.tolist()}
  allmetrics[str(seed)]={};allcounts[str(seed)]=counts
  for a,rows in predictions.items():
   name=f'seed{seed}_{a}';write(f'349_{name}_predictions.json',rows)
   ev,stats,log,_=E.evaluate(dataset,rows);C.text(HERE/f'349_{name}_COCO.txt',log)
   totals={k:sum(r['counts']['0.5'][k] for r in counts[a]) for k,v in counts[a][0]['counts']['0.5'].items() if isinstance(v,int)}
   allmetrics[str(seed)][a]={**stats,**totals};say(f'{name}: AP={stats["AP"]} APs={stats["APs"]} B={totals["B"]}')
   # Positive annotation IDs for bootstrap to avoid COCO zero-match sentinel.
   ds=B.positive_gt_ids(copy.deepcopy(dataset));bev,_,_,_=E.evaluate(ds,rows);p=BN.prepare(bev);prepared[name]=p
   ix=np.random.default_rng(2026102103).integers(0,548,size=(1000,548),dtype=np.int64)
   for index in (0,999):
    fast=BN.fast_ap(p,ix[index]);native=B.ap_aps(B.sampled_evaluator(bev,ix[index],ap_only=True));assert fast==native
    proof.append({'case':name,'replica':index,'native_equals_fast':True})
   if a=='baseline':
    ds2,ps2=U.independent_clone(ds,rows,ix[0]);fresh,_,_,_=E.evaluate(ds2,ps2);assert B.ap_aps(fresh)==BN.fast_ap(p,ix[0])
   bcounts[name]=np.array([r['counts']['0.5']['B'] for r in counts[a]])
  write(f'349_seed{seed}_正式逐图计数.json',counts)
 write('349_val548指标与G校准.json',{'metrics':allmetrics,'G_effect':geffects,'evaluator':E.evidence(),'geometry_independent_max_abs_error':max(geomerrors),'png_paths':pngs})
 # Reverse format check against the frozen geometry validator.
 try:G.orig_xywh_to_orig_xyxy(torch.tensor([[0.,0.,-1.,1.]]))
 except AssertionError:pass
 else:raise AssertionError('wrong coordinate format not rejected')
 with open(HERE/'349_bootstrap共享索引.npy','xb') as f:np.save(f,ix)
 replicas={k:[] for k in prepared}
 for index,sample in enumerate(ix):
  for name,p in prepared.items():replicas[name].append(BN.fast_ap(p,sample)+[int(bcounts[name][sample].sum())])
  if (index+1)%10==0:say(f'paired bootstrap {index+1}/1000')
 ci={};diff={}
 for name,values in replicas.items():ci[name]=np.percentile(values,[2.5,97.5],axis=0).tolist()
 for seed in (1,2):
  for a in ('baseline','N','S','G'):
   difference=np.array(replicas[f'seed{seed}_R'])-np.array(replicas[f'seed{seed}_{a}'])
   diff[f'seed{seed}_R_minus_{a}']=np.percentile(difference,[2.5,97.5],axis=0).tolist()
 write('349_bootstrap逐次AP_APs_B.json',replicas)
 write('349_bootstrap_result.json',{'seed':2026102103,'replicates':1000,'metrics_order':['AP','APs','B'],'percentile95':ci,'paired_difference_percentile95':diff,'native_equivalence_proof':proof,'method':'Shared548 image replicas; global stable score merge and precision envelope recomputed, duplicate image identity cloned; never average per-image AP.'})
 judge=D.judge(allmetrics,{s:g['delta']>=.01 for s,g in geffects.items()})
 judge['G_effect']=geffects;judge['G_v4_evidence_rule']='delta<.01: R-G四项不得支持R；.01至.03校准能力受限；>=.03正式对照。'
 write('349_十二项判定_result.json',judge)
 C.verify_seal(C.HERE/'348_交付封存SHA256.json')
 for p,h in sources.items():C.verify_record(p,h)
 write('349_评估_result.json',{'status':'已完成，封存后停下待主控','metrics':allmetrics,'decision':judge,'wall_seconds':time.perf_counter()-start,'head_image_exposures':4384,'global_forwards':0,'optimizer_steps':0,'heldout_read':False,'sources':sources})
 lines=['# 349 固定末轮 CPU 评估','',judge['direction_statement'],'','| seed | arm | AP | APs | B |','|---|---|---:|---:|---:|']
 for s,arms in allmetrics.items():
  for a,m in arms.items():lines.append(f'| {s} | {a} | {m["AP"]:.8f} | {m["APs"]:.8f} | {m["B"]} |')
 lines+=['','G Δ、全部12项方向、总体AP差和配对区间分别见对应result JSON。无调参、无加轮、无best选择、未读heldout3236。','已封存并停下，等待主控核对。']
 C.text(HERE/'349_一页中文汇总.md','\n'.join(lines))
 say(C.seal_tree(HERE,'349_交付封存SHA256.json'))
if __name__=='__main__':main()
