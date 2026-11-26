"""Same-panel all-count control, reusing all cached identical inputs."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import pathlib,sys,json,importlib.util,time,copy,gc
ROOT=pathlib.Path(__file__).resolve().parent;WORK=ROOT.parents[1]
sys.path.insert(0,str(WORK/'英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1'))
import common as n
torch,np=n.torch,n.np
OUT=ROOT/'77_all_count_frozen'

def module(name,file):
 s=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 start=time.monotonic();OUT.mkdir(exist_ok=False);resource=n.resource_limits();torch.set_grad_enabled(False)
 renderer=module('render61','61_prepare_layout_rgb.py');detector=module('detect63','63_frozen_layout_detection.py')
 newgeo=json.loads((ROOT/'74_layout_counterexamples/01_all_count_geometry.json').read_text(encoding='utf8'))
 geo={(r['seed'],r['position']):r for r in newgeo['rows']}
 traces=json.loads((ROOT/'49_augmentation_replay_audit_v4/01_transform_traces.json').read_text(encoding='utf8'))
 events={(e['seed'],e['position']):e for e in traces}
 tags=json.loads((ROOT/'49_augmentation_replay_audit_v4/05_tagged_source_identity.json').read_text(encoding='utf8'))
 ids={(t['seed'],t['position']):t for t in tags}
 cases=[];corpus={}
 for p in sorted((ROOT/'62_layout_rgb').glob('case*.pt')):
  case=torch.load(p,map_location='cpu',weights_only=False);key=case['event']['seed'],case['event']['position'];row=geo[key]
  if not row['different_from_risk']:
   image=case['images']['simple_risk'];target=case['targets']['simple_risk']
  else:
   event=events[key];ident=ids[key];rgb,mask=renderer.restore(event)
   rr={'pieces':row['pieces'],'arms':{'all_count':{'layout':row['layout'],'scales':row['scales']}}}
   scale,shift,cell=renderer.mapping('all_count',rr,None)
   if row['layout']=='original':image=case['images']['original']
   else:
    made,_=renderer.render(rgb,mask,scale,shift,cell);image=renderer.to_input(made)
   boxes=np.array(ident['boxes'],dtype=np.float64).reshape(-1,4);slots=np.array([i['source_slot'] for i in ident['instances']])
   changed=renderer.transformed(boxes,slots,scale,shift)
   assert ((changed[:,:2]-changed[:,2:]/2)>-1e-5).all() and ((changed[:,:2]+changed[:,2:]/2)<1+1e-5).all()
   m=boxes[:,2:].min(1)*640;new=changed[:,2:].min(1)*640
   assert int(((m<4)&(new>=4)).sum())==row['all_count_cross4']
   target={'boxes':torch.tensor(changed,dtype=torch.float32),'labels':case['targets']['original']['labels']}
  fingerprint=renderer.tensor_hash(image)
  if fingerprint in corpus:assert torch.equal(corpus[fingerprint],image)
  else:corpus[fingerprint]=image
  cases.append({'event':case['event'],'input_sha256':fingerprint,'target':target,'original_sub4':case['original_sub4']})
 torch.save({'cases':cases,'images':corpus},OUT/'00_all_count_inputs.pt')
 base=json.loads((ROOT/'64_frozen_layout_detection/02_results.json').read_text(encoding='utf8'))
 rawrows=json.loads((ROOT/'64_frozen_layout_detection/01_rows.json').read_text(encoding='utf8'))
 existing={(r['model_seed'],r['augmentation_seed'],r['position']):r for r in rawrows}
 protocol=json.loads((WORK/'英文小论文/data/09_DN初始化冻结诊断_20261020_v1/01_锁定资产与诊断口径.json').read_text(encoding='utf8'))
 recipes=json.loads((WORK/'英文小论文/data/08_匹配聚合覆盖探针_20261019_v1/baseline_recipes.json').read_text(encoding='utf-8-sig'))
 rows=[];total={};decisions={};state_equal={};forwards=0;news={}
 for seed in [1,2]:
  pred=torch.load(ROOT/f'64_frozen_layout_detection/seed{seed}_unique_predictions.pt',map_location='cpu',weights_only=False)
  keys=sorted(k for k in corpus if k not in pred);news[str(seed)]=len(keys);assert len(keys)<=16
  cp=pathlib.Path(protocol['checkpoints'][str(seed)]['path']);cp_sha=n.sha(cp)
  assert cp_sha==protocol['checkpoints'][str(seed)]['sha256']
  if keys:
   raw=torch.load(cp,map_location='cpu',weights_only=False);recipe=copy.deepcopy(recipes[str(seed)]);recipe['HGNetv2']['pretrained']=False
   model=n.build_model(recipe);model.load_state_dict(raw['model'],strict=True);model.eval();del raw;gc.collect();before=n.model_hash(model)
   for off in range(0,len(keys),4):
    assert time.monotonic()-start<1200 and n.psutil.virtual_memory().available/2**30>8
    kk=keys[off:off+4];result=model(torch.stack([corpus[k] for k in kk]));forwards+=1;assert forwards<=8
    for j,k in enumerate(kk):pred[k]={f:result[f][j].clone() for f in ['pred_boxes','pred_logits']}
   state_equal[str(seed)]=n.model_hash(model)==before;assert state_equal[str(seed)];del model;gc.collect()
  else:state_equal[str(seed)]=True
  assert n.sha(cp)==cp_sha
  torch.save({k:pred[k] for k in corpus},OUT/f'seed{seed}_all_count_predictions.pt')
  rr=[]
  for c in cases:
   r=detector.metrics(pred[c['input_sha256']],c['target'],c['original_sub4'])
   key=(seed,c['event']['seed'],c['event']['position']);old=existing[key]['arms']['original']
   for name in ['geo05','geo075','class05','class075']:
    a=np.array(old[f'perGT_{name}']);b=np.array(r[f'perGT_{name}']);r[f'new_{name}']=int((~a&b).sum());r[f'lost_{name}']=int((a&~b).sum())
   rr.append(r);rows.append({'model_seed':seed,'augmentation_seed':c['event']['seed'],'position':c['event']['position'],'all_count':r})
  sums={'cases':len(rr)}
  for k in rr[0]:
   if not isinstance(rr[0][k],list):sums[k]=sum(r[k] for r in rr)
  total[str(seed)]=sums;risk=base['totals'][str(seed)]['simple_risk']
  decisions[str(seed)]={'risk_sub4_class05':risk['sub4_class05'],'all_count_sub4_class05':sums['sub4_class05'],'risk_strictly_higher':risk['sub4_class05']>sums['sub4_class05']}
  del pred;gc.collect()
 n.save_json(OUT/'01_rows.json',rows)
 n.save_json(OUT/'02_results.json',{'state':'complete','resource':resource,'totals':total,'decision':decisions,'retain_sub4_specificity_both':all(d['risk_strictly_higher'] for d in decisions.values()),
  'model_state_unchanged':state_equal,'new_unique_images_per_model':news,'actual_new_batch_forwards':forwards,'model_backwards':0,'optimizer_steps':0,'new_training':False,'heldout_reads':False,
  'checkpoint_sources':base['sources'],'seconds':time.monotonic()-start,'scope':'Prospective 72 control on same development panel, not a full replication of RA-MOSAIC/bird paper, AP, or training transfer.',
  'source_sha256':{f:n.sha(ROOT/f) for f in ['72_固定曝光布局_反例与得失首检协议.txt','75_all_count冻结补充执行.txt','74_layout_counterexamples/01_all_count_geometry.json','64_frozen_layout_detection/02_results.json']},'script_sha256':n.sha(__file__)})
 n.save_json(OUT/'03_SHA256.json',{p.name:n.sha(p) for p in OUT.iterdir() if p.is_file()})
 print(json.dumps({'state':'complete','totals':total,'decision':decisions,'new_forwards':forwards,'seconds':time.monotonic()-start},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
