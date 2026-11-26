"""Evaluate the predeclared RGB layouts with two unchanged BASE24 online models."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import pathlib,sys,json,time,gc,copy
ROOT=pathlib.Path(__file__).resolve().parent;WORK=ROOT.parents[1]
sys.path.insert(0,str(WORK/'英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1'))
import common as n
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
torch,np=n.torch,n.np
RGB=ROOT/'62_layout_rgb';OUT=ROOT/'64_frozen_layout_detection'
ARMS=['original','instance_objective','simple_risk','fixed_random','global_tight','neutral']

def count(a):
 if a.shape[1]==0:return 0
 return int((maximum_bipartite_matching(csr_matrix(a.numpy()),perm_type='column')>=0).sum())

def metrics(pred,target,small):
 boxes,logits=pred['pred_boxes'],pred['pred_logits']
 assert boxes.shape==(300,4) and logits.shape==(300,10)
 pair=n.box_iou(n.box_cxcywh_to_xyxy(boxes),n.box_cxcywh_to_xyxy(target['boxes']))[0]
 value,index=torch.topk(logits.sigmoid().flatten(),300)
 q,c=index//10,index%10
 hit=pair[q]*(c[:,None]==target['labels'][None,:])
 result={'GT':len(small),'sub4':int(small.sum()),'N_over_Q':len(small)>300}
 for name,a in [('geo',pair),('class',hit)]:
  for threshold,suffix in [(.5,'05'),(.75,'075')]:
   result[f'all_{name}{suffix}']=count(a>=threshold)
   result[f'sub4_{name}{suffix}']=count(a[:,small]>=threshold)
   result[f'perGT_{name}{suffix}']=(a>=threshold).any(0).tolist()
 for score,label in [(.001,'0001'),(.3,'03')]:
  a=hit*(value[:,None]>=score)
  result[f'sub4_class05_score{label}']=count(a[:,small]>=.5)
 return result

def log(row):
 print(json.dumps(row),flush=True)
 with (OUT/'progress.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')

def main():
 start=time.monotonic();OUT.mkdir(exist_ok=False);resource=n.resource_limits();torch.set_grad_enabled(False)
 assert count(torch.tensor([[True,True],[False,False]]))==1
 assert count(torch.tensor([[True,False],[False,True]]))==2
 rgbresult=json.loads((RGB/'01_results.json').read_text(encoding='utf8'));assert rgbresult['all_RGB_identity_checks_passed']
 corpus={};cases=[]
 for p in sorted(RGB.glob('case*.pt')):
  data=torch.load(p,map_location='cpu',weights_only=False);cases.append(data)
  for arm,image in data['images'].items():
   key=data['metadata'][arm]['input_sha256']
   if key in corpus:assert torch.equal(corpus[key],image)
   else:corpus[key]=image
 keys=sorted(corpus);assert ((len(keys)+3)//4)*2<=48
 protocol=json.loads((WORK/'英文小论文/data/09_DN初始化冻结诊断_20261020_v1/01_锁定资产与诊断口径.json').read_text(encoding='utf8'))
 recipes=json.loads((WORK/'英文小论文/data/08_匹配聚合覆盖探针_20261019_v1/baseline_recipes.json').read_text(encoding='utf-8-sig'))
 rows=[];sources=[];actual_forwards=0;model_equal={}
 n.save_json(OUT/'00_implementation.json',{'script_sha256':n.sha(__file__),'resource':resource,'cases':len(cases),
  'unique_input_images':len(keys),'protocols':{f:n.sha(ROOT/f) for f in ['53_固定曝光布局_冻结检测首检协议.txt','54_阶段2渲染与整体裁紧控制_前向前补充.txt']},
  'RGB_result_sha256':n.sha(RGB/'01_results.json'),'model_backwards':0,'optimizer_steps':0,'new_training':False})
 for seed in [1,2]:
  cp=pathlib.Path(protocol['checkpoints'][str(seed)]['path']);sha=n.sha(cp)
  assert sha==protocol['checkpoints'][str(seed)]['sha256']
  raw=torch.load(cp,map_location='cpu',weights_only=False)
  recipe=copy.deepcopy(recipes[str(seed)]);recipe['HGNetv2']['pretrained']=False
  model=n.build_model(recipe);model.load_state_dict(raw['model'],strict=True);model.eval();del raw;gc.collect()
  before=n.model_hash(model);predictions={}
  for offset in range(0,len(keys),4):
   assert time.monotonic()-start<1200 and n.psutil.virtual_memory().available/2**30>8
   selected=keys[offset:offset+4];images=torch.stack([corpus[k] for k in selected])
   result=model(images);actual_forwards+=1
   assert torch.isfinite(result['pred_boxes']).all() and torch.isfinite(result['pred_logits']).all()
   for i,k in enumerate(selected):predictions[k]={field:result[field][i].clone() for field in ['pred_boxes','pred_logits']}
   log({'seed':seed,'unique_inputs_completed':offset+len(selected),'total_unique':len(keys),'seconds':round(time.monotonic()-start,2)})
  model_equal[str(seed)]=n.model_hash(model)==before;assert model_equal[str(seed)]
  assert n.sha(cp)==sha
  sources.append({'seed':seed,'checkpoint':str(cp),'sha256':sha,'weight_key':'model','eval_mode':True,'strict_loaded':True})
  torch.save(predictions,OUT/f'seed{seed}_unique_predictions.pt')
  for case in cases:
   armrows={}
   for arm in ARMS:
    if arm not in case['images']:continue
    armrows[arm]=metrics(predictions[case['metadata'][arm]['input_sha256']],case['targets'][arm],case['original_sub4'])
   original=armrows['original']
   for arm,r in armrows.items():
    for name in ['geo05','geo075','class05','class075']:
     a=torch.tensor(original[f'perGT_{name}']);b=torch.tensor(r[f'perGT_{name}'])
     r[f'new_{name}']=int((~a&b).sum());r[f'lost_{name}']=int((a&~b).sum())
   rows.append({'model_seed':seed,'augmentation_seed':case['event']['seed'],'position':case['event']['position'],'arms':armrows})
  del predictions,model;gc.collect()
 totals={}
 for seed in [1,2]:
  totals[str(seed)]={}
  for arm in ARMS:
   rr=[r['arms'][arm] for r in rows if r['model_seed']==seed and arm in r['arms']]
   totals[str(seed)][arm]={'cases':len(rr)}
   for k in rr[0] if rr else []:
    if not isinstance(rr[0][k],list):totals[str(seed)][arm][k]=sum(r[k] for r in rr)
 decision={}
 for seed,t in totals.items():
  c=t['instance_objective'];comparisons={a:c['sub4_class05']>t[a]['sub4_class05'] for a in ['original','simple_risk','fixed_random']}
  decision[seed]={'strict_sub4_class05':comparisons,'all_geo05_not_lower':c['all_geo05']>=t['original']['all_geo05']}
  decision[seed]['passed']=all(comparisons.values()) and decision[seed]['all_geo05_not_lower']
 n.save_json(OUT/'01_rows.json',rows)
 n.save_json(OUT/'02_results.json',{'state':'complete','sources':sources,'totals':totals,'decision':decision,
  'both_passed':all(d['passed'] for d in decision.values()),'model_state_unchanged':model_equal,'actual_batch_forwards':actual_forwards,
  'unique_images_per_model':len(keys),'case_counts_by_augmentation_seed':{str(s):sum(c['event']['seed']==s for c in cases) for s in [1,2]},
  'model_backwards':0,'optimizer_steps':0,'new_training':False,'heldout_reads':False,'seconds':time.monotonic()-start,
  'scope':'Frozen inference on a new training-data development panel; grouped coverage, not AP or training transfer. Neutral NA events not scored as zero.'})
 n.save_json(OUT/'03_SHA256.json',{p.name:n.sha(p) for p in OUT.iterdir() if p.is_file()})
 log({'state':'complete','decision':decision,'forwards':actual_forwards,'seconds':time.monotonic()-start})

if __name__=='__main__':main()
