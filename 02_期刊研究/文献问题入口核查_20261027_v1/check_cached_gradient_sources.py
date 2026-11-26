"""Local derivative of unchanged native MAL on fixed cached outputs. No autograd."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import pathlib,json,hashlib,ast,types,time,traceback,importlib.util,math
import numpy as np
import torch
import torch.nn.functional as F
BASE=pathlib.Path(__file__).resolve().parent;ROOT=BASE.parent
DATA=ROOT/'data/08_匹配聚合覆盖探针_20261019_v1';CACHE=DATA/'isolated_v3'
PARTNER=ROOT/'讨论伙伴_候选保留主线取证_20261027_v1/文献成立条件取证_20261027_v1'
CODE=ROOT/'code/dn_pre_frozen_diagnostic_20261020_v1/native/engine/deim'
PROTO=BASE/'08_频次解释_缓存局部梯度来源_冻结协议.json';OUT=BASE/'gradient_sources_attempt01'
LOW=[7,6,8];HIGH=[3,0,9]
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(p,x):pathlib.Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def ratio(a,b):return a/b if b>0 else None
def math_controls():
 # Actual stopped-weight derivative: freeze weight at center before varying z.
 results=[]
 for z,t,w in [(-8.,0.,.01),(-1.,.3,1.),(2.,.2,1.),(0.,1.,1.)]:
  def loss(x):return w*(max(x,0)-x*t+math.log1p(math.exp(-abs(x))))
  h=1e-5;finite=(loss(z+h)-loss(z-h))/(2*h);analytic=w*(1/(1+math.exp(-z))-t)
  assert abs(finite-analytic)<1e-9
  results.append({'z':z,'target':t,'fixed_weight':w,'finite_difference':finite,'analytic':analytic})
 return results
def combine(rows,classes):
 sums={k:sum(r[k] for r in rows if r['class'] in classes) for k in ['support','self_down','cross_down','head_cross_down','other_cross_down','background_down','positive_queries','target_instances','unmatched_target_instances','negative_foreground_queries','background_queries']}
 sums['class_ids']=classes;sums['cross_over_support']=ratio(sums['cross_down'],sums['support']);sums['all_down_over_support']=ratio(sums['self_down']+sums['cross_down']+sums['background_down'],sums['support'])
 return sums
def main():
 start=time.monotonic();OUT.mkdir(exist_ok=False);torch.set_num_threads(2);torch.set_grad_enabled(False)
 schema=read(PARTNER/'BASE24_schema_preflight.json');semantic=read(PARTNER/'BASE24_semantics_preflight.json')
 bindings={str(PROTO):sha(PROTO),str(pathlib.Path(__file__)):sha(__file__),str(PARTNER/'BASE24_schema_preflight.json'):sha(PARTNER/'BASE24_schema_preflight.json'),str(PARTNER/'BASE24_semantics_preflight.json'):sha(PARTNER/'BASE24_semantics_preflight.json')}
 for p,h in schema['source_hashes'].items():assert sha(p)==h;bindings[p]=h
 for rel,r in semantic['native_source_checks'].items():
  p=CODE/pathlib.Path(rel).name;assert sha(p)==r['expected'];bindings[str(p)]=sha(p)
 recipe=read(DATA/'baseline_recipes.json')
 spec=importlib.util.spec_from_file_location('original_box_ops_for_local_signal',CODE/'box_ops.py');box=importlib.util.module_from_spec(spec);spec.loader.exec_module(box)
 source=(CODE/'deim_criterion.py').read_text(encoding='utf8');tree=ast.parse(source)
 clazz=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DEIMCriterion')
 wanted=['_get_src_permutation_idx','_get_vfl_weight_warmup_factor','_vfl_weighted_bce_loss','loss_labels_mal']
 nodes=[n for n in clazz.body if isinstance(n,ast.FunctionDef) and n.name in wanted];assert len(nodes)==4
 module=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]));env={'torch':torch,'F':F,'box_iou':box.box_iou,'box_cxcywh_to_xyxy':box.box_cxcywh_to_xyxy}
 exec(compile(module,str(CODE/'deim_criterion.py'),'exec'),env)
 dummy=types.SimpleNamespace(num_classes=10,gamma=1.5,mal_alpha=None,no_weight_vfl_epoch=-1,no_weight_vfl_warmup_iter=500,epoch=24)
 for name in wanted:setattr(dummy,name,types.MethodType(env[name],dummy))
 assert dummy._get_vfl_weight_warmup_factor()==1.
 result={'scope':read(PROTO)['population'],'math_controls':math_controls(),'native_methods_AST_extracted_without_model_construction':wanted,'native_method_lines':{n.name:[n.lineno,n.end_lineno] for n in nodes},'inputs':bindings,'seeds':{},'new_model_forwards':0,'model_constructions':0,'checkpoint_loads':0,'autograd_backward_VJP':0,'optimizer_steps':0,'heldout_reads':0}
 for seed,letter in [(1,'A'),(2,'B')]:
  cfg=recipe[str(seed)]['DEIMCriterion'];assert cfg['gamma']==1.5 and cfg.get('mal_alpha') is None and cfg.get('no_weight_vfl_epoch',-1)==-1
  lossweight=cfg['weight_dict']['loss_mal'];per=[];batchinfo=[]
  for batch in range(8):
   p=CACHE/letter/'frozen_predictions/development'/f'batch{batch:02d}_predictions.pt';a=CACHE/letter/'development'/f'batch{batch:02d}_analysis.json';prior=read(a)
   assert sha(p)==prior['prediction_sha256']==schema['source_hashes'][str(p)]
   value=torch.load(p,map_location='cpu',weights_only=True);o=value['outputs'];targets=value['targets']
   assert value['group']==prior['group']=='development' and value['tag']==prior['tag']==f'batch{batch:02d}' and prior['seed']==seed
   z=o['pred_logits'];pred=o['pred_boxes'];assert z.shape==(4,300,10) and pred.shape==(4,300,4) and not z.requires_grad and z.device.type=='cpu' and z.dtype==torch.float32
   indices=[tuple(torch.tensor(x,dtype=torch.int64) for x in r['final']) for r in prior['branch_matches']]
   owner=torch.full((4,300),-1,dtype=torch.int64);ious=torch.zeros((4,300),dtype=torch.float32);GTcounts=torch.zeros(10,dtype=torch.int64)
   for i,(t,(q,g)) in enumerate(zip(targets,indices)):
    assert t['boxes'].shape==(len(t['labels']),4) and len(q)==len(g)==len(set(q.tolist()))==len(set(g.tolist()))
    assert len(t['labels'])==prior['inputs'][i]['GT'] and int(q.min())>=0 and int(q.max())<300 and int(g.min())>=0 and int(g.max())<len(t['labels'])
    GTcounts+=torch.bincount(t['labels'],minlength=10)
    owner[i,q]=t['labels'][g]
    v=box.box_iou(box.box_cxcywh_to_xyxy(pred[i,q]),box.box_cxcywh_to_xyxy(t['boxes'][g]))[0].diag()
    assert torch.isfinite(v).all() and (v>=0).all() and (v<=1).all();ious[i,q]=v
   N=max(sum(len(t['labels']) for t in targets),1);pval=z.sigmoid();tgt=torch.zeros_like(z);weight=pval.pow(1.5)
   idx=dummy._get_src_permutation_idx(indices);labels=owner[idx]
   tgt[idx[0],idx[1],labels]=ious[idx].pow(1.5);weight[idx[0],idx[1],labels]=1.
   native=float(dummy.loss_labels_mal(o,targets,indices,N)['loss_mal'])*lossweight
   direct=float(F.binary_cross_entropy_with_logits(z,tgt,weight=weight,reduction='none').mean(1).sum()*z.shape[1]/N)*lossweight
   saved=prior['batch_losses']['native_union']['losses']['loss_mal'];tol=1e-5+1e-5*abs(saved)
   assert abs(native-saved)<=tol and abs(direct-native)<=1e-7,('native_scalar_fidelity',seed,batch,native,saved,direct)
   gradraw=weight*(pval-tgt);assert torch.isfinite(gradraw).all()
   rows=[]
   for c in range(10):
    own=owner==c;other=(owner>=0)&(owner!=c);head=other&((owner==3)|(owner==0)|(owner==9));bg=owner<0;g=gradraw[:,:,c]
    assert (g[other|bg]>=0).all()
    support=float((-g[own]).clamp_min(0).double().sum());self_down=float(g[own].clamp_min(0).double().sum());cross=float(g[other].double().sum());hd=float(g[head].double().sum());bd=float(g[bg].double().sum())
    raw={'support':support,'self_down':self_down,'cross_down':cross,'head_cross_down':hd,'other_cross_down':cross-hd,'background_down':bd}
    normalized={k:v*lossweight/N for k,v in raw.items()}
    row={'seed':seed,'batch':batch,'class':c,'num_boxes':N,'loss_mal_weight':lossweight,'positive_queries':int(own.sum()),'target_instances':int(GTcounts[c]),'unmatched_target_instances':int(GTcounts[c])-int(own.sum()),'negative_foreground_queries':int(other.sum()),'background_queries':int(bg.sum()),'raw_mass':raw,**normalized}
    assert abs(float(g.double().sum())-(-support+self_down+cross+bd))<1e-8
    rows.append(row);per.append(row)
   assert sum(r['positive_queries'] for r in rows)==int((owner>=0).sum())
   batchinfo.append({'batch':batch,'num_boxes':N,'ordinary_positive_queries':int((owner>=0).sum()),'ordinary_unmatched_queries':int((owner<0).sum()),'native_scalar':native,'saved_native_scalar':saved,'absolute_scalar_difference':abs(native-saved),'scalar_tolerance_frozen':tol,'source_ids':prior['source_ids']})
  groups={name:combine(per,ids) for name,ids in [('low3',LOW),('high3',HIGH),('all10',list(range(10)))]+[(f'class{c}',[c]) for c in range(10)]}
  low,high=groups['low3'],groups['high3'];gate=bool(low['cross_over_support'] is not None and high['cross_over_support'] is not None and low['cross_over_support']>high['cross_over_support'] and low['head_cross_down']>low['background_down'] and low['head_cross_down']>low['self_down'])
  result['seeds'][str(seed)]={'groups':groups,'local_compatibility_screen':gate,'batches':batchinfo,'effective_gamma':1.5,'effective_mal_alpha':None,'weighted_warmup_factor':1.,'recipe_alpha_not_used_in_MAL':cfg['alpha'],'normalization':'own final, max(all augmented GT,1),not GO union; target.mixup not read by native branch'}
  save(OUT/f'seed{seed}_class_batch_rows.json',per)
 for p,h in bindings.items():assert sha(p)==h,('input_changed',p)
 result['both_local_screens_pass']=all(s['local_compatibility_screen'] for s in result['seeds'].values());result['seconds']=time.monotonic()-start
 save(OUT/'results.json',result);save(OUT/'sha256.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
 print(json.dumps({'seeds':{s:{'low3':r['groups']['low3'],'high3':r['groups']['high3'],'screen':r['local_compatibility_screen']} for s,r in result['seeds'].items()},'both_pass':result['both_local_screens_pass'],'seconds':result['seconds']}))
if __name__=='__main__':
 try:main()
 except Exception as e:
  if OUT.exists():save(OUT/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'scientific_result':'NA'})
  raise
