"""One bounded forward pass over fixed inputs; physically separate sealed outputs.
No matching/loss analysis is performed here, including for heldout outputs.
"""
import argparse,datetime,hashlib,json,os,pathlib,random,subprocess,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'probe'),str(ROOT/'v3')]
import numpy as np
import torch
from PIL import Image
from probe_runtime import verify_package,build_model,cpu_tree
from seal_v3 import sha,write_json,verify_files
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def model_sha(model):
 h=hashlib.sha256()
 for n,t in model.state_dict().items():h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main(seed):
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 verify_package();revision=json.loads((ROOT/'v3/manifest.json').read_text());verify_files(ROOT,revision)
 plan=json.loads((ROOT/'sample_plan.json').read_text());y=json.loads((ROOT/'baseline_recipes.json').read_text())[str(seed)]
 cp=ROOT/'assets'/f'baseline24_seed{seed}.pth';assert sha(cp)==plan['checkpoints'][str(seed)]['sha256']
 inputs=json.loads((ROOT/'inputs/manifest.json').read_text());assert inputs['formal_inputs']==64 and inputs['plan_sha256']==sha(ROOT/'sample_plan.json')
 assert torch.cuda.is_available() and torch.cuda.device_count()==1
 busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 if busy:
  old=pathlib.Path(os.environ['PROBE_OLD_ROOT']);plan_old=json.loads((old/'plan.json').read_text())
  assert len(busy.splitlines())<=3,'Unexpected extra GPU task; prioritize UAVDT'
  for arm in plan_old['jobs']:
   job=f'uavdt_{arm}_full96_seed{plan_old["seed"]}';r=json.loads((old/'jobs'/(job+'.json')).read_text())
   assert r['state']=='running'
   p=pathlib.Path('/proc')/str(r['pid']);s=(p/'stat').read_text().rsplit(')',1)[1].split()
   assert s[19]==str(r['start_ticks']) and s[0]!='Z'
  free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
  assert free>=6144,'Less than6GiB free; do not interrupt UAVDT for this background probe'
  torch.cuda.set_per_process_memory_fraction(.16)
 outdir=ROOT/f'isolated_results_seed{seed}';outdir.mkdir(exist_ok=False)
 for group in ['smoke','development','heldout']:(outdir/group).mkdir()
 raw=torch.load(cp,map_location='cpu',weights_only=False);assert raw['last_epoch']==23
 model=build_model(y);model.load_state_dict(raw['model'],strict=True);del raw
 model.requires_grad_(False);model.train();before=model_sha(model);model.cuda()
 saved={n:b.detach().clone() for n,b in model.named_buffers()};events=[];records={g:[] for g in ['smoke','development','heldout']};started=time.monotonic()
 def restore():
  with torch.no_grad():
   for n,b in model.named_buffers():b.copy_(saved[n])
 def forward(images,targets,tag,group,dn_seed,metadata):
  restore();rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
  random.seed(dn_seed);np.random.seed(dn_seed);torch.manual_seed(dn_seed);torch.cuda.manual_seed_all(dn_seed)
  gt=[{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in targets]
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
  with torch.no_grad():output=model(images.cuda(),gt)
  torch.cuda.synchronize();seconds=time.perf_counter()-t;peak=torch.cuda.max_memory_allocated()
  frozen={'outputs':cpu_tree(output),'targets':cpu_tree(targets),'tag':tag,'group':group,'forward_DN_seed':dn_seed}
  restore();random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
  assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
  path=outdir/group/(tag+'_predictions.pt')
  with path.open('xb') as f:torch.save(frozen,f);f.flush();os.fsync(f.fileno())
  row={'relative':path.name,'bytes':path.stat().st_size,'sha256':sha(path),'generated_utc':now(),**metadata}
  path.chmod(0o444);records[group].append(row)
  with (outdir/'generation_seals.jsonl').open('a') as f:f.write(json.dumps({'group':group,**row})+'\n');f.flush();os.fsync(f.fileno())
  event={'tag':tag,'group':group,'forward_seconds':seconds,'peak_allocated_MiB':peak/2**20,'output_sha256':row['sha256']}
  events.append(event);temp=outdir/'progress.next';temp.write_text(json.dumps({'finished_batches':len(events),'last':event,'elapsed_seconds':time.monotonic()-started}));os.replace(temp,outdir/'progress.json')
  print(json.dumps(event),flush=True)
 for bi in range(2):
  images=[];targets=[]
  for k in range(4*bi,4*bi+4):
   z=np.load(ROOT/'smoke'/f'{k:03d}_targets.npz')
   images.append(torch.from_numpy(np.asarray(Image.open(ROOT/'smoke'/f'{k:03d}.png').convert('RGB')).copy()).permute(2,0,1).float()/255)
   targets.append({'boxes':torch.from_numpy(z['boxes'].copy()).float(),'labels':torch.from_numpy(z['labels'].copy()).long()})
  forward(torch.stack(images),targets,f'smoke{bi:02d}','smoke',39261019+bi,{'batch':bi,'source_ids':[]})
 for item in inputs['batches']:
  p=ROOT/'inputs'/item['file'];assert sha(p)==item['sha256'];v=torch.load(p,map_location='cpu',weights_only=False)
  assert hashlib.sha256(v['images'].numpy().tobytes()).hexdigest()==item['float_input_sha256']
  forward(v['images'],v['targets'],f"batch{item['batch']:02d}",item['group'],v['spec']['forward_DN_seed'],{'batch':item['batch'],'source_ids':item['source_ids'],'input_sha256':item['sha256']})
  # Seal a group as soon as its final batch has been generated.
  group=item['group']
  if len(records[group])==8:
   write_json(outdir/group/'seal.json',{'group':group,'batches':8,'inputs':32,'files':records[group],'sealed_utc':now(),'outcome_analysis_performed':False})
   (outdir/group/'seal.json').chmod(0o444);(outdir/group).chmod(0o555)
 write_json(outdir/'smoke/seal.json',{'group':'smoke','batches':2,'inputs':8,'files':records['smoke'],'sealed_utc':now(),'outcome_analysis_performed':False})
 restore();model.cpu();assert model_sha(model)==before and sha(cp)==plan['checkpoints'][str(seed)]['sha256'];verify_package();verify_files(ROOT,revision)
 write_json(outdir/'complete.json',{'state':'complete','seed':seed,'formal_inputs':64,'smoke_inputs':8,'optimizer_steps':0,'model_gradients':0,'BN_and_state_restored':True,'checkpoint_sha256':sha(cp),'plan_sha256':sha(ROOT/'sample_plan.json'),'revision_manifest_sha256':sha(ROOT/'v3/manifest.json'),'partition_seal_sha256':{g:sha(outdir/g/'seal.json') for g in records},'elapsed_seconds':time.monotonic()-started,'completed_utc':now(),'torch':torch.__version__,'GPU':torch.cuda.get_device_name(0),'outcome_analysis_performed':False,'shared_GPU_with_existing_UAVDT':bool(busy),'timings_not_used_as_exclusive_latency_measurement':True})
 print(json.dumps({'state':'complete','seed':seed,'formal_inputs':64,'optimizer_steps':0}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[1,2],required=True);a=p.parse_args()
 try:main(a.seed)
 except BaseException as e:
  d=ROOT/f'isolated_results_seed{a.seed}';d.mkdir(exist_ok=True)
  write_json(d/'failure.json',{'error':type(e).__name__+': '+str(e),'traceback':traceback.format_exc(),'automatic_retry':False});raise
