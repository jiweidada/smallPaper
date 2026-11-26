"""Read-only paired CPU replay of existing baseline states; passive tensor observations."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import sys,json,hashlib,time,argparse
from pathlib import Path
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
torch.set_num_threads(2)
OUT=Path(__file__).resolve().parent;PAIR=OUT.parent;ROOT=PAIR.parent
BASE=ROOT/'data/00_实验计划与证据索引/本次24轮双基线模型下载_20261009_v1'
S3=ROOT/'code/unit_instance_completion_20261021_v1/342_codex_stage3_20261022_v1'
NATIVE=ROOT/'code/dn_pre_frozen_diagnostic_20261020_v1/native'
EXP=ROOT/'code/fine_residual_screen24_20261009_v2'
VAL=Path(r'C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val')
sys.path.insert(0,str(NATIVE));sys.path.insert(0,str(EXP))
from geometry import ResidualGeometryDEIM
FIELDS=['encoder_index','encoder_selected_logits','encoder_initial_boxes','pre_boxes','pre_logits','final_boxes','final_logits']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(n,j):
 with open(OUT/n,'x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False)
def state_sha(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):
  v=v.detach().cpu().contiguous();h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
 return h.hexdigest()
class Observer:
 def __init__(self,m):
  self.m=m;self.data={};self.handles=[];self.decoder=m.decoder;self.original_select=m.decoder._select_topk
  assert m.decoder.query_select_method=='default'
  def select(memory,logits,anchors,topk):
   out=self.original_select(memory,logits,anchors,topk)
   _,ids=torch.topk(logits.max(-1).values,topk,dim=-1)
   mem=memory.gather(1,ids.unsqueeze(-1).repeat(1,1,memory.shape[-1]))
   anc=anchors.gather(1,ids.unsqueeze(-1).repeat(1,1,4))
   assert torch.equal(mem,out[0]) and torch.equal(anc,out[2])
   self.data['encoder_index']=ids[0].detach().cpu().numpy().copy()
   self.data['encoder_selected_logits']=logits.gather(1,ids.unsqueeze(-1).repeat(1,1,10))[0].detach().cpu().numpy().copy()
   return out
  m.decoder._select_topk=select
  def initial(module,inputs):
   ref=inputs[1];assert ref.shape==(1,300,1,4)
   self.data['encoder_initial_boxes']=ref[0,:,0,:].detach().cpu().numpy().copy()
  def decoder(module,inputs,output):
   assert len(output)==6 and output[0].shape==(1,1,300,4) and output[1].shape==(1,1,300,10)
   self.data['pre_boxes']=output[4][0].detach().cpu().numpy().copy()
   self.data['pre_logits']=output[5][0].detach().cpu().numpy().copy()
   self.data['observed_final_boxes']=output[0][-1,0].detach().cpu().numpy().copy()
   self.data['observed_final_logits']=output[1][-1,0].detach().cpu().numpy().copy()
  self.handles.append(m.decoder.decoder.layers[0].register_forward_pre_hook(initial))
  self.handles.append(m.decoder.decoder.register_forward_hook(decoder))
 def reset(self):self.data={}
 def close(self):
  for h in self.handles:h.remove()
  self.m.decoder._select_topk=self.original_select
def main(phase):
 ph=sha(OUT/'protocol.json');assert ph==(OUT/'protocol.sha256').read_text('utf-8').strip()
 if phase=='preflight':
  write('start.json',{'unix':time.time(),'protocol_sha256':ph,'phase':'preflight','observations_originals_unchanged':True})
 start=read(OUT/'start.json')['unix'];deadline=start+900
 if phase=='complete':assert read(OUT/'preflight.json')['status']=='PASS'
 ann=read(VAL/'annotations/val.json');ims=sorted(ann['images'],key=lambda x:x['id']);assert len(ims)==548
 ah=sha(VAL/'annotations/val.json');assert ah=='0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'
 source={str(Path(__file__)):sha(__file__)}
 for module in list(sys.modules.values()):
  p=getattr(module,'__file__',None)
  if p and p.endswith('.py') and str(ROOT/'code') in p and Path(p).is_file():source[p]=sha(p)
 counts={'model_builds':0,'checkpoint_deserializations':0,'global_forwards':0,'skipped_existing':0,'image_decodes':0,'optimizer_steps':0}
 rows=[];branches=[];forward_io_seconds=0.;sources={};output_bytes=sum(p.stat().st_size for p in OUT.glob('seed*_*/*.npz'))
 prior=read(PAIR/'attempt02/results.json')
 try:
  for seed in (1,2):
   cp=BASE/f'seed{seed}/run/last.pth';ch=sha(cp);assert ch==prior['bindings'][str(seed)]['checkpoint_sha256'];sources[str(cp)]=ch
   cfgp=BASE/f'seed{seed}/bundle/resolved.yml';cfg=read(cfgp);sources[str(cfgp)]=sha(cfgp)
   ck=torch.load(cp,map_location='cpu',weights_only=True);counts['checkpoint_deserializations']+=1;assert ck['last_epoch']==23
   for branch,key in [('raw','model'),('EMA','ema')]:
    state=ck['model'] if branch=='raw' else ck['ema']['module']
    m=ResidualGeometryDEIM(cfg['ResidualGeometryDEIM']['recipe'],arm='baseline').cpu().eval();m.load_state_dict(state,strict=True)
    for p in m.parameters():p.requires_grad_(False)
    assert all(not p.requires_grad and p.device.type=='cpu' for p in m.parameters()) and all(not x.training for x in m.modules())
    counts['model_builds']+=1;before=state_sha(m.state_dict());assert before==state_sha(state)
    obs=Observer(m);folder=OUT/f'seed{seed}_{branch}';folder.mkdir(exist_ok=True)
    selected=[i for i in ims if i['id'] in (0,1,107,362)] if phase=='preflight' else ims
    raw_max_box=raw_max_logit=0.;new=0
    for pos,im in enumerate(selected):
     if time.time()>deadline:raise TimeoutError('Frozen900s total budget reached')
     iid=im['id'];dest=folder/f'image_{iid:06d}.npz';rawpath=S3/f'342_raw_val_seed{seed}/342_{iid:06d}.pt'
     old=torch.load(rawpath,map_location='cpu',weights_only=True)
     assert old['checkpoint_sha256']==ch and old['annotation_sha256']==ah and old['image_id']==iid
     if dest.exists():
      assert phase=='complete'
      with np.load(dest,allow_pickle=False) as z:
       assert str(z['checkpoint_sha256'])==ch and str(z['protocol_sha256'])==ph and int(z['image_id'])==iid and str(z['branch'])==branch
       if branch=='raw':assert np.array_equal(z['final_boxes'],old['final300_boxes'].numpy()) and np.array_equal(z['final_logits'],old['final300_logits'].numpy())
      counts['skipped_existing']+=1;continue
     t=time.perf_counter();src=VAL/'images'/im['file_name'];ih=sha(src);assert ih==old['source_image_sha256'];sources[str(src)]=ih
     with Image.open(src) as f:pil=f.convert('RGB');assert pil.size==(im['width'],im['height'])
     x=TF.to_tensor(TF.resize(pil,[640,640])).unsqueeze(0);counts['image_decodes']+=1;obs.reset()
     with torch.no_grad():out=m(x)
     counts['global_forwards']+=1;new+=1
     pb=out['pred_boxes'][0].detach().cpu().numpy().copy();lg=out['pred_logits'][0].detach().cpu().numpy().copy()
     assert pb.shape==(300,4) and lg.shape==(300,10) and np.isfinite(pb).all() and np.isfinite(lg).all()
     assert np.array_equal(pb,obs.data.pop('observed_final_boxes')) and np.array_equal(lg,obs.data.pop('observed_final_logits'))
     obs.data['final_boxes']=pb;obs.data['final_logits']=lg;assert set(obs.data)==set(FIELDS)
     assert all(np.isfinite(v).all() for v in obs.data.values())
     if branch=='raw':
      eb=float(np.max(np.abs(pb-old['final300_boxes'].numpy())));el=float(np.max(np.abs(lg-old['final300_logits'].numpy())))
      raw_max_box=max(raw_max_box,eb);raw_max_logit=max(raw_max_logit,el)
      if eb!=0 or el!=0:raise AssertionError(f'Raw numerical parity failed seed{seed} image{iid}: box{eb},logit{el}')
     stage_invalid={k:int((obs.data[k][:,2:]<=0).any(-1).sum()) for k in ('encoder_initial_boxes','pre_boxes','final_boxes')}
     sv,sf=torch.sigmoid(torch.from_numpy(lg)).flatten().topk(300);sq=(sf//10)[sv>.001].numpy()
     selected_invalid=int((pb[sq,2:]<=0).any(-1).sum())
     data={**obs.data,'image_id':np.array(iid),'seed':np.array(seed),'branch':np.array(branch),'original_wh':np.array([im['width'],im['height']]),
      'checkpoint_sha256':np.array(ch),'annotation_sha256':np.array(ah),'source_image_sha256':np.array(ih),'protocol_sha256':np.array(ph)}
     tmp=folder/f'image_{iid:06d}.incomplete'
     with open(tmp,'xb') as f:np.savez_compressed(f,**data)
     os.rename(tmp,dest);output_bytes+=dest.stat().st_size
     if output_bytes>200*1024**2:raise RuntimeError('Frozen200MiB observation output budget reached')
     elapsed=time.perf_counter()-t;forward_io_seconds+=elapsed
     rows.append({'seed':seed,'branch':branch,'image_id':iid,'path':str(dest),'sha256':sha(dest),'source_raw_cache_sha256':sha(rawpath),
      'raw_final_max_abs_boxes':raw_max_box if branch=='raw' else None,'raw_final_max_abs_logits':raw_max_logit if branch=='raw' else None,
      'nonpositive_boxes':stage_invalid,'nonpositive_selected_final_candidates':selected_invalid,'forward_io_seconds':elapsed})
     if phase=='preflight' or new%64==0:print(f'{phase}: seed{seed} {branch} {pos+1}/{len(selected)} new={new} elapsed={time.time()-start:.1f}s bytes={output_bytes}',flush=True)
    after=state_sha(m.state_dict());assert after==before and all(p.grad is None for p in m.parameters())
    obs.close();branches.append({'seed':seed,'branch':branch,'state_sha256_before':before,'state_sha256_after':after,
      'new_forwards':new,'raw_max_abs_box':raw_max_box if branch=='raw' else None,'raw_max_abs_logits':raw_max_logit if branch=='raw' else None,
      'all_grad_none':True,'all_eval_CPU':True});del m,obs
   del ck
  assert all(sha(p)==h for p,h in source.items()) and all(sha(p)==h for p,h in sources.items())
  elapsed=time.time()-start
  if phase=='preflight':
   estimate=forward_io_seconds/counts['global_forwards']*2192+30
   status='PASS' if estimate<900 else 'BUDGET_ESTIMATE_EXCEEDS900'
   write('preflight.json',{'status':status,'counts':counts,'branches':branches,'entries':rows,'estimated_total_forward_IO_seconds_plus30':estimate,
    'output_bytes':output_bytes,'elapsed_since_start':elapsed,'protocol_sha256':ph,'source_sha256':source,'input_sha256':sources})
   assert status=='PASS'
  else:
   assert counts['global_forwards']+counts['skipped_existing']==2192
   total_forwards=counts['global_forwards']+read(OUT/'preflight.json')['counts']['global_forwards'];assert total_forwards==2192
   write('complete.json',{'status':'COMPLETE_READONLY_CPU_REPLAY','counts_this_phase':counts,'total_global_forwards':total_forwards,'branches':branches,
    'entries':rows,'output_bytes':output_bytes,'elapsed_since_preflight_start':elapsed,'protocol_sha256':ph,'source_sha256':source,'input_sha256':sources,
    'model_parameter_updates':0,'optimizer_constructions':0,'training':False,'GPU':False,'remote':False,'test_reads':0})
  print(f'{phase} COMPLETE, elapsed={elapsed:.1f}s forwards={counts["global_forwards"]}',flush=True)
 except Exception as exc:
  import traceback
  write(phase+'_failure.json',{'status':'FAILED_PRESERVED','error':repr(exc),'traceback':traceback.format_exc(),'counts':counts,'entries':rows,
    'branches':branches,'elapsed_since_start':time.time()-start,'protocol_sha256':ph,'training_steps':0})
  raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['preflight','complete'],required=True);main(p.parse_args().phase)
