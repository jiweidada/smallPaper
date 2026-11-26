"""One-shot fixed-pixel phase diagnosis; frozen FP32 YOLO26, no training path."""
from pathlib import Path
import ast,copy,hashlib,importlib.metadata,json,os,signal,subprocess,sys,time,traceback,zipfile
import cv2,numpy as np,torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils import ops,nms,events
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.torch_utils import unwrap_model
events.enabled=False
R=Path(__file__).resolve().parent
CFG=json.loads((R/'phase_protocol_FROZEN.json').read_text())
E=R/'evidence';START=time.monotonic();STAGE='preflight';DONE=0
COUNT=dict(whole_detector_forwards=0,extra_native_O_decodes=0,all_native_head_decodes=0,optimizer_updates=0,standalone_TAL_calls=0)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 return h.hexdigest()
def write(p,o):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def guard():assert time.monotonic()-START<=CFG['caps']['wall_seconds'],'wall cap'
def bump(name):
 guard();assert COUNT[name]<CFG['caps'][name],name+' cap'
 COUNT[name]+=1
 if name=='whole_detector_forwards':status()
def status(extra=None):
 o=dict(status='RUNNING',stage=STAGE,pid=os.getpid(),completed_case_shifts=DONE,counts=dict(COUNT),elapsed_seconds=time.monotonic()-START)
 if extra:o.update(extra)
 write(R/'status.json',o)
def digest(model):
 h=hashlib.sha256()
 for name,x in sorted(model.state_dict().items()):
  h.update(name.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def timeout_handler(signum,frame):raise TimeoutError('Frozen wall-time cap')
def main():
 global STAGE,DONE
 with (R/'worker_once.lock').open('x') as f:f.write(str(os.getpid()))
 signal.signal(signal.SIGALRM,timeout_handler);signal.alarm(CFG['caps']['wall_seconds'])
 manifest=json.loads((R/'execution_inputs_manifest.json').read_text())
 assert sha(__file__)==manifest['worker_sha256'] and sha(R/'phase_protocol_FROZEN.json')==manifest['protocol_sha256']
 for package,expected in CFG['versions'].items():assert importlib.metadata.version(package)==expected,(package,importlib.metadata.version(package),expected)
 for row in CFG['source_files']:
  target=Path(CFG['host_package']).parent/row['path'];assert sha(target)==row['sha256'],str(target)
 for path,h in CFG['remote_assets'].items():assert sha(path)==h,path
 for c in CFG['cases']:
  assert sha(c['path'])==c['sha256'];assert sha(c['reference_capture'])==c['reference_capture_sha256']
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader,nounits'],text=True).strip()
 procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).strip()
 assert len(gpu.splitlines())==1 and int(gpu.split(',')[-1])<=1024 and not procs,(gpu,procs)
 E.mkdir(exist_ok=False);(E/'captures').mkdir()
 write(E/'start.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],GPU_before=gpu,protocol_sha256=sha(R/'phase_protocol_FROZEN.json'),worker_sha256=sha(__file__),torchvision_imported_before_model='torchvision' in sys.modules))
 helper_path=Path(CFG['evaluator']);source=helper_path.read_text();tree=ast.parse(source)
 nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ['prepare_pixels','detection_records']]
 assert len(nodes)==2;env=globals().copy();exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned-evaluator-helpers','exec'),env)
 torch.set_num_threads(4)
 assert torch.backends.cuda.matmul.allow_tf32==False and torch.backends.cudnn.allow_tf32==True and torch.backends.cudnn.benchmark==False
 model=YOLO(CFG['checkpoint']['path']).model.cuda().float().eval().requires_grad_(False);head=model.model[-1]
 assert head.nc==10 and head.reg_max==1 and head.stride.tolist()==[8.,16.,32.]
 head.end2end=False;head.export=False;head.max_det=500;head.agnostic_nms=False
 before=digest(model);assert before==CFG['reference_state_digest']
 assert ('torchvision' in sys.modules)==CFG['torchvision_imported_before_inference']
 model.register_forward_pre_hook(lambda m,args:bump('whole_detector_forwards'))
 original_decode=head._inference
 def counted_decode(raw):bump('all_native_head_decodes');return original_decode(raw)
 head._inference=counted_decode
 write(E/'runtime_model_info.json',dict(model_class=model.__class__.__name__,head_class=head.__class__.__name__,nc=head.nc,reg_max=head.reg_max,strides=head.stride.tolist(),state_digest_before=before,parameters=sum(x.numel() for x in model.parameters()),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark))
 outputs={a:[] for a in CFG['output_groups']};proof=[];first=None;lineage=[];old_predictions={}
 for k,path in CFG['reference_predictions'].items():
  by_image={}
  for row in json.loads(Path(path).read_text()):by_image.setdefault(row['image_id'],[]).append(row)
  old_predictions[k]=by_image
 STAGE='collect';torch.cuda.reset_peak_memory_stats()
 for ci,c in enumerate(CFG['cases']):
  with Image.open(c['path']) as im:pixels=np.asarray(im.convert('RGB'))
  tensor=env['prepare_pixels'](pixels,c['geometry']);l,t,r,b=c['geometry']['pad_ltrb'];rw,rh=c['geometry']['resized_wh']
  assert min(t,b)>=32 and tensor.shape==(1,3,640,640)
  fill=tensor[0,:,0,0];assert torch.equal(fill,fill[0].expand_as(fill))
  assert torch.all(tensor[:,:,:t,:]==fill.view(1,3,1,1)) and torch.all(tensor[:,:,t+rh:,:]==fill.view(1,3,1,1))
  reference=np.load(c['reference_capture'],allow_pickle=False)
  for dy in CFG['shifts_y']:
   guard();head.end2end=False;shifted=torch.roll(tensor,shifts=dy,dims=2)
   assert torch.equal(torch.roll(shifted,shifts=-dy,dims=2),tensor)
   assert torch.equal(shifted[:,:,t+dy:t+dy+rh,l:l+rw],tensor[:,:,t:t+rh,l:l+rw])
   proof.append(dict(case_index=ci,image_id=c['image_id'],shift_y=dy,real_pixels_exact=True,full_inverse_exact=True,original_pad=c['geometry']['pad_ltrb'],shifted_pad=[l,t+dy,r,b-dy]))
   with torch.inference_mode():
    decoded,raw=model(shifted);assert set(raw)=={'one2many','one2one'}
    parts=dict(M=raw['one2many'],O=raw['one2one'])
    assert all(torch.equal(a,b) for a,b in zip(parts['M']['feats'],parts['O']['feats']))
    box=dict(M=ops.xywh2xyxy(decoded[:,:4,:].transpose(1,2)).contiguous());prob=dict(M=decoded[:,4:,:].transpose(1,2).contiguous())
    head.end2end=True;bump('extra_native_O_decodes');other=head._inference(parts['O']);head.end2end=False
    box['O']=other[:,:4,:].transpose(1,2).contiguous();prob['O']=other[:,4:,:].transpose(1,2).contiguous()
    for k in ['M','O']:
     assert torch.isfinite(parts[k]['boxes']).all() and torch.isfinite(parts[k]['scores']).all()
     if dy==0:
      for n in ['boxes','scores']:assert np.array_equal(parts[k][n].squeeze(0).cpu().numpy(),reference[k+'_'+n]),('zero-raw-fidelity',ci,k,n)
      assert np.array_equal(box[k].squeeze(0).cpu().numpy(),reference[k+'_decoded_xyxy']),('zero-decode-fidelity',ci,k)
    capture={k+'_'+n:parts[k][n].squeeze(0).cpu().numpy() for k in ['M','O'] for n in ['boxes','scores']}
    capture.update({k+'_decoded_xyxy':box[k].squeeze(0).cpu().numpy() for k in ['M','O']})
    np.savez_compressed(E/'captures'/('case_%03d_shift_%+03d.npz'%(ci,dy)),**capture)
    z=torch.cat([ops.xyxy2xywh(box['M']).transpose(1,2),prob['M'].transpose(1,2)],dim=1)
    ds,idx=nms.non_max_suppression(z,conf_thres=.001,iou_thres=.7,multi_label=True,agnostic=False,max_det=500,nc=10,end2end=False,return_idxs=True)
    mdet=ds[0];mid=idx[0]
    scores,labels,ids=head.get_topk_index(prob['O'],500);odet=torch.cat([head._gather(box['O'],ids),scores,labels],dim=-1)[0]
    if ci==0 and dy==0:first=dict(tensor=shifted.clone(),raw={k:{n:v[n].clone() for n in ['boxes','scores']} for k,v in parts.items()},native_O=odet.unsqueeze(0).clone())
    oid=ids[0];keep=odet[:,4]>=.001;odet=odet[keep];oid=oid[keep]
    meta=copy.deepcopy(c);meta['geometry']['pad_ltrb']=[l,t+dy,r,b-dy]
    for k,det,ids1 in [('M',mdet,mid),('O',odet,oid)]:
     _,original=env['detection_records'](det,meta)
     if dy==0:assert original==old_predictions[k][c['image_id']],('zero-native-prediction-fidelity',ci,k)
     outputs[k+'_'+str(dy)].extend(original)
     lineage.append(dict(case_index=ci,image_id=c['image_id'],shift_y=dy,branch=k,anchors=ids1.cpu().tolist()))
   DONE+=1;status()
  reference.close()
  if (ci+1)%4==0:print(json.dumps(dict(completed_images=ci+1,counts=COUNT,elapsed_seconds=time.monotonic()-START)),flush=True)
 STAGE='repeat';head.end2end=True
 with torch.inference_mode():
  repeat,raw=model(first['tensor']);assert torch.equal(repeat,first['native_O'])
  for k,name in [('M','one2many'),('O','one2one')]:
   for n in ['boxes','scores']:assert torch.equal(raw[name][n],first['raw'][k][n]),('repeat',k,n)
 after=digest(model);assert after==before
 for k,rows in outputs.items():write(E/(k+'_predictions_original.json'),rows)
 write(E/'pixel_and_zero_fidelity.json',dict(status='PASS_ALL_REAL_PIXELS_AND_ZERO_RAW_AND_NATIVE_REPEAT',proof=proof,zero_fidelity_cases=len(CFG['cases']),native_repeat_bitwise=True,state_digest_after=after,optimizer_updates=0))
 write(E/'lineage.json',lineage)
 for row in CFG['source_files']:assert sha(Path(CFG['host_package']).parent/row['path'])==row['sha256']
 for path,h in CFG['remote_assets'].items():assert sha(path)==h
 for c in CFG['cases']:assert sha(c['path'])==c['sha256']
 assert COUNT['whole_detector_forwards']==169 and COUNT['extra_native_O_decodes']==168 and COUNT['all_native_head_decodes']==337
 result=dict(status='NATIVE_COLLECTION_COMPLETE_NO_ALGORITHM_CLAIM',counts=COUNT,images=len(CFG['cases']),shifts=CFG['shifts_y'],zero_fidelity=True,repeat_fidelity=True,weights_unchanged=True,peak_GPU_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,peak_GPU_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3,elapsed_seconds=time.monotonic()-START)
 write(E/'result.json',result)
 for name in ['phase_protocol_FROZEN.json','execution_inputs_manifest.json','native_phase_worker.py']:(E/name).write_bytes((R/name).read_bytes())
 inventory={str(p.relative_to(E)):sha(p) for p in sorted(E.rglob('*')) if p.is_file()};write(E/'file_manifest.json',dict(files_sha256=inventory))
 STAGE='archive';status();archive=R/'native_evidence.zip'
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for p in sorted(E.rglob('*')):
   if p.is_file():z.write(p,str(p.relative_to(E)))
 complete=dict(result,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,archived_files=len(inventory)+1,wall_seconds=time.monotonic()-START)
 write(R/'complete.json',complete);status(dict(status='COMPLETED',stage='complete'));signal.alarm(0)
 print(json.dumps(complete),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(R/'exception.json',dict(status='FAILED_NO_RETRY',stage=STAGE,counts=COUNT,error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-START))
  status(dict(status='FAILED_NO_RETRY'));raise
