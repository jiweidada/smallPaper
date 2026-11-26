"""Frozen49 geometric support certificate. No detector, RGB reads, or RNG calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import sys,json,pathlib,hashlib,time,traceback,collections,platform
import numpy as np
import torch,torchvision
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F, InterpolationMode
from PIL import Image, __version__ as pil_version

BASE=pathlib.Path(__file__).resolve().parent
ROOT=BASE.parent
MAIN=ROOT/'持续寻找研究主线_20261020_v2'
DATA=MAIN/'49_augmentation_replay_audit_v4'
ANN=pathlib.Path('C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-train/annotations/train.json')
OUT=BASE/'label_pixel_support_attempt01'
PROTO=BASE/'03_训练视图标签与源像素_冻结协议.json'
CODE=ROOT/'code/dn_pre_frozen_diagnostic_20261020_v1/native/engine/data'
RES=getattr(Image,'Resampling',Image)

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def tensor_plain(t):return t.as_subclass(torch.Tensor) if isinstance(t,tv_tensors.TVTensor) else t
def resized(w,h):return (320,int(320*h/w)) if w<=h else (int(320*w/h),320)
def interval_support(lo,hi,n,m):
 a=np.zeros((1,n),dtype=np.float32);a[0,max(0,int(np.floor(lo))):min(n,int(np.ceil(hi)))]=1
 v=np.asarray(Image.fromarray(a,mode='F').resize((m,1),RES.BILINEAR)).reshape(-1)
 positive=np.flatnonzero(v>0)
 if len(positive)==0:return 0,0
 assert np.all(v[positive[0]:positive[-1]+1]>0)
 return int(positive[0]),int(positive[-1]+1)
def full_mask_check(meta,event,params,W,H,support_positive):
 w,h=meta['raw_size'];x1,y1,x2,y2=meta['raw_xyxy']
 a=np.zeros((h,w),dtype=np.float32)
 a[max(0,int(np.floor(y1))):min(h,int(np.ceil(y2))),max(0,int(np.floor(x1))):min(w,int(np.ceil(x2)))]=1
 image=Image.fromarray(a,mode='F').resize(tuple(meta['resized']),RES.BILINEAR)
 full=Image.new('F',(W,H),0);full.paste(image,tuple(meta['offset']))
 full=F.affine(full,**params,interpolation=InterpolationMode.NEAREST,fill=0)
 count=int(np.count_nonzero(np.asarray(full)>0))
 assert (count>0)==support_positive,('support_certificate_failed',event['seed'],event['position'],meta['source_slot'],meta['annotation_id'],count)
 return count

def main():
 started=time.monotonic();OUT.mkdir(exist_ok=False)
 torch.set_num_threads(2)
 assert torch.__version__=='2.5.1' and torchvision.__version__=='0.20.1'
 assert sha(ANN)=='32c1cae852e084430ea74503eb660f0707ec6b5e65d0998ce3ec8b448265c883'
 inputs=[PROTO,ANN,DATA/'01_transform_traces.json',DATA/'05_tagged_source_identity.json',MAIN/'51_mosaic_layout_geometry/01_rows.json',CODE/'transforms/mosaic.py',CODE/'transforms/_transforms.py',CODE/'_misc.py',pathlib.Path(__file__)]
 bindings={str(p):sha(p) for p in inputs}
 save(OUT/'preflight.json',{'inputs':bindings,'python':sys.executable,'torch':torch.__version__,'torchvision':torchvision.__version__,'pillow':pil_version,'platform':platform.platform(),'no_model':True,'no_raw_RGB':True})
 traces=json.loads((DATA/'01_transform_traces.json').read_text(encoding='utf8'))
 tags={(r['seed'],r['position']):r for r in json.loads((DATA/'05_tagged_source_identity.json').read_text(encoding='utf8'))}
 geom={(r['seed'],r['position']):r for r in json.loads((MAIN/'51_mosaic_layout_geometry/01_rows.json').read_text(encoding='utf8'))}
 data=json.loads(ANN.read_text(encoding='utf8'));anns=collections.defaultdict(list)
 for a in data['annotations']:anns[a['image_id']].append(a)
 rows=[];events=[];controls=[];unique={s:{'all':set(),'retained':set(),'dropped':set(),'images':set()} for s in [1,2]}
 totals={s:collections.Counter() for s in [1,2]}
 for event in traces:
  if len(event['source_ids'])!=4:continue
  seed,pos=event['seed'],event['position'];tag=tags[seed,pos];prior=geom[seed,pos]
  assert tag['boxes']==event['final_GT'] and tag['source_ids']==event['source_ids']
  affine=[p for p in event['params'] if p['class']=='RandomAffine'];assert len(affine)==1
  W,H=affine[0]['input_sizes'][0];params=affine[0]['params']
  sizes=[resized(*s['raw_size']) for s in event['sources']]
  assert W==2*max(s[0] for s in sizes) and H==2*max(s[1] for s in sizes)
  boxes=[];meta=[]
  for slot,source in enumerate(event['sources']):
   w,h=source['raw_size'];selected=[a for a in anns[source['image_id']] if not a.get('iscrowd',0)]
   b=torch.tensor([a['bbox'] for a in selected],dtype=torch.float32).reshape(-1,4)
   b[:,2:]+=b[:,:2];b[:,0::2].clamp_(0,w);b[:,1::2].clamp_(0,h)
   keep=(b[:,2:]>b[:,:2]).all(1);selected=[a for a,k in zip(selected,keep.tolist()) if k];b=b[keep]
   assert len(b)==source['raw_GT']
   raw=b.clone();b=F.resize(tv_tensors.BoundingBoxes(b,format='XYXY',canvas_size=(h,w)),size=[320]);assert b.canvas_size==sizes[slot][::-1]
   dx=(slot%2)*(W//2);dy=(slot//2)*(H//2)
   b=tensor_plain(b)+torch.tensor([dx,dy,dx,dy],dtype=torch.float32);boxes.append(b)
   for a,raw_box in zip(selected,raw.tolist()):
    meta.append({'seed':seed,'position':pos,'source_slot':slot,'source_image_id':source['image_id'],'annotation_id':a['id'],'class':a['category_id'],'raw_size':[w,h],'raw_xyxy':raw_box,'resized':list(sizes[slot]),'offset':[dx,dy]})
  b=tv_tensors.BoundingBoxes(torch.cat(boxes),format='XYXY',canvas_size=(H,W))
  b=F.affine(b,**params);b,keep1=F.sanitize_bounding_boxes(b,min_size=1,min_area=1)
  idx1=torch.arange(len(meta))[keep1]
  flips=sum(p['class']=='RandomHorizontalFlip' for p in event['params']);assert flips in [0,1]
  if flips:b=F.horizontal_flip(b)
  b=F.resize(b,size=[640,640]);b,keep2=F.sanitize_bounding_boxes(b,min_size=1,min_area=1)
  idx2=idx1[keep2]
  b=torchvision.ops.box_convert(tensor_plain(b),in_fmt='xyxy',out_fmt='cxcywh')/torch.tensor([640,640,640,640])
  target=torch.tensor(event['final_GT'],dtype=torch.float32).reshape(-1,4)
  assert torch.equal(b,target),('box_parity',seed,pos,float((b-target).abs().max()) if b.shape==target.shape and b.numel() else str((b.shape,target.shape)))
  retained=set(idx2.tolist());stage1=set(idx1.tolist())
  decoded=[{'annotation_id':meta[i]['annotation_id'],'source_image_id':meta[i]['source_image_id'],'source_slot':meta[i]['source_slot']} for i in idx2.tolist()]
  assert decoded==tag['instances']
  assert [meta[i]['class'] for i in idx2.tolist()]==event['final_classes']
  # Exact nearest-neighbour input coordinate lineage through the recorded affine.
  coord=np.arange(1,W*H+1,dtype=np.int32).reshape(H,W)
  warped=np.asarray(F.affine(Image.fromarray(coord,mode='I'),**params,interpolation=InterpolationMode.NEAREST,fill=0))
  src=warped.astype(np.int64)-1;valid=src>=0
  sx=np.where(valid,src%W,-1);sy=np.where(valid,src//W,-1)
  observed=[]
  for slot,(rw,rh) in enumerate(sizes):
   dx=(slot%2)*(W//2);dy=(slot//2)*(H//2)
   count=int(np.count_nonzero(valid&(sx>=dx)&(sx<dx+rw)&(sy>=dy)&(sy<dy+rh)))
   assert count==prior['pieces'][slot]['pixels'],('source_pixel_gate',seed,pos,slot,count,prior['pieces'][slot]['pixels'])
   observed.append(count)
  visible=np.zeros(W*H,dtype=np.int32);visible[src[valid]]=1
  integral=np.pad(visible.reshape(H,W),((1,0),(1,0))).cumsum(0,dtype=np.int64).cumsum(1,dtype=np.int64)
  zero_retained=[];control_indices=[]
  if retained:control_indices.append(min(retained))
  dropped=set(range(len(meta)))-retained
  if dropped:control_indices.append(min(dropped))
  for i,m in enumerate(meta):
   w,h=m['raw_size'];rw,rh=m['resized'];dx,dy=m['offset'];x1,y1,x2,y2=m['raw_xyxy']
   a,c=interval_support(x1,x2,w,rw);d,f=interval_support(y1,y2,h,rh)
   a+=dx;c+=dx;d+=dy;f+=dy
   count=int(integral[f,c]-integral[d,c]-integral[f,a]+integral[d,a])
   status='retained' if i in retained else ('drop_sanitize2' if i in stage1 else 'drop_sanitize1')
   row={**m,'status':status,'support_rect_canvas':[a,d,c,f],'distinct_selected_support_cells':count,'zero_support':count==0}
   rows.append(row)
   totals[seed]['initial_valid']+=1;totals[seed][status]+=1
   totals[seed][status+('_zero' if count==0 else '_nonzero')]+=1
   unique[seed]['all'].add(m['annotation_id']);unique[seed]['images'].add(m['source_image_id']);unique[seed]['retained' if i in retained else 'dropped'].add(m['annotation_id'])
   if i in retained and count==0:zero_retained.append(i)
  eventrows=rows[-len(meta):] if meta else []
  for i in sorted(set(control_indices+zero_retained)):
   direct=full_mask_check(meta[i],event,params,W,H,eventrows[i]['distinct_selected_support_cells']>0)
   controls.append({'seed':seed,'position':pos,'source_slot':meta[i]['source_slot'],'annotation_id':meta[i]['annotation_id'],'status':eventrows[i]['status'],'zero_support':eventrows[i]['zero_support'],'direct_output_positive_cells':direct})
  totals[seed]['mosaic_events']+=1
  events.append({'seed':seed,'position':pos,'initial_valid':len(meta),'final_retained':len(retained),'sanitize1_removed':len(meta)-len(stage1),'sanitize2_removed':len(stage1)-len(retained),'retained_zero_support':len(zero_retained),'source_pixel_counts':observed,'all_gates_passed':True})
  if len(events)%20==0:print(json.dumps({'events_completed':len(events),'seconds':time.monotonic()-started}),flush=True)
 for p,h in bindings.items():assert sha(p)==h,('input_changed',p)
 save(OUT/'rows.json',rows);save(OUT/'events.json',events);save(OUT/'certificate_controls.json',controls)
 summary={'status':'complete','scope':json.loads(PROTO.read_text(encoding='utf8'))['scope'],'totals':{str(s):dict(totals[s]) for s in [1,2]},'unique':{str(s):{k:len(v) for k,v in unique[s].items()} for s in [1,2]},'gates':{'geometry_exact_all_events':True,'source_pixels_exact_all_events':True,'2D_certificate_controls':len(controls),'all_controls_pass':True},'positive_event_counts':{str(s):sum(e['retained_zero_support']>0 and e['seed']==s for e in events) for s in [1,2]},'new_model_constructions':0,'model_forwards':0,'optimizer_steps':0,'raw_RGB_reads':0,'new_random_augmentation_draws':0,'seconds':time.monotonic()-started,'inputs':bindings}
 save(OUT/'results.json',summary);save(OUT/'sha256.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
 print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as e:
  if OUT.exists():save(OUT/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'scientific_result':'NA: fidelity/computation did not complete'})
  raise
