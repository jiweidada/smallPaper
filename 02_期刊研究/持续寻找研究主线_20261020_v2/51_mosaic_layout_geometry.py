"""The registered fixed-source, fixed-exposure 27-layout feasibility check."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import pathlib,sys,json,itertools,hashlib,time
ROOT=pathlib.Path(__file__).resolve().parent;WORK=ROOT.parents[1]
sys.path.insert(0,str(WORK/'英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1'))
import common as n
from torchvision.transforms.v2 import functional as F, InterpolationMode
from PIL import Image
np=n.np
DATA=ROOT/'49_augmentation_replay_audit_v4';OUT=ROOT/'51_mosaic_layout_geometry'
PROTO=ROOT/'50_Mosaic固定曝光布局_几何首检协议.txt'
ARMS=['original','instance_objective','simple_risk','fixed_random']
EPS=1e-9

def gate(cross,total):return None if total==0 else 10*cross>=total

def resized(size):
 w,h=size
 return (320,int(320*h/w)) if w<=h else (int(320*w/h),320)

def cells(p):
 x,y,z=p
 return np.array([[640*x,640*y],[640*(1-x),640*z],[640*x,640*(1-y)],[640*(1-x),640*(1-z)]])

def choose(candidates,objective):
 # candidates original first, then lexicographic layout order: preserve exact ties.
 values=[objective(c) for c in candidates]
 return candidates[max(range(len(values)),key=lambda i:(values[i],-i))]

def support_containers(event,boxes,slots):
 affine=[p for p in event['params'] if p['class']=='RandomAffine']
 assert len(affine)==1
 W,H=affine[0]['input_sizes'][0]
 sizes=[resized(s['raw_size']) for s in event['sources']]
 assert len(sizes)==4 and 2*max(s[0] for s in sizes)==W and 2*max(s[1] for s in sizes)==H
 mask=Image.new('L',(W,H),0)
 for i,(w,h) in enumerate(sizes):
  x=(i%2)*(W//2);y=(i//2)*(H//2)
  mask.paste(i+1,(x,y,x+w,y+h))
 mask=F.affine(mask,**affine[0]['params'],interpolation=InterpolationMode.NEAREST,fill=0)
 flips=sum(p['class']=='RandomHorizontalFlip' for p in event['params'])
 assert flips in [0,1]
 if flips:mask=F.horizontal_flip(mask)
 a=np.asarray(mask); assert np.isin(a,[0,1,2,3,4]).all()
 cx,cy,bw,bh=boxes.T
 xy=np.stack([cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2],axis=1)*640
 pieces=[]
 for slot in range(4):
  yy,xx=np.where(a==slot+1)
  has_pixels=len(xx)>0;local=xy[slots==slot]
  bounds=[]
  if has_pixels:bounds.append(np.array([xx.min()*640/W,yy.min()*640/H,(xx.max()+1)*640/W,(yy.max()+1)*640/H]))
  if len(local):bounds.append(np.r_[local[:,:2].min(0),local[:,2:].max(0)])
  if not bounds:pieces.append({'size':[0.,0.],'pixels':0,'GT':0,'bounds':None});continue
  b=np.stack(bounds);lo=b[:,:2].min(0);hi=b[:,2:].max(0)
  # The exact same source support plus annotation extent; no other-source pixels.
  assert (hi>lo).all()
  pieces.append({'size':(hi-lo).tolist(),'pixels':int(len(xx)),'GT':int(len(local)),'bounds':np.r_[lo,hi].tolist()})
 assert sum(v['GT'] for v in pieces)==len(boxes)
 return pieces

def main():
 start=time.monotonic();OUT.mkdir(exist_ok=False);resources=n.resource_limits()
 assert gate(1,10) is True and gate(0,10) is False and gate(0,0) is None
 assert np.allclose(cells((.5,.5,.5)),[[320,320]]*4)
 traces=json.loads((DATA/'01_transform_traces.json').read_text(encoding='utf8'))
 tags=json.loads((DATA/'05_tagged_source_identity.json').read_text(encoding='utf8'))
 identities={(t['seed'],t['position']):t for t in tags}
 totals={str(s):{a:{'original_sub4':0,'crossed_to_ge4':0,'original_ge4_to_sub4':0,'GT':0} for a in ARMS} for s in [1,2]}
 rows=[];layouts=list(itertools.product([.4,.5,.6],repeat=3))
 for event in traces:
  if len(event['source_ids'])!=4:continue
  ident=identities[event['seed'],event['position']]
  assert ident['source_ids']==event['source_ids'] and ident['boxes']==event['final_GT']
  boxes=np.array(ident['boxes'],dtype=float).reshape(-1,4)
  slots=np.array([g['source_slot'] for g in ident['instances']],dtype=int)
  assert ((slots>=0)&(slots<4)).all()
  pieces=support_containers(event,boxes,slots)
  sizes=np.array([p['size'] for p in pieces]);active=(sizes>0).all(1)
  m=boxes[:,2:].min(1)*640;small=m<4
  counts=np.bincount(slots[small],minlength=4)
  candidates=[{'layout':'original','scales':np.ones(4)}]
  for layout in layouts:
   scale=np.ones(4);scale[active]=np.min(cells(layout)[active]/sizes[active],axis=1)
   if (scale[active]<.8-EPS).any():continue
   changed=m*scale[slots]
   if ((m>=4)&(changed<4-EPS)).any():continue
   candidates.append({'layout':list(layout),'scales':scale})
  selected={'original':candidates[0],
   'instance_objective':choose(candidates,lambda c:np.minimum(m[small]*c['scales'][slots[small]]/4,1).sum()),
   'simple_risk':choose(candidates,lambda c:(counts*c['scales']).sum())}
  prefix=f"{event['seed']}:{event['position']}:"
  selected['fixed_random']=min(candidates,key=lambda c:hashlib.sha256((prefix+str(c['layout'])).encode()).hexdigest())
  metrics={}
  for arm,c in selected.items():
   new=m*c['scales'][slots]
   stat={'original_sub4':int(small.sum()),'crossed_to_ge4':int((small&(new>=4)).sum()),
    'original_ge4_to_sub4':int(((m>=4)&(new<4)).sum()),'GT':len(m)}
   if arm!='original':assert stat['original_ge4_to_sub4']==0
   for k,v in stat.items():totals[str(event['seed'])][arm][k]+=v
   metrics[arm]={'layout':c['layout'],'scales':c['scales'].tolist(),**stat}
  rows.append({'seed':event['seed'],'position':event['position'],'batch':event['batch'],
   'source_ids':event['source_ids'],'pieces':pieces,'non_original_feasible':len(candidates)-1,
   'instance_equals_risk':selected['instance_objective']['layout']==selected['simple_risk']['layout'],'arms':metrics})
 decisions={s:gate(r['instance_objective']['crossed_to_ge4'],r['instance_objective']['original_sub4']) for s,r in totals.items()}
 n.save_json(OUT/'01_rows.json',rows)
 n.save_json(OUT/'02_results.json',{'state':'complete','resources':resources,'totals':totals,'per_seed_physical_gate':decisions,
  'both_passed':all(v is True for v in decisions.values()),
  'events':{str(s):sum(r['seed']==s for r in rows) for s in [1,2]},
  'events_any_alternative':{str(s):sum(r['seed']==s and r['non_original_feasible']>0 for r in rows) for s in [1,2]},
  'instance_risk_same_every_event':all(r['instance_equals_risk'] for r in rows),
  'GT_identity_preserved':True,'source_identity_preserved':True,'protocol_sha256':n.sha(PROTO),
  'data_files_sha256':{f:n.sha(DATA/f) for f in ['01_transform_traces.json','05_tagged_source_identity.json']},
  'model_forwards':0,'model_backwards':0,'optimizer_steps':0,'new_training':False,'heldout_reads':False,
  'seconds':time.monotonic()-start,'scope':'Physical feasibility of the registered finite packing family; not detector benefit or all augmentation variants.'})
 n.save_json(OUT/'03_SHA256.json',{p.name:n.sha(p) for p in OUT.iterdir() if p.is_file()})
 print(json.dumps({'totals':totals,'decisions':decisions,'rows':len(rows),'seconds':time.monotonic()-start}))

if __name__=='__main__':main()
