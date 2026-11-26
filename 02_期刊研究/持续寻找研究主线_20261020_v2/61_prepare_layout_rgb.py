"""Locked RGB reconstruction and layout inputs, without constructing a detector."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import pathlib,sys,json,hashlib,time
ROOT=pathlib.Path(__file__).resolve().parent;WORK=ROOT.parents[1]
sys.path.insert(0,str(WORK/'英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1'))
import common as n
from torchvision.transforms.v2 import functional as F, InterpolationMode, RandomPhotometricDistort
from PIL import Image
torch,np=n.torch,n.np
DATA=ROOT/'49_augmentation_replay_audit_v4';OUT=ROOT/'62_layout_rgb'
IMAGES=pathlib.Path('D:/数据集/visdrone2019/VisDrone2019-DET-train/images')
ARMS=['original','instance_objective','simple_risk','fixed_random','global_tight','neutral']

def tensor_hash(v):return hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()

def to_input(im):
 return F.pil_to_tensor(F.resize(im,[640,640],interpolation=InterpolationMode.BILINEAR,antialias=True)).float()/255.

def restore(event):
 affine=next(p for p in event['params'] if p['class']=='RandomAffine')
 W,H=affine['input_sizes'][0];rgb=Image.new('RGB',(W,H),0);mask=Image.new('L',(W,H),0)
 for i,source in enumerate(event['sources']):
  p=IMAGES/source['file_name'];assert p.resolve().parent==IMAGES.resolve()
  raw=Image.open(p).convert('RGB')
  assert hashlib.sha256(raw.tobytes()).hexdigest()==source['raw_pixel_sha256']
  im=F.resize(raw,[320],interpolation=InterpolationMode.BILINEAR,antialias=True)
  x=(i%2)*(W//2);y=(i//2)*(H//2)
  rgb.paste(im,(x,y));mask.paste(i+1,(x,y,x+im.width,y+im.height))
 rgb=F.affine(rgb,**affine['params'],interpolation=InterpolationMode.NEAREST,fill=0)
 mask=F.affine(mask,**affine['params'],interpolation=InterpolationMode.NEAREST,fill=0)
 photo=next(p for p in event['params'] if p['class']=='RandomPhotometricDistort')
 pp=dict(photo['params'])
 if pp['channel_permutation'] is not None:pp['channel_permutation']=torch.tensor(pp['channel_permutation'],dtype=torch.long)
 rgb=RandomPhotometricDistort()._transform(rgb,pp)
 if any(p['class']=='RandomHorizontalFlip' for p in event['params']):rgb=F.horizontal_flip(rgb);mask=F.horizontal_flip(mask)
 expected=event['final640_sha256'];actual=tensor_hash(to_input(rgb));assert actual==expected,(event['seed'],event['position'],'original RGB SHA',actual,expected)
 return rgb,mask

def cells(layout):
 x,y,z=np.array(layout)*640
 origins=np.array([[0,0],[x,0],[0,y],[x,z]])
 sizes=np.array([[x,y],[640-x,z],[x,640-y],[640-x,640-z]])
 return origins,sizes

def mapping(arm,row,global_row):
 if arm=='original':return np.ones(4),np.zeros((4,2)),None
 if arm=='global_tight':return np.full(4,global_row['scale']),np.tile(global_row['translation'],(4,1)),None
 if arm=='neutral':
  layout=[.5,.5,.5];origin,cell=cells(layout)
  sizes=np.array([p['size'] for p in row['pieces']]);active=(sizes>0).all(1)
  scale=np.ones(4);scale[active]=np.min(cell[active]/sizes[active],axis=1)
 else:
  chosen=row['arms'][arm];layout=chosen['layout'];scale=np.array(chosen['scales'])
  if layout=='original':return np.ones(4),np.zeros((4,2)),None
  origin,cell=cells(layout)
 offsets=[]
 for i,piece in enumerate(row['pieces']):
  if piece['bounds'] is None:offsets.append([0.,0.]);continue
  offsets.append(origin[i]+(cell[i]-scale[i]*np.array(piece['size']))/2-scale[i]*np.array(piece['bounds'][:2]))
 return scale,np.array(offsets),(origin,cell)

def transformed(boxes,slots,scale,shift):
 result=boxes.copy();result[:,:2]=boxes[:,:2]*scale[slots,None]+shift[slots]/640;result[:,2:]*=scale[slots,None]
 return result

def render(rgb,mask,scale,shift,cell):
 W,H=rgb.size;image=np.array(rgb);ids=np.array(mask);bg=image[ids==0]
 fill=tuple(int(v) for v in bg[0]) if len(bg) else (0,0,0)
 if len(bg):assert np.all(bg==fill),'native source0 not constant'
 out=Image.new('RGBA',(W,H),fill+(255,))
 x=(np.arange(W)+.5)*640/W;y=(np.arange(H)+.5)*640/H
 for i in range(4):
  rgba=np.zeros((H,W,4),dtype=np.uint8);sel=ids==i+1
  rgba[:,:,:3][sel]=image[sel];rgba[:,:,3][sel]=255
  layer=Image.fromarray(rgba,'RGBA')
  tx,ty=shift[i]*np.array([W/640,H/640]);s=float(scale[i])
  layer=layer.transform((W,H),Image.Transform.AFFINE,(1/s,0,-tx/s,0,1/s,-ty/s),resample=Image.Resampling.BILINEAR,fillcolor=(0,0,0,0))
  if cell is not None:
   origin,size=cell;left,top=origin[i];right,bottom=origin[i]+size[i]
   a=np.array(layer);inside=(x[None,:]>=left)&(x[None,:]<right)&(y[:,None]>=top)&(y[:,None]<bottom)
   a[~inside]=0;layer=Image.fromarray(a,'RGBA')
  out=Image.alpha_composite(out,layer)
 return out.convert('RGB'),fill

def main():
 start=time.monotonic();OUT.mkdir(exist_ok=False);resources=n.resource_limits()
 events=json.loads((DATA/'01_transform_traces.json').read_text(encoding='utf8'))
 ids=json.loads((DATA/'05_tagged_source_identity.json').read_text(encoding='utf8'))
 tagged={(r['seed'],r['position']):r for r in ids}
 batch=json.loads((DATA/'02_plain_batch_identity.json').read_text(encoding='utf8'))
 geometry=json.loads((ROOT/'51_mosaic_layout_geometry/01_rows.json').read_text(encoding='utf8'))
 geo={(r['seed'],r['position']):r for r in geometry}
 globals=json.loads((ROOT/'55_global_tight_geometry.json').read_text(encoding='utf8'))
 glob={(r['seed'],r['position']):r for r in globals['rows']}
 selected=[]
 for seed in [1,2]:
  candidates=[e for e in events if e['seed']==seed and len(e['source_ids'])==4
      and all(p['pixels']>0 for p in geo[e['seed'],e['position']]['pieces'])
      and all(v is None for v in batch[str(seed)][e['batch']]['mixup'])]
  selected+=candidates[:8]
 n.save_json(OUT/'00_selection.json',[{'seed':e['seed'],'position':e['position'],'batch':e['batch']} for e in selected])
 results=[]
 for k,event in enumerate(selected):
  assert time.monotonic()-start<900 and n.psutil.virtual_memory().available/2**30>8
  key=event['seed'],event['position'];ident=tagged[key];row=geo[key]
  before=torch.get_rng_state().clone();rgb,mask=restore(event)
  assert torch.equal(before,torch.get_rng_state()),'deterministic reconstruction changed RNG'
  original=to_input(rgb)
  identity,fill=render(rgb,mask,np.ones(4),np.zeros((4,2)),None)
  assert np.array_equal(np.array(identity),np.array(rgb)),'source-mask identity RGB mismatch'
  assert torch.equal(to_input(identity),original),'identity resized input mismatch'
  boxes=np.array(ident['boxes'],dtype=np.float64).reshape(-1,4)
  slots=np.array([x['source_slot'] for x in ident['instances']],dtype=int)
  images={};targets={};metadata={}
  for arm in ARMS:
   scale,shift,cell=mapping(arm,row,glob[key])
   changed=transformed(boxes,slots,scale,shift);m=boxes[:,2:].min(1)*640;new=changed[:,2:].min(1)*640
   if arm=='neutral' and ((scale<.8-1e-9).any() or ((m>=4)&(new<4-1e-9)).any()):
    metadata[arm]={'state':'NA_infeasible'};continue
   if arm=='original':image=original;assert np.array_equal(changed,boxes)
   elif arm=='global_tight':
    # A genuinely simple whole-image affine control: no per-source alpha seams.
    s=scale[0];tx,ty=shift[0]*np.array([rgb.width/640,rgb.height/640])
    image=to_input(rgb.transform(rgb.size,Image.Transform.AFFINE,(1/s,0,-tx/s,0,1/s,-ty/s),resample=Image.Resampling.BILINEAR,fillcolor=fill))
   else:
    made,_=render(rgb,mask,scale,shift,cell);image=to_input(made)
   assert np.isfinite(changed).all() and (changed[:,2:]>0).all()
   lo=changed[:,:2]-changed[:,2:]/2;hi=changed[:,:2]+changed[:,2:]/2
   assert lo.min()>-1e-5 and hi.max()<1+1e-5,(key,arm,'box outside frame')
   if arm in row['arms']:
    assert int(((m<4)&(new>=4)).sum())==row['arms'][arm]['crossed_to_ge4']
   images[arm]=image
   targets[arm]={'boxes':torch.tensor(changed,dtype=torch.float32),'labels':torch.tensor(event['final_classes'],dtype=torch.int64)}
   metadata[arm]={'state':'ready','scale':scale.tolist(),'shift':shift.tolist(),'input_sha256':tensor_hash(image),
      'GT':len(boxes),'sub4_original':int((m<4).sum()),'crossed':int(((m<4)&(new>=4)).sum()),
      'max_box_float32_error_px':float(np.max(np.abs(targets[arm]['boxes'].numpy()-changed))*640)}
  torch.save({'images':images,'targets':targets,'original_sub4':torch.tensor(boxes[:,2:].min(1)*640<4),
    'source_instances':ident['instances'],'metadata':metadata,'event':{'seed':key[0],'position':key[1],'batch':event['batch']}},OUT/f'case{k:02d}.pt')
  results.append({'seed':key[0],'position':key[1],'source_identity_preserved':True,'original_sha_exact':True,'split_recompose_RGB_exact':True,'arms':metadata})
 n.save_json(OUT/'01_results.json',{'state':'complete','resources':resources,'cases':results,'seconds':time.monotonic()-start,
  'all_RGB_identity_checks_passed':True,'model_constructed':False,'model_forwards':0,'model_backwards':0,'new_training':False,
  'protocols':{f:n.sha(ROOT/f) for f in ['53_固定曝光布局_冻结检测首检协议.txt','54_阶段2渲染与整体裁紧控制_前向前补充.txt']}})
 n.save_json(OUT/'02_SHA256.json',{p.name:n.sha(p) for p in OUT.iterdir() if p.is_file()})
 print(json.dumps({'state':'complete','cases':len(selected),'seconds':time.monotonic()-start,'all_RGB_identity_checks_passed':True}))

if __name__=='__main__':main()
