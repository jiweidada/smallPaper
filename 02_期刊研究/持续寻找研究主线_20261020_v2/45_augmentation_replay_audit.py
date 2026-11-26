"""Data-only instrument parity and source/transform availability; no detector."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import sys,json,random,hashlib,pathlib,types,copy,time,datetime,traceback
ROOT=pathlib.Path(__file__).resolve().parent; WORK=ROOT.parents[1]
CODE=WORK/'英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1'
sys.path.insert(0,str(CODE))
import common as n
sys.path.append(str(WORK/'英文小论文/code/distribution_context_decoder_20261004_v1/local_deps'))
from engine.data.transforms.container import Compose
from engine.data.dataset.coco_dataset import CocoDetection
from engine.data.dataloader import BatchImageCollateFunction
import torchvision
from PIL import Image
torch,np=n.torch,n.np
OUT=ROOT/'45_augmentation_replay_audit'
CFG=WORK/'英文小论文/data/00_实验计划与证据索引/本次24轮双基线模型下载_20261009_v1/seed1/bundle/resolved.yml'
ANN=WORK/'outputs/scale_protection12_20261010_v1/terminal_collection/seed1_target_protect_v1/extracted/reference/train_annotations.json'
IMAGES=pathlib.Path('D:/数据集/visdrone2019/VisDrone2019-DET-train/images')

def plain(v):
 if isinstance(v,torch.Tensor):return v.detach().cpu().tolist()
 if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [plain(x) for x in v]
 if isinstance(v,(str,int,float,bool)) or v is None:return v
 return str(v)

def digest(v):
 if isinstance(v,Image.Image):return hashlib.sha256(v.tobytes()).hexdigest()
 return hashlib.sha256(v.as_subclass(torch.Tensor).contiguous().numpy().tobytes()).hexdigest()

def rng():
 return {'python':hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
         'numpy':hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
         'torch':digest(torch.get_rng_state())}

def reset(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def main():
 started=time.monotonic();OUT.mkdir(exist_ok=False)
 resource=n.resource_limits()
 assert n.sha(ANN)=='32c1cae852e084430ea74503eb660f0707ec6b5e65d0998ce3ec8b448265c883'
 y=json.loads(CFG.read_text(encoding='utf8'))
 cfg=copy.deepcopy(y['train_dataloader']['dataset']['transforms']);cfg.pop('type')
 trans=Compose(**cfg)
 dataset=CocoDetection(str(IMAGES),str(ANN),trans,num_classes=10)
 c=copy.deepcopy(y['train_dataloader']['collate_fn']);c.pop('type')
 collate=BatchImageCollateFunction(**c)
 dataset.set_epoch(4);collate.set_epoch(4)
 assert len(dataset)==6471 and not dataset.cache_flag
 mosaic=next(v for v in trans.transforms if type(v).__name__=='Mosaic')
 assert mosaic.use_cache is False
 positions=[round(i*(len(dataset)-1)/127) for i in range(128)]
 old=json.loads((WORK/'outputs/scale_protection12_20261010_v1/data_collection_v1/diagnostics/data_probe.json').read_text(encoding='utf8'))
 record={'enabled':False,'tagged':False,'current':None}; traces=[]
 original_load=dataset.load_item
 def load(index):
  im,t=original_load(index)
  if record['enabled']:
   record['current']['sources'].append({'index':int(index),'image_id':int(dataset.ids[index]),
       'raw_size':list(im.size),'file_name':dataset.coco.imgs[dataset.ids[index]]['file_name'],
       'raw_pixel_sha256':digest(im),'raw_GT':len(t['labels'])})
  if record['tagged']:
   anns=[a for a in dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=[dataset.ids[index]])) if not a.get('iscrowd',0)]
   box=torch.tensor([a['bbox'] for a in anns],dtype=torch.float32).reshape(-1,4)
   box[:,2:]+=box[:,:2];box[:,0::2].clamp_(0,im.width);box[:,1::2].clamp_(0,im.height)
   kept=(box[:,2:]>box[:,:2]).all(1)
   valid=[a for a,k in zip(anns,kept.tolist()) if k]
   assert torch.equal(box[kept],t['boxes'].as_subclass(torch.Tensor))
   assert [a['category_id'] for a in valid]==t['labels'].tolist()
   t['labels']=torch.tensor([a['id']+1 for a in valid],dtype=torch.int64)
  return im,t
 dataset.load_item=load
 # Instance methods keep each native transform's type and its original dispatch.
 saved=[]
 for name,module in trans.named_modules():
  if hasattr(module,'_get_params'):
   method=module._get_params
   def wrap(self,flat,_method=method,_name=name):
    result=_method(flat)
    if record['enabled']:
     images=[v for v in flat if isinstance(v,Image.Image)]
     record['current']['params'].append({'module':_name,'class':type(self).__name__,
       'input_sizes':[list(v.size) for v in images],'params':plain(result)})
    return result
   saved.append((module,method));module._get_params=types.MethodType(wrap,module)
 source_to_class={a['id']+1:a['category_id'] for a in dataset.coco.anns.values()}
 normal={};results={};historical=[];snapshots=0
 for seed in [1,2]:
  for mode in ['plain','recorded','tagged']:
   reset(seed);record['enabled']=mode=='recorded';record['tagged']=mode=='tagged'
   batches=[];events=[]
   for batch in range(32):
    assert time.monotonic()-started<900
    assert n.psutil.virtual_memory().available/2**30>8
    items=[];sample_rows=[]
    for pos in positions[batch*4:batch*4+4]:
     event={'seed':seed,'batch':batch,'position':pos,'sources':[],'params':[]}
     record['current']=event
     im,t=dataset[pos]
     labels=t['labels'].tolist()
     decoded=[source_to_class[a] for a in labels] if mode=='tagged' else labels
     sample_rows.append({'image_sha256':digest(im),'boxes_sha256':digest(t['boxes']),
       'classes':decoded,'source_ids':t.get('mosaic_source_image_ids',torch.empty(0,dtype=torch.int64)).tolist()})
     if mode=='recorded':
      event['final_GT']=plain(t['boxes']);event['final_classes']=decoded
      event['source_ids']=sample_rows[-1]['source_ids'];event['final640_sha256']=digest(im)
      events.append(event)
      if len(event['source_ids'])==4 and sum(e['sources']!=[] for e in events)<=128:
       # Save the first eight Mosaic image outputs per seed, independent of model outcomes.
       prior=sum(1 for e in events[:-1] if len(e['source_ids'])==4)
       if prior<8:
        torch.save({'image':im.as_subclass(torch.Tensor).clone(),'targets':n.cpu_tree(t),'event':event},OUT/f'seed{seed}_mosaic{prior:02d}.pt');snapshots+=1
     items.append((im,t))
    images,targets=collate(items)
    lab=[t['labels'].tolist() for t in targets]
    if mode=='tagged':lab=[[source_to_class[a] for a in ids] for ids in lab]
    row={'samples':sample_rows,'image_sha256':digest(images),
      'boxes_sha256':[digest(t['boxes']) for t in targets],'classes':lab,
      'mixup':[plain(t.get('mixup',None)) for t in targets], 'shape':list(images.shape),'rng':rng()}
    batches.append(row)
    if mode!='plain':assert row==normal[seed][batch],(seed,mode,batch,'instrument/tag parity')
    if mode=='plain':
     ref=old['modes'][f'{seed}:native']['ordinary_hashes'][batch]
     historical.append({'seed':seed,'batch':batch,'image_equal':row['image_sha256']==ref['batch']['image'],
       'boxes_equal':row['boxes_sha256']==ref['batch']['boxes'],
       'classes_equal':row['classes']==ref['batch']['labels'],'shape_equal':row['shape']==ref['batch']['shape']})
   if mode=='plain':normal[seed]=batches
   if mode=='recorded':traces+=events
   results[f'{seed}:{mode}']={'batches':len(batches),'final_rng':rng(),'Mosaic_events':sum(len(e['source_ids'])==4 for e in events)}
   print(json.dumps({'seed':seed,'mode':mode,'batches':32,'seconds':round(time.monotonic()-started,2)}),flush=True)
 n.save_json(OUT/'01_transform_traces.json',traces)
 n.save_json(OUT/'02_plain_batch_identity.json',normal)
 n.save_json(OUT/'03_results.json',{'state':'complete','resources':resource,'torchvision':torchvision.__version__,
   'torchvision_path':torchvision.__file__,'annotation_sha256':n.sha(ANN),'config_sha256':n.sha(CFG),
   'positions':positions,'new_fixture':True,'historical_batch_checks':historical,
   'instrumented_plain_exact':True,'tagged_plain_exact':True,'passes':results,'saved_Mosaic_snapshots':snapshots,
   'model_constructed':False,'model_forwards':0,'model_backwards':0,'optimizer_steps':0,'heldout_reads':False,
   'seconds':time.monotonic()-started,'scope':'Availability/parity audit, no new augmentation candidate or detection benefit evaluated.'})
 n.save_json(OUT/'04_sha256.json',{str(p.relative_to(OUT)):n.sha(p) for p in OUT.iterdir() if p.is_file()})
 print('COMPLETE',flush=True)

if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True)
  n.save_json(OUT/'failure.json',{'traceback':traceback.format_exc(),'new_training':False})
  raise
