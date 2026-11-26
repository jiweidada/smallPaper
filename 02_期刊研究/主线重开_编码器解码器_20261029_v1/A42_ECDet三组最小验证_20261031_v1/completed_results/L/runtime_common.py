from __future__ import annotations
import contextlib,hashlib,io,json,os,pathlib,random,sys,time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from PIL import Image

R=pathlib.Path(__file__).resolve().parent
SOURCE=R/'source/EdgeCrafter-d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecdetseg'
sys.path.insert(0,str(SOURCE))
from engine.core import YAMLConfig
from detail_reader import StemDetailReader

SEED=20261031
MODEL_COUNTER={'forwards':0}
BASE=pathlib.Path('/root/c10_region_restore_20261029_v1')
TRAIN_IMAGES=BASE/'dataset/train/images'
TRAIN_ANNS=BASE/'dataset/train/annotations'
VAL_IMAGES=BASE/'architecture_probe_v1/a22_vssd_dome_pilot12_20261030_v1/val_inputs'

def digest(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for v in iter(lambda:f.read(1<<20),b''):h.update(v)
    return h.hexdigest()

def save_json(p,x):
    p=pathlib.Path(p);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8');os.replace(temp,p)

def seed_all(seed=SEED):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def create_components(arm='B'):
    assert arm in ['B','S','L']
    seed_all()
    cfg=YAMLConfig(str(SOURCE/'configs/ecdet/ecdet_s.yml'))
    cfg.yaml_cfg.update(num_classes=10,remap_mscoco_category=False)
    cfg.yaml_cfg['ViTAdapter']['skip_load_backbone']=True
    model=cfg.model
    def count_forward(module,args):MODEL_COUNTER['forwards']+=1
    model.register_forward_pre_hook(count_forward)
    initial=torch.load(R/'common_init_10class.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(initial['model'],strict=True)
    inner=model.decoder.decoder
    inner.detail_mode=arm
    inner.detail_reader=None
    if arm!='B':
        torch.manual_seed(SEED+1)
        reader=StemDetailReader(channels=48,hidden_dim=192)
        inner.detail_reader=reader
        inner.detail_feature=None
        def store_feature(module,args,output):
            inner.detail_feature=output
        model.backbone.backbone.patch_embed.convs[1].register_forward_hook(store_feature)
        def clear_feature(module,args,output):
            inner.detail_feature=None
        model.register_forward_hook(clear_feature)
    return model,cfg.criterion,cfg.postprocessor

def fp32_tree(x):
    if torch.is_tensor(x):return x.float() if x.is_floating_point() else x
    if isinstance(x,dict):return {k:fp32_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [fp32_tree(v) for v in x]
    if isinstance(x,tuple):return tuple(fp32_tree(v) for v in x)
    return x

def make_optimizer(model):
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        lr=2.5e-5 if name.startswith('backbone.') else 5e-4
        wd=0.0 if any(v in name for v in ['norm','bn','bias']) else 1e-4
        groups.setdefault((lr,wd),[]).append(p)
    return torch.optim.AdamW([{'params':ps,'lr':lr,'initial_lr':lr,'weight_decay':wd}for (lr,wd),ps in groups.items()],betas=(0.9,0.999))

def boxes_from_annotation(path,width,height):
    valid=[];ignore=[];omitted={}
    for line in pathlib.Path(path).read_text().splitlines():
        z=[float(v.strip()) for v in line.split(',') if v.strip()]
        if len(z)<8:raise ValueError('Malformed VisDrone annotation')
        x,y,w,h,score,cls,trunc,occ=z[:8]
        x0=max(0.0,min(width,x));y0=max(0.0,min(height,y))
        x1=max(0.0,min(width,x+w));y1=max(0.0,min(height,y+h))
        if x1<=x0 or y1<=y0:continue
        if int(cls)==0:ignore.append([x0,y0,x1,y1])
        if score>0 and 1<=int(cls)<=10:valid.append([x0,y0,x1,y1,int(cls)-1])
        else:omitted[str(int(cls))]=omitted.get(str(int(cls)),0)+1
    return valid,ignore,omitted

class VisDroneDataset(Dataset):
    def __init__(self,split='train'):
        self.split=split;self.epoch=0
        self.images=TRAIN_IMAGES if split=='train' else VAL_IMAGES
        self.anns=TRAIN_ANNS if split=='train' else VAL_IMAGES
        self.rows=[]
        for i,p in enumerate(sorted(self.images.glob('*.jpg'))):
            with Image.open(p) as image:w,h=image.size
            ann=self.anns/(p.stem+'.txt')
            valid,ignore,omitted=boxes_from_annotation(ann,w,h)
            self.rows.append({'id':i+1,'file':p.name,'width':w,'height':h,'valid':valid,'ignore':ignore,'omitted':omitted,'annotation':str(ann)})

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        r=self.rows[index]
        with Image.open(self.images/r['file']) as src:image=src.convert('RGB')
        boxes=torch.tensor([z[:4] for z in r['valid']],dtype=torch.float32).reshape(-1,4)
        labels=torch.tensor([int(z[4]) for z in r['valid']],dtype=torch.int64)
        flip=False
        if self.split=='train':
            raw=hashlib.sha256((str(SEED)+':'+str(self.epoch)+':'+r['file']).encode()).digest()
            flip=bool(raw[0]&1)
        if flip:
            image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            boxes[:,[0,2]]=r['width']-boxes[:,[2,0]]
        image=image.resize((640,640),Image.Resampling.BILINEAR)
        pix=np.asarray(image,dtype=np.float32)/255.0
        tensor=torch.from_numpy(pix.transpose(2,0,1).copy())
        tensor=(tensor-tensor.new_tensor([.485,.456,.406])[:,None,None])/tensor.new_tensor([.229,.224,.225])[:,None,None]
        norm=boxes/boxes.new_tensor([r['width'],r['height'],r['width'],r['height']])
        cxcy=(norm[:,:2]+norm[:,2:])/2;wh=norm[:,2:]-norm[:,:2]
        target={'boxes':torch.cat([cxcy,wh],dim=-1),'labels':labels,'image_id':torch.tensor(r['id']),'orig_size':torch.tensor([r['width'],r['height']])}
        return tensor,target

def collate(batch):return torch.stack([v[0] for v in batch]),[v[1] for v in batch]

def targets_to_device(targets,device):return [{k:v.to(device) for k,v in t.items()} for t in targets]

def data_manifest(dataset):
    rows=[]
    for r in dataset.rows:
        rows.append({'id':r['id'],'file':r['file'],'width':r['width'],'height':r['height'],'valid_gt':len(r['valid']),'image_sha256':digest(dataset.images/r['file']),'annotation_sha256':digest(r['annotation'])})
    return {'split':dataset.split,'images':len(rows),'valid_gt':sum(r['valid_gt'] for r in rows),'root':str(dataset.images),'rows':rows,'annotation_policy':'score>0,classes1..10,clip to image; class0 rectangles retained only for evaluation ignore filtering; others11 excluded from scored categories'}
