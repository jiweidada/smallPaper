import pathlib,hashlib,random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
SEED=20261101
DATA=pathlib.Path('/25085400120/DEIM/work/DEIM_1572862469/dataset_visdrone')
TRAIN_IMAGES=DATA/'VisDrone2019-DET-train/images'
TRAIN_ANNS=DATA/'VisDrone2019-DET-train/annotations'
VAL_IMAGES=pathlib.Path('/root/uav_a48_20261101_v1/val_inputs')

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

def seed_all(seed=SEED):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def fp32_tree(x):
    if torch.is_tensor(x):return x.float() if x.is_floating_point() else x
    if isinstance(x,dict):return {k:fp32_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [fp32_tree(v) for v in x]
    if isinstance(x,tuple):return tuple(fp32_tree(v) for v in x)
    return x
