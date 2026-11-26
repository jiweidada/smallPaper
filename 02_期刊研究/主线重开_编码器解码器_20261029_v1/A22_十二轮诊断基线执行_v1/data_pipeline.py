"""All valid original GT, exact fixed resize, deterministic per-image training flip."""
import hashlib
from pathlib import Path
from PIL import Image
import torch
from runtime_common import ROOT,TRAIN,rd,author_transform

def annotation_rows(path,width,height):
    rows=[]
    for uid,line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines()):
        if not line.strip():continue
        v=[int(x.strip()) for x in line.split(',') if x.strip()];assert len(v)==8
        x,y,w,h,score,cat,_,_=v
        l,r=max(0,min(width,x)),max(0,min(width,x+w))
        t,b=max(0,min(height,y)),max(0,min(height,y+h))
        if score>0 and 1<=cat<=10 and r>l and b>t:
            rows.append((uid,cat,l,t,r,b))
    return rows

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self,epoch,force_flip=None):
        self.rows=rd(ROOT/'train_asset_manifest.json')['files']
        self.epoch=epoch;self.transform=None;self.force_flip=force_flip
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];W,H=row['width'],row['height']
        if self.transform is None:self.transform=author_transform()
        with Image.open(TRAIN/'images'/row['image']) as im:
            assert im.size==(W,H);image=self.transform(im.convert('RGB')).squeeze(0)
        rows=annotation_rows(TRAIN/'annotations'/row['annotation'],W,H)
        assert len(rows)==row['valid_GT'] and rows
        labels=torch.tensor([r[1] for r in rows],dtype=torch.int64)
        boxes=torch.tensor([[(l+r)/(2*W),(t+b)/(2*H),(r-l)/W,(b-t)/H] for _,_,l,t,r,b in rows],dtype=torch.float32)
        flip=hashlib.sha256(f'42301|{self.epoch}|{row["image"]}|flip'.encode()).digest()[0]<128
        if self.force_flip is not None:flip=self.force_flip
        if flip:image=torch.flip(image,[-1]);boxes[:,0]=1.0-boxes[:,0]
        area=torch.tensor([(r-l)*800/W*(b-t)*800/H for _,_,l,t,r,b in rows],dtype=torch.float32)
        target={'labels':labels,'boxes':boxes,'area':area,'iscrowd':torch.zeros(len(rows),dtype=torch.int64),
            'image_id':torch.tensor([index],dtype=torch.int64),'orig_size':torch.tensor([W,H]),'size':torch.tensor([800,800])}
        return image,target,{'image':row['image'],'index':index,'flip':flip,'GT':len(rows)}

def collate(items):
    return torch.stack([x[0] for x in items]),[x[1] for x in items],[x[2] for x in items]

def pixel_edits(x,case):
    mean=x.double().mean(dim=(1,2),keepdim=True)
    masks=[case['neighbour_mask'],case['background_mask']];deltas=[];norms=[]
    for x1,y1,x2,y2 in masks:
        d=mean-x[:,y1:y2,x1:x2].double();deltas.append(d);norms.append(float(torch.linalg.vector_norm(d)))
    energy=min(norms);out=[]
    tx1,ty1,tx2,ty2=case['target_guard']
    for (x1,y1,x2,y2),d,norm in zip(masks,deltas,norms):
        alpha=energy/norm if norm else 0.0;y=x.clone()
        y[:,y1:y2,x1:x2]=(x[:,y1:y2,x1:x2].double()+alpha*d).float()
        assert torch.equal(y[:,ty1:ty2,tx1:tx2],x[:,ty1:ty2,tx1:tx2]);out.append(y)
    return out,energy
