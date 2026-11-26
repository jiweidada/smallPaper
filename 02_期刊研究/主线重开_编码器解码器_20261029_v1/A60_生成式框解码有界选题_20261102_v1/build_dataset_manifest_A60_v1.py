"""One read-only split/header/annotation manifest before the training freeze."""
from pathlib import Path
from PIL import Image
import hashlib
import json
import time

ROOT=Path('/root/uav_a60_20261102_v1')
TRAIN=Path('/25085400120/DEIM/work/DEIM_1572862469/dataset_visdrone/VisDrone2019-DET-train')
VAL=Path('/root/uav_a48_20261101_v1/val_inputs')

def main():
    dest=ROOT/'dataset_manifest_v1.json';assert not dest.exists();start=time.monotonic()
    splits={}
    for split,images,anns,expected in [('train',TRAIN/'images',TRAIN/'annotations',6471),('val',VAL,VAL,548)]:
        names=sorted(p.name for p in images.glob('*.jpg'));assert len(names)==expected
        rows=[]
        for i,name in enumerate(names):
            ip=images/name;ap=anns/(Path(name).stem+'.txt');raw=ap.read_bytes()
            with Image.open(ip) as im:w,h=im.size
            valid=[];ignore=[]
            for line in raw.decode().splitlines():
                z=[float(v.strip()) for v in line.split(',') if v.strip()];assert len(z)>=8
                x,y,bw,bh,score,cat=z[:6]
                a=max(0.,min(w,x));b=max(0.,min(h,y));c=max(0.,min(w,x+bw));d=max(0.,min(h,y+bh))
                if c<=a or d<=b:continue
                if int(cat)==0:ignore.append([a,b,c,d])
                if score>0 and 1<=int(cat)<=10:valid.append([a,b,c,d,int(cat)-1])
            st=ip.stat()
            row=dict(id=i+1,file=name,width=w,height=h,image_bytes=st.st_size,image_mtime_ns=st.st_mtime_ns,
                annotation_sha256=hashlib.sha256(raw).hexdigest(),valid_GT=len(valid))
            if split=='val':row.update(valid=valid,ignore=ignore)
            rows.append(row)
        splits[split]=dict(image_dir=str(images),annotation_dir=str(anns),rows=rows,
            images=len(rows),valid_GT=sum(r['valid_GT'] for r in rows),max_GT=max(r['valid_GT'] for r in rows))
    assert splits['train']['max_GT']==902
    result=dict(status='FROZEN_SPLIT_HEADER_AND_ANNOTATION_MANIFEST',splits=splits,
        detector_forwards=0,optimizer_updates=0,header_reads=7019,annotation_file_reads=7019,
        image_hash_scope='Byte size/mtime and image dimensions frozen; full JPEG byte hashes not calculated. Actual augmented tensors are stream-hashed during training.',
        elapsed_seconds=time.monotonic()-start)
    with dest.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(status=result['status'],sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        bytes=dest.stat().st_size,elapsed_seconds=result['elapsed_seconds'],
        splits={k:{z:v[z] for z in ['images','valid_GT','max_GT']} for k,v in splits.items()})),flush=True)

if __name__=='__main__':main()
