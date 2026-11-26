"""One bounded annotation/header-only pass; no torch or detector imports."""
from pathlib import Path
import collections
import hashlib
import json
import math
import time
from PIL import Image

ROOT=Path('/root/uav_a54_20261101_v1/A58_uploads')
OUT=Path('/root/uav_a58_20261102_v1/annotation_geometry_v1')
DATA=Path('/25085400120/DEIM/work/DEIM_1572862469/dataset_visdrone')

def union_overlap_fraction(box,regions):
    x0,y0,x1,y1=[float(v) for v in box]
    area=max(0,x1-x0)*max(0,y1-y0)
    if area<=0:return 0.0
    rs=[]
    for r in regions:
        a,b,c,d=max(x0,r[0]),max(y0,r[1]),min(x1,r[2]),min(y1,r[3])
        if c>a and d>b:rs.append((a,b,c,d))
    xs=sorted({v for r in rs for v in (r[0],r[2])})
    covered=0.0
    for a,c in zip(xs[:-1],xs[1:]):
        spans=sorted((r[1],r[3]) for r in rs if r[0]<c and r[2]>a)
        end=None;height=0.0
        for lo,hi in spans:
            if end is None:height+=hi-lo;end=hi
            elif hi>end:height+=hi-max(lo,end);end=hi
        covered+=(c-a)*height
    return covered/area

def quantile(values,q):
    v=sorted(values)
    if not v:return None
    loc=(len(v)-1)*q;i=int(math.floor(loc));j=int(math.ceil(loc))
    return v[i]+(v[j]-v[i])*(loc-i)

def summary(rows):
    occupancy=[x[0] for x in rows]
    return dict(objects=len(rows),own_bbox_occupancy_quantiles={str(q):quantile(occupancy,q) for q in [0,.1,.25,.5,.75,.9,.95,1]},
        own_bbox_fraction_at_most={str(t):dict(count=sum(v<=t for v in occupancy),fraction=None if not rows else sum(v<=t for v in occupancy)/len(rows)) for t in [.0625,.125,.25,.5]},
        same_region_bbox_center_count={str(k):sum(r[1]==k for r in rows) for k in sorted({r[1] for r in rows})})

OUT.mkdir(exist_ok=False)
with (OUT/'run_once.lock').open('x') as f:f.write('ONE_ANNOTATION_PASS_NO_MODEL')
start=time.monotonic()
scope=json.loads((ROOT/'annotation_geometry_scope_FROZEN_v1.json').read_text())
result=dict(status='RUNNING',scope=scope,model_constructors=0,model_forwards=0,optimizer_updates=0,AP_evaluations=0,splits={})
for split in ['train','val']:
    d=DATA/('VisDrone2019-DET-'+split)
    images=sorted((d/'images').glob('*.jpg'))
    assert len(images)==scope['expected_images'][split]
    valid_count=ignored_count=0;all_rows=[];small_rows=[];perclass=collections.defaultdict(list);digest=hashlib.sha256()
    for path in images:
        with Image.open(path) as im:w,h=im.size
        ann=d/'annotations'/(path.stem+'.txt');raw=ann.read_bytes()
        digest.update(path.name.encode());digest.update(json.dumps([w,h]).encode());digest.update(raw)
        valid=[];ignore=[]
        for line in raw.decode().splitlines():
            z=[float(v.strip()) for v in line.split(',') if v.strip()]
            assert len(z)>=8
            x,y,bw,bh,score,cls,trunc,occ=z[:8]
            x0=max(0.,min(w,x));y0=max(0.,min(h,y));x1=max(0.,min(w,x+bw));y1=max(0.,min(h,y+bh))
            if x1<=x0 or y1<=y0:continue
            if int(cls)==0:ignore.append([x0,y0,x1,y1])
            if score>0 and 1<=int(cls)<=10:valid.append([x0,y0,x1,y1,int(cls)])
        valid_count+=len(valid)
        kept=[]
        for box in valid:
            if split=='val' and union_overlap_fraction(box[:4],ignore)>=.5:
                ignored_count+=1;continue
            kept.append(box)
        mapped=[]
        for x0,y0,x1,y1,cls in kept:
            original_area=(x1-x0)*(y1-y0)
            a,b,c,e=x0*640/w,y0*640/h,x1*640/w,y1*640/h
            gx=min(19,max(0,int(((a+c)/2)//32)));gy=min(19,max(0,int(((b+e)/2)//32)))
            ix=max(0.,min(c,(gx+1)*32)-max(a,gx*32));iy=max(0.,min(e,(gy+1)*32)-max(b,gy*32))
            frac=ix*iy/1024
            assert 0<=frac<=1
            mapped.append((frac,(gx,gy),original_area<=1024,cls))
        counts=collections.Counter(r[1] for r in mapped)
        for frac,cell,small,cls in mapped:
            row=(frac,counts[cell]);all_rows.append(row)
            if small:small_rows.append(row);perclass[cls].append(row)
    assert valid_count==scope['expected_valid_boxes_before_ignore'][split]
    if split=='val':assert len(all_rows)==scope['expected_val_scored_boxes'] and ignored_count==157
    result['splits'][split]=dict(images=len(images),valid_boxes_before_ignore=valid_count,
        ignored_valid_boxes=ignored_count,scored_or_training_boxes=len(all_rows),
        dimensions_and_annotation_stream_sha256=digest.hexdigest(),all_sizes=summary(all_rows),
        small_original_area=summary(small_rows),small_by_class={str(k):summary(v) for k,v in sorted(perclass.items())})
result.update(status='PASS_SINGLE_ANNOTATION_HEADER_PASS',monotonic_seconds=time.monotonic()-start,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope_sha256=hashlib.sha256((ROOT/'annotation_geometry_scope_FROZEN_v1.json').read_bytes()).hexdigest())
with (OUT/'result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result),flush=True)
