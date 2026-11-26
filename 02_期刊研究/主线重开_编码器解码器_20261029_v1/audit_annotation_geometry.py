from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import time

root = Path(__file__).resolve().parent
previous = root.parent / '前期调研_码率受限航拍小目标_20261028_v1' / '远端验证_20261029_v1'
protocol_path = root / '04_标注几何审计预定方案.json'
protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
manifest_path = previous / 'dataset_source_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
source = Path(manifest['source'])
metadata_path = root.parent / '相位机制论文_20261018' / '修订_20261018_v2' / 'data' / 'annotations' / 'train' / 'converted.json'
metadata_blob = metadata_path.read_bytes()
assert hashlib.sha256(metadata_blob).hexdigest() == manifest['source_metadata_sha256']
images = {im['file_name']: im for im in json.loads(metadata_blob)['images']}
names = sorted(row['file_name'] for row in manifest['files'])
manifest_lookup = {r['file_name']:r for r in manifest['files']}
assert len(names) == protocol['expected_contract']['train_images'] == len(images)
ranked = sorted(names, key=lambda n:(hashlib.sha256(('compress-dev-v1:'+n).encode()).hexdigest(), n))
memberships = {n: ['all_train6471', 'fit6171'] for n in names}
for i,n in enumerate(ranked[:300]):
    memberships[n] = ['all_train6471', ('diagnostic100' if i<100 else 'tuning100' if i<200 else 'confirmation100')]
out = root / 'annotation_geometry_audit_v1'
out.mkdir(exist_ok=True)
aggregates = defaultdict(Counter)
invalid = []
positive_total = 0
ann_digest = hashlib.sha256()
started = time.monotonic()
with (out/'per_image.csv').open('w',newline='',encoding='utf-8') as stream:
    fieldnames=['file_name','geometry','input_size','positives','size_lt8','size_8_16','size_16_32','size_ge32']
    fieldnames += [f'{metric}_s{s}' for s in protocol['grid_strides'] for metric in ('both_dim_lt_stride','center_shared_any','small_center_shared_small')]
    writer=csv.DictWriter(stream,fieldnames=fieldnames)
    writer.writeheader()
    for name in names:
        width,height=images[name]['width'],images[name]['height']
        annpath=source/'annotations'/(Path(name).stem+'.txt')
        blob=annpath.read_bytes()
        assert hashlib.sha256(blob).hexdigest()==manifest_lookup[name]['annotations']['sha256']
        ann_digest.update(name.encode()+b'\0'+blob)
        boxes=[]
        for row_number,line in enumerate(blob.decode('utf-8-sig').splitlines(),1):
            if not line.strip():continue
            f=[float(v) for v in line.rstrip(',').split(',')]
            assert len(f)>=8 and all(math.isfinite(v) for v in f)
            x,y,w,h,score,cat=f[:6]
            x1=max(0,min(width,x)); y1=max(0,min(height,y))
            x2=max(0,min(width,x+w)); y2=max(0,min(height,y+h))
            if x2<=x1 or y2<=y1:
                invalid.append({'file_name':name,'row':row_number,'category':int(cat),'score':int(score)})
                continue
            if int(score)==1 and 1<=int(cat)<=10:
                boxes.append((x1,y1,x2,y2))
        positive_total += len(boxes)
        for size in protocol['input_sizes']:
            for geometry in protocol['input_geometries']:
                if geometry=='square stretch':
                    sx,sy=size/width,size/height
                    dx=dy=0
                else:
                    sx=sy=min(size/width,size/height)
                    dx=round((size-round(width*sx))/2-.1)
                    dy=round((size-round(height*sy))/2-.1)
                scaled=[((x1+x2)*.5*sx+dx,(y1+y2)*.5*sy+dy,(x2-x1)*sx,(y2-y1)*sy) for x1,y1,x2,y2 in boxes]
                row={'file_name':name,'geometry':geometry,'input_size':size,'positives':len(scaled)}
                counts=Counter()
                for cx,cy,bw,bh in scaled:
                    d=math.sqrt(bw*bh)
                    counts['size_lt8' if d<8 else 'size_8_16' if d<16 else 'size_16_32' if d<32 else 'size_ge32']+=1
                row.update({k:counts[k] for k in fieldnames[4:8]})
                for stride in protocol['grid_strides']:
                    cells=Counter((int(cx//stride),int(cy//stride)) for cx,cy,bw,bh in scaled)
                    small_cells=Counter((int(cx//stride),int(cy//stride)) for cx,cy,bw,bh in scaled if bw*bh<1024)
                    row[f'both_dim_lt_stride_s{stride}']=sum(bw<stride and bh<stride for cx,cy,bw,bh in scaled)
                    row[f'center_shared_any_s{stride}']=sum(cells[(int(cx//stride),int(cy//stride))]>=2 for cx,cy,bw,bh in scaled)
                    row[f'small_center_shared_small_s{stride}']=sum(small_cells[(int(cx//stride),int(cy//stride))]>=2 for cx,cy,bw,bh in scaled if bw*bh<1024)
                writer.writerow(row)
                for subset in memberships[name]:
                    aggregate=aggregates[(subset,geometry,size)]
                    aggregate['images']+=1
                    for key,value in row.items():
                        if key not in ('file_name','geometry','input_size'):aggregate[key]+=value
assert positive_total==protocol['expected_contract']['valid_positives'], positive_total
results=[]
for (subset,geometry,size),counter in sorted(aggregates.items()):
    item={'subset':subset,'geometry':geometry,'input_size':size,**dict(counter)}
    n=counter['positives']; ns=n-counter['size_ge32']
    item['small_lt32_fraction']=ns/n if n else None
    for stride in protocol['grid_strides']:
        item[f'both_dim_lt_stride_fraction_s{stride}']=counter[f'both_dim_lt_stride_s{stride}']/n if n else None
        item[f'center_shared_any_fraction_s{stride}']=counter[f'center_shared_any_s{stride}']/n if n else None
        item[f'small_center_shared_small_fraction_s{stride}']=counter[f'small_center_shared_small_s{stride}']/ns if ns else None
    results.append(item)
receipt={'status':'PASS','recorded_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started,
    'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'source_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'metadata_sha256':hashlib.sha256(metadata_blob).hexdigest(),
    'annotation_files_sha256_verified':len(names),'annotations_ordered_content_sha256':ann_digest.hexdigest(),
    'source_images':len(names),'valid_positives':positive_total,'invalid_after_clipping':invalid,
    'model_forwards':0,'training_updates':0,'test_or_val_images_accessed':False,'results':results}
(out/'completed.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:receipt[k] for k in ('status','elapsed_seconds','source_images','valid_positives','annotation_files_sha256_verified')},ensure_ascii=False))
for row in results:
    if row['subset']=='all_train6471':
        print(json.dumps({k:v for k,v in row.items() if k in ('geometry','input_size','small_lt32_fraction','size_lt8') or k.startswith('center_shared_any_fraction') or k.startswith('small_center_shared_small_fraction')},ensure_ascii=False))
