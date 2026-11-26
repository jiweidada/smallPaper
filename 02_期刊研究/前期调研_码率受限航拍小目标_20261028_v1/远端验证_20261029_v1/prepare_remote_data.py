from pathlib import Path
from collections import Counter
import hashlib,json,math,tarfile,time
from PIL import Image
ROOT=Path('/root/c10_region_restore_20261029_v1')
manifest=json.loads((ROOT/'assets/dataset_source_manifest.json').read_text())
archive=ROOT/'assets/visdrone_train_source.tar'
archive_digest=hashlib.sha256()
with archive.open('rb') as f:
    for chunk in iter(lambda:f.read(8*1024*1024),b''):archive_digest.update(chunk)
assert archive_digest.hexdigest()==manifest['archive_sha256']
if not (ROOT/'dataset/train/images').exists():
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            p=(ROOT/m.name).resolve()
            assert str(p).startswith(str(ROOT)+'/dataset/train/') and m.isfile() and not m.issym() and not m.islnk()
        tar.extractall(ROOT)
names=sorted(row['file_name'] for row in manifest['files'])
assert len(names)==6471
ranked=sorted(names,key=lambda s:(hashlib.sha256(('compress-dev-v1:'+s).encode()).hexdigest(),s))
dev=ranked[:300]; fit=sorted(set(names)-set(dev))
subsets={'fit':fit,'dev':dev,'diagnostic':dev[:100],'tuning':dev[100:200],'confirmation':dev[200:]}
labels_dir=ROOT/'dataset/train/labels';labels_dir.mkdir(exist_ok=True)
categories=['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']
counts=Counter(); all_images=[]; all_annotations=[]; invalid=[]; ann_id=0
source_map={r['file_name']:r for r in manifest['files']}
for image_id,name in enumerate(names):
    p=ROOT/'dataset/train/images'/name
    txt=ROOT/'dataset/train/annotations'/(p.stem+'.txt')
    for folder,file in [('images',p),('annotations',txt)]:
        assert hashlib.sha256(file.read_bytes()).hexdigest()==source_map[name][folder]['sha256']
    with Image.open(p) as im:width,height=im.size
    all_images.append({'id':image_id,'file_name':name,'width':width,'height':height})
    rows=[]
    for row_id,line in enumerate(txt.read_text(encoding='utf-8-sig').splitlines(),1):
        if not line.strip():continue
        f=[float(v) for v in line.rstrip(',').split(',')]
        assert len(f)>=8 and all(math.isfinite(x) for x in f)
        x,y,w,h,score,cat=f[:6];cat=int(cat)
        x1,y1=max(0,min(width,x)),max(0,min(height,y));x2,y2=max(0,min(width,x+w)),max(0,min(height,y+h))
        if x2<=x1 or y2<=y1:
            counts['dropped_nonpositive_after_clipping']+=1;invalid.append({'file_name':name,'row':row_id,'bbox':f[:4],'category':cat});continue
        positive=int(score)==1 and 1<=cat<=10
        if positive:
            values=[(x1+x2)/2/width,(y1+y2)/2/height,(x2-x1)/width,(y2-y1)/height]
            assert all(0<=v<=1 for v in values)
            rows.append(str(cat-1)+' '+' '.join(format(v,'.10g') for v in values))
            counts['training_positive']+=1
        # Internal COCO-style ignore policy: 0/11 cover all classes; score=0 covers own valid class.
        target_categories=[cat-1] if 1<=cat<=10 else list(range(10)) if cat in (0,11) else []
        for category in target_categories:
            all_annotations.append({'id':ann_id,'image_id':image_id,'category_id':category,
                'bbox':[x1,y1,x2-x1,y2-y1],'area':(x2-x1)*(y2-y1),'iscrowd':int(not positive),
                'source_row':row_id,'stable_uid':name+':'+str(row_id)})
            ann_id+=1
    (labels_dir/(p.stem+'.txt')).write_text('\n'.join(rows)+'\n')
for key,files in subsets.items():
    (ROOT/('dataset/'+key+'_images.txt')).write_text('\n'.join(str(ROOT/'dataset/train/images'/n) for n in files)+'\n')
    ids={im['id'] for im in all_images if im['file_name'] in set(files)}
    if key!='fit':
        obj={'info':{'description':'internal COCO-style VisDrone, not official challenge score'},'licenses':[],
             'images':[im for im in all_images if im['id'] in ids],
             'annotations':[a for a in all_annotations if a['image_id'] in ids],
             'categories':[{'id':i,'name':n} for i,n in enumerate(categories)]}
        (ROOT/('dataset/'+key+'_gt_orig.json')).write_text(json.dumps(obj))
yaml='path: '+str(ROOT/'dataset')+'\ntrain: fit_images.txt\nval: dev_images.txt\nnames:\n'+''.join('  '+str(i)+': '+n+'\n' for i,n in enumerate(categories))
(ROOT/'dataset/data.yaml').write_text(yaml)
split={'algorithm':"SHA256('compress-dev-v1:'+filename), hash then filename ascending",'subsets':subsets,
       'counts':{k:len(v) for k,v in subsets.items()},'fit_dev_overlap':len(set(fit)&set(dev)),
       'scene_disjoint_claimed':False,'test_or_test_dev_used':False}
(ROOT/'receipts/split_manifest.json').write_text(json.dumps(split,indent=2))
result={'status':'PASS','archive_sha256':archive_digest.hexdigest(),'source_file_hashes_verified':6471*2,
        'counts':dict(counts),'invalid_boxes':invalid,'split_counts':split['counts'],
        'raw_data_unchanged':True,'labels_rule':'score1/category1..10, clip native xyxy then discard nonpositive, map category-1',
        'evaluation_ignore_rule':'internal COCO crowd; 0/11 copied across10 categories, score0 valid category crowd in own class',
        'classes':categories}
(ROOT/'receipts/data_preparation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
