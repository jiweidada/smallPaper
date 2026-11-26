"""Raw source inventory. No ML/model imports, training, fitting or forwards."""
import json, hashlib, collections, re, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
E = ROOT / 'data/00_实验计划与证据索引'
DATA = Path('C:/Users/jiwei/Desktop/DEIM/dataset_visdrone')

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()

def item_shape(x):
    if isinstance(x,dict):
        return {k:(len(v) if isinstance(v,(list,dict)) else v) for k,v in x.items()}
    return {'type':type(x).__name__, 'length':len(x) if hasattr(x,'__len__') else None,
            'first':x[0] if isinstance(x,list) and x else None}

started=time.perf_counter()
report={'scope':'Raw train/val metadata and mature132 final EMA prediction inventory; no model import',
        'datasets':{},'predictions':{},'sources':{},'curated_history_keyword_hits':[]}
splits={}
for split in ['train','val']:
    p=DATA/f'VisDrone2019-DET-{split}'/'annotations'/f'{split}.json'
    d=read(p);report['sources'][str(p)]=sha(p)
    ims={r['id']:r for r in d['images']}
    byim=collections.defaultdict(list)
    for a in d['annotations']:byim[a['image_id']].append(a)
    groups=collections.defaultdict(list)
    cats=collections.Counter()
    perclass_groups=collections.defaultdict(set)
    flags=collections.Counter()
    for im in ims.values():
        prefix=Path(im['file_name']).stem.split('_')[0]
        groups[prefix].append(im['id'])
    for a in d['annotations']:
        cats[a['category_id']]+=1
        prefix=Path(ims[a['image_id']]['file_name']).stem.split('_')[0]
        perclass_groups[a['category_id']].add(prefix)
        x,y,w,h=a['bbox']
        flags['nonpositive_box']+=int(w<=0 or h<=0)
        flags['iscrowd']+=int(a.get('iscrowd',0)!=0)
        flags['ignore']+=int(a.get('ignore',0)!=0)
        flags['touch_image_edge']+=int(x<=0 or y<=0 or x+w>=ims[a['image_id']]['width'] or y+h>=ims[a['image_id']]['height'])
    report['datasets'][split]={
        'images':len(ims),'GT':len(d['annotations']),'categories':d['categories'],
        'first_image':next(iter(ims.values())),'first_GT':d['annotations'][0],
        'class_counts':dict(cats),'flags':dict(flags),
        'filename_prefix_groups':len(groups),'filename_prefix_group_sizes':dict(sorted((k,len(v)) for k,v in groups.items())),
        'class_prefix_group_count':{str(k):len(v) for k,v in perclass_groups.items()},
        'image_counts_by_GT':dict(sorted(collections.Counter(len(byim[i]) for i in ims).items())),
        'prefix_meaning':'Filename token only; not independently verified as scene, flight, video or independent sampling unit'}
    splits[split]={'images':ims,'groups':groups}
report['train_val_common_filename_prefixes']=sorted(set(splits['train']['groups'])&set(splits['val']['groups']))
report['train_val_same_full_filename']=sorted({r['file_name'] for r in splits['train']['images'].values()} & {r['file_name'] for r in splits['val']['images'].values()})

base=E/'正式132轮基线整图双问题诊断_20261008_v1/native_transform_v2'
for seed in [1,2]:
    sd=base/f'seed{seed}'
    p=sd/'predictions.json';pred=read(p)
    report['sources'][str(p)]=sha(p)
    report['sources'][str(sd/'evaluation.json')]=sha(sd/'evaluation.json')
    report['predictions'][str(seed)]={'rows':len(pred),'first':pred[0],
        'image_count':len({r['image_id'] for r in pred}),
        'evaluation':read(sd/'evaluation.json'),
        'completion':read(sd/'completion.json')}
    zpath=sd/'arrays/000000.npz'
    # Metadata only; no load of model tensors/checkpoints.
    import zipfile
    with zipfile.ZipFile(zpath) as z:report['predictions'][str(seed)]['array_keys']=[n[:-4] for n in z.namelist()]

pat=re.compile('跨图|跨场景|跨序列|图像级|图级|全局排序|亮度|序列泄漏|近重复')
excluded={'多模型协作_20261005','专家组讨论_Claude_GPT_20261007'}
for dr in E.iterdir():
    if not dr.is_dir() or dr.name in excluded:continue
    for p in dr.iterdir():
        if not p.is_file() or p.suffix not in ['.txt','.md'] or p.stat().st_size>65000:continue
        for li,line in enumerate(p.read_text(encoding='utf-8-sig',errors='replace').splitlines(),1):
            if pat.search(line):
                report['curated_history_keyword_hits'].append({'file':str(p),'line':li,'text':line[:600]})
report['elapsed_seconds']=time.perf_counter()-started
out=HERE/'01_原始来源盘点.json'
out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'output':str(out),'datasets':{k:{t:v[t] for t in ['images','GT','class_counts','flags','filename_prefix_groups','class_prefix_group_count']} for k,v in report['datasets'].items()},
    'common_prefixes':len(report['train_val_common_filename_prefixes']), 'same_full_filenames':len(report['train_val_same_full_filename']),
    'predictions':{k:{t:v[t] for t in ['rows','image_count','array_keys']} for k,v in report['predictions'].items()},
    'history_hits':len(report['curated_history_keyword_hits']),'seconds':report['elapsed_seconds']},ensure_ascii=False,indent=2))
