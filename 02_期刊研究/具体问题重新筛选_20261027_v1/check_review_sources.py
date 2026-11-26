"""Post-hoc source checks of frozen reviewed cases; no model/GT changes."""
import json, hashlib, collections, csv
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
VAL=Path('C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val')
BASE=ROOT/'data/00_实验计划与证据索引/正式132轮基线整图双问题诊断_20261008_v1/native_transform_v2'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def ov(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float).reshape(-1,4)
    inter=np.maximum(0,np.minimum(a[:2]+a[2:],b[:,:2]+b[:,2:])-np.maximum(a[:2],b[:,:2])).prod(1)
    return inter/np.maximum(a[2:].prod()+b[:,2:].prod(1)-inter,1e-300),inter/max(a[2:].prod(),1e-300)
ds=read(VAL/'annotations/val.json');by=collections.defaultdict(list)
for a in ds['annotations']:by[a['image_id']].append(a)
sample=read(HERE/'03_漏标审计_样本与总体.json')
first={v['case_id']:v for v in csv.DictReader((HERE/'07_原像素首轮观察.csv').open(encoding='utf-8'))}
preds={s:read(BASE/f'seed{s}/predictions.json') for s in [1,2]}
pb={s:collections.defaultdict(list) for s in [1,2]}
for s in [1,2]:
    for k,d in enumerate(preds[s]):pb[s][d['image_id']].append(k)
rows=[]
for v in sample['cases']:
    if v['review_role']!='candidate' or first[v['case_id']]['pixel_reading']!='V':continue
    p=VAL/'annotations'/Path(v['file_name']).with_suffix('.txt')
    raw=np.array([[float(x) for x in ln.split(',')] for ln in p.read_text().splitlines() if ln.strip()])
    I,A=ov(v['bbox'],raw[:,:4]);cx=v['bbox'][0]+v['bbox'][2]/2;cy=v['bbox'][1]+v['bbox'][3]/2
    inc=(raw[:,0]<=cx)&(cx<=raw[:,0]+raw[:,2])&(raw[:,1]<=cy)&(cy<=raw[:,1]+raw[:,3])
    z={'case_id':v['case_id'],'sample_seed':v['seed'],'image_id':v['image_id'],'file_name':v['file_name'],
        'predicted_class':v['category_id'],'first_pixel_note':first[v['case_id']]['observation'],
        'raw_TXT_path':str(p),'raw_TXT_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'raw_lines':len(raw),'converted_GT':len(by[v['image_id']]),'raw_class_counts':dict(collections.Counter(raw[:,5].astype(int).tolist())),
        'max_IoU_all_raw_rows':float(I.max(initial=0)),'max_detection_IoA_all_raw_rows':float(A.max(initial=0)),
        'raw_rows_containing_prediction_center':[{'line1':int(j)+1,'row':raw[j].tolist()} for j in np.flatnonzero(inc)],
        'other_seed_fixed_native_overlap':{}}
    for s in [1,2]:
        kk=sorted(pb[s][v['image_id']],key=lambda k:-preds[s][k]['score']);nc=collections.Counter();kept=[]
        for k in kk:
            c=preds[s][k]['category_id']
            if nc[c]<100:kept.append(k)
            nc[c]+=1
        same=[k for k in kept if preds[s][k]['category_id']==v['category_id']]
        M,_=ov(v['bbox'],[preds[s][k]['bbox'] for k in same])
        cand=[(k,float(u)) for k,u in zip(same,M) if u>=.5]
        cand=sorted(cand,key=lambda t:-preds[s][t[0]]['score'])
        z['other_seed_fixed_native_overlap'][str(s)]=[{'prediction_row':k,'score':preds[s][k]['score'],'IoU_with_selected_prediction_box':u,'bbox':preds[s][k]['bbox']} for k,u in cand]
    rows.append(z)
output={'scope':'Post-hoc raw-text/context and cross-seed spatial consistency checks of V candidate occurrences. Selected prediction boxes are not newly annotated GT; agreement is not truth or actual TP.',
    'candidates':rows,'new_model_forwards':0,'training':0,'changed_source_GT':False}
(HERE/'08_可见目标候选_原TXT与双seed核查.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'candidates':[{k:z[k] for k in ['case_id','sample_seed','image_id','predicted_class','raw_lines','max_IoU_all_raw_rows','max_detection_IoA_all_raw_rows']} | {'overlap_found_both_seeds':all(z['other_seed_fixed_native_overlap'][str(s)] for s in [1,2])} for z in rows]},ensure_ascii=False,indent=2))
