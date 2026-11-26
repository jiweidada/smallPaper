"""Read existing test-dev matching caches only. No inference/training/AP redefinition."""
import csv,datetime,hashlib,json,math,pathlib
import numpy as np
WORK=pathlib.Path(__file__).resolve().parents[2]
PAPER=WORK/'相位机制论文_20261018'
CACHE=PAPER/'修订_20261018_v3/testdev_analysis/models'
ANN=PAPER/'修订_20261018_v2/data/annotations/test-dev/converted.json'
OUT=WORK/'data/00_实验计划与证据索引/C稿尺度代价只读初筛_20261020_v1'
OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ann=json.loads(ANN.read_text(encoding='utf8'));images={v['id']:v for v in ann['images']};classes={v['id']:v['name'] for v in ann['categories']}
valid=[g for g in ann['annotations'] if not g.get('ignore',0) and not g.get('iscrowd',0)]
groups={name:[] for name in ['original_COCO_large_ge96','input640_32to64','input640_ge64']}
for g in valid:
 w,h=g['bbox'][2:];im=images[g['image_id']];size=math.sqrt(w*h*640*640/(im['width']*im['height']))
 if g.get('area',w*h)>=96*96:groups['original_COCO_large_ge96'].append(g)
 if 32<=size<64:groups['input640_32to64'].append(g)
 if size>=64:groups['input640_ge64'].append(g)
source_files={str(ANN):sha(ANN)};matches={}
for seed in [0,1,2]:
 for arm in ['C','C-mean']+(['baseline'] if seed<2 else []):
  meta=json.loads((CACHE/f'seed{seed}_{arm}.json').read_text());assert meta['annotations_sha256']==sha(ANN)
  path=PAPER/'修订_20261018_v3'/meta['matches_npz'];assert sha(path)==meta['matches_sha256']
  source_files[str(path)]=sha(path)
  with np.load(path,allow_pickle=False) as z:matches[seed,arm]={t:{int(i):b for i,b in zip(z['ids_'+t],z['boxes_'+t])} for t in ['0.5','0.75']}
rows=[]
for seed in [0,1,2]:
 for ref in ['C-mean']+(['baseline'] if seed<2 else []):
  for t in ['0.5','0.75']:
   cm,rm=matches[seed,'C'][t],matches[seed,ref][t]
   for group,gt in groups.items():
    for cid in [None]+sorted(classes):
     selected=[g for g in gt if cid is None or g['category_id']==cid]
     common=sum(g['id'] in cm and g['id'] in rm for g in selected)
     new=sum(g['id'] in cm and g['id'] not in rm for g in selected)
     lost=sum(g['id'] not in cm and g['id'] in rm for g in selected)
     rows.append({'seed':seed,'comparison':'C - '+ref,'group':group,'category':'all' if cid is None else classes[cid],
       'GT':len(selected),'IoU':t,'score_threshold':.3,'common':common,'new':new,'lost':lost,
       'neither':len(selected)-common-new-lost,'net_new_minus_lost':new-lost,
       'delta_detection_rate_pp':100*(new-lost)/len(selected) if selected else None})
with (OUT/'testdev_类别尺度_共同新增丢失.csv').open('x',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
p=PAPER/'修订_20261018_v5/supplementary/full12_original_precision.csv';source_files[str(p)]=sha(p)
full=list(csv.DictReader(p.open(encoding='utf-8-sig')));metrics=[]
for seed in ['0','1','2']:
 by={r['arm']:r for r in full if r['seed']==seed and r['split']=='testdev'}
 for ref in ['C-mean']+(['baseline'] if seed!='2' else []):
  metrics.append({'seed':int(seed),'comparison':'C - '+ref,**{m:float(by['C'][m+'_percent'])-float(by[ref][m+'_percent']) for m in ['AP','AP50','AP75','APs','APm','APl','ARs','ARl']}})
write={'status':'existing_outputs_read_only_diagnostic','new_training':False,'new_inference':False,
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'testdev_GT_counts':{g:len(v) for g,v in groups.items()},
 'paired_v5_metric_deltas_pp':metrics,'source_sha256':source_files,
 'limits':['Cached matches use score0.3, same-class one-to-one descending IoU; this is detection-state diagnosis, not COCO AP contribution decomposition.',
 'Original COCO large and input640 scales are different, overlapping definitions; report separately.',
 'Classes/scales are exploratory subdivisions; no significance or gating-method effectiveness claim.',
 'Seed2 has no corresponding baseline in the current manuscript evidence.']}
(OUT/'来源与初筛摘要.json').write_text(json.dumps(write,ensure_ascii=False,indent=2),encoding='utf8')
selected=[r for r in rows if r['group']=='original_COCO_large_ge96' and r['IoU']=='0.75']
print(json.dumps({'saved':str(OUT),'GT_counts':write['testdev_GT_counts'],'all_class_summary':[r for r in selected if r['category']=='all'],
 'largest_net_losses':sorted([r for r in selected if r['category']!='all'],key=lambda r:r['net_new_minus_lost'])[:8]},ensure_ascii=False,indent=2))
