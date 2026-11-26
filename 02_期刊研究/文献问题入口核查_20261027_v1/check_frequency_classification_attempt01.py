"""All-GT fixed prediction diagnostic. No torch, model, RGB or new evaluation."""
import json,pathlib,hashlib,collections,time,traceback,csv
import numpy as np
BASE=pathlib.Path(__file__).resolve().parent;ROOT=BASE.parent
E=ROOT/'data/00_实验计划与证据索引'
CACHE=E/'正式132轮基线整图双问题诊断_20261008_v1/native_transform_v2'
PREV=ROOT/'具体问题重新筛选_20261027_v1'
DATA=pathlib.Path('C:/Users/jiwei/Desktop/DEIM/dataset_visdrone')
TRAIN=DATA/'VisDrone2019-DET-train/annotations/train.json'
VAL=DATA/'VisDrone2019-DET-val/annotations/val.json'
PROTO=BASE/'04_类别频次与几何条件错类_冻结协议.json'
OUT=BASE/'frequency_classification_attempt01'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def iou(a,b):
 wh=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]))
 it=wh.prod(-1);return it/np.maximum((a[:,2:]-a[:,:2]).prod(-1)[:,None]+(b[:,2:]-b[:,:2]).prod(-1)[None,:]-it,1e-300)
def summarize(rows,threshold,lo,hi):
 out={'threshold':threshold,'groups':{},'class_rows':{},'confusion':np.zeros((10,10),int).tolist()};matrix=np.zeros((10,10),int)
 for ids,name in [(lo,'low3'),(hi,'high3'),(list(range(10)),'all10')]:
  rr=[r for r in rows if r['category_id'] in ids];gg=[r for r in rr if r['owner_iou']>=threshold]
  wrong=[r for r in gg if r['owner_class']!=r['category_id']]
  out['groups'][name]={'class_ids':ids,'all_GT':len(rr),'actual_TP50':sum(r['actual_TP50'] for r in rr),'actual_TP75':sum(r['actual_TP75'] for r in rr),'geometry_available':len(gg),'geometry_unavailable':len(rr)-len(gg),'owner_wrong':len(wrong),'owner_wrong_rate':len(wrong)/len(gg) if gg else None,'owner_wrong_but_actual_TP':sum(r['actual_TP50' if threshold==.5 else 'actual_TP75'] for r in wrong),'owner_wrong_actual_miss':sum(not r['actual_TP50' if threshold==.5 else 'actual_TP75'] for r in wrong),'owner_correct_actual_miss':sum(r['owner_class']==r['category_id'] and not r['actual_TP50' if threshold==.5 else 'actual_TP75'] for r in gg),'wrong_to_more_frequent':sum(r['pred_train_count']>r['true_train_count'] for r in wrong),'wrong_to_less_frequent':sum(r['pred_train_count']<r['true_train_count'] for r in wrong)}
 for c in range(10):
  gg=[r for r in rows if r['category_id']==c and r['owner_iou']>=threshold]
  out['class_rows'][str(c)]={'all_GT':sum(r['category_id']==c for r in rows),'geometry_available':len(gg),'wrong':sum(r['owner_class']!=c for r in gg),'wrong_to_more_frequent':sum(r['owner_class']!=c and r['pred_train_count']>r['true_train_count'] for r in gg)}
  for r in gg:matrix[c,r['owner_class']]+=1
 out['confusion']=matrix.tolist()
 strata=collections.defaultdict(lambda:[[],[]])
 for r in rows:
  if r['owner_iou']<threshold:continue
  group=0 if r['category_id'] in lo else (1 if r['category_id'] in hi else -1)
  if group<0:continue
  key=(r['area_bin'],r['ratio_bin'],r['neighbor_bin']);strata[key][group].append(r['owner_class']!=r['category_id'])
 common={k:v for k,v in strata.items() if len(v[0]) and len(v[1])};total=sum(len(a)+len(b) for a,b in common.values())
 low=sum((len(a)+len(b))*np.mean(a) for a,b in common.values())/total if total else None
 high=sum((len(a)+len(b))*np.mean(b) for a,b in common.values())/total if total else None
 out['common_support']={'cells':len(common),'low_n':sum(len(a) for a,b in common.values()),'high_n':sum(len(b) for a,b in common.values()),'low_wrong_standardized':float(low) if low is not None else None,'high_wrong_standardized':float(high) if high is not None else None,'delta_low_minus_high':float(low-high) if total else None,'cell_rows':[{'key':list(k),'low_n':len(a),'low_wrong':sum(a),'high_n':len(b),'high_wrong':sum(b),'pooled_weight':(len(a)+len(b))/total} for k,(a,b) in sorted(common.items())]}
 l=out['groups']['low3']['owner_wrong_rate'];h=out['groups']['high3']['owner_wrong_rate']
 out['raw_delta_low_minus_high']=l-h if l is not None and h is not None else None
 out['direction_gate']=bool(l is not None and h is not None and total and l>h and low>high)
 return out
def main():
 start=time.monotonic();OUT.mkdir(exist_ok=False)
 inputs=[PROTO,TRAIN,VAL,pathlib.Path(__file__),PREV/'delivery_manifest.json'];bindings={str(p):sha(p) for p in inputs}
 sealed={r['path']:r['sha256'] for r in read(PREV/'delivery_manifest.json')['inputs']+read(PREV/'delivery_manifest.json')['outputs']}
 assert sha(TRAIN)=='32c1cae852e084430ea74503eb660f0707ec6b5e65d0998ce3ec8b448265c883'
 train=read(TRAIN);v=read(VAL);images={i['id']:i for i in v['images']};trainims={i['id']:i for i in train['images']};freq=collections.Counter()
 for a in train['annotations']:
  if a.get('iscrowd',0):continue
  im=trainims[a['image_id']];x,y,w,h=a['bbox'];x1=np.clip(x,0,im['width']);y1=np.clip(y,0,im['height']);x2=np.clip(x+w,0,im['width']);y2=np.clip(y+h,0,im['height'])
  if x2>x1 and y2>y1:freq[a['category_id']]+=1
 ordered=sorted(range(10),key=lambda c:(freq[c],c));lo=ordered[:3];hi=sorted(range(10),key=lambda c:(-freq[c],c))[:3]
 definitions={'train_valid_instances':dict(freq),'low3':lo,'high3':hi,'categories':v['categories'],'definitions_from_predictions':False,'inputs':bindings}
 save(OUT/'group_definitions_before_prediction_reads.json',definitions)
 annby=collections.defaultdict(list)
 for a in v['annotations']:annby[a['image_id']].append(a)
 result={'GT':len(v['annotations']),'images':len(v['images']),'group_definitions':definitions,'seeds':{},'inputs':bindings,'new_model_forwards':0,'new_CPU_evaluator_calls':0,'training':0}
 for seed in [1,2]:
  cp=CACHE/f'seed{seed}';mp=cp/'array_manifest.json';bindings[str(mp)]=sha(mp);manifest={r['file']:r['sha256'] for r in read(mp)}
  matchp=PREV/f'06_seed{seed}_fixed_rank_arrays.npz';rankp=PREV/f'06_seed{seed}_rank_results.json';bindings[str(matchp)]=sha(matchp);bindings[str(rankp)]=sha(rankp)
  assert bindings[str(matchp)]==sealed[str(matchp)] and bindings[str(rankp)]==sealed[str(rankp)]
  hits={50:set(),75:set()}
  with np.load(matchp) as z:
   for c in range(10):
    m=z[f'a0_c{c}_matches'];ig=z[f'a0_c{c}_ignored']
    for th,ti in [(50,0),(75,5)]:hits[th].update(int(x)-1 for x in m[ti][(m[ti]>0)&~ig[ti]])
  assert [len(hits[t]) for t in [50,75]]==([29682,18702] if seed==1 else [29745,18889])
  rows=[]
  for iid,im in sorted(images.items()):
   p=cp/'arrays'/f'{iid:06d}.npz';h=sha(p);assert h==manifest[p.name];bindings[str(p)]=h
   with np.load(p) as z:box=z['layer_boxes'][-1].astype(float);logits=z['logits'].astype(float);assert int(z['image_id'])==iid and list(z['image_wh'])==[im['width'],im['height']]
   assert box.shape==(300,4) and logits.shape==(300,10) and np.isfinite(box).all() and np.isfinite(logits).all()
   g=annby[iid];gt=np.array([a['bbox'] for a in g],float).reshape(-1,4);gt[:,2:]+=gt[:,:2]
   pred=np.concatenate([box[:,:2]-box[:,2:]/2,box[:,:2]+box[:,2:]/2],1)*np.array([im['width'],im['height'],im['width'],im['height']])
   ov=iou(gt,pred);owner=ov.argmax(1);best=ov[np.arange(len(g)),owner];neigh=iou(gt,gt);np.fill_diagonal(neigh,0);neigh=neigh.max(1) if len(g) else np.empty(0)
   for j,a in enumerate(g):
    q=int(owner[j]);c=a['category_id'];pr=int(logits[q].argmax());w=a['bbox'][2]*640/im['width'];hh=a['bbox'][3]*640/im['height'];area=np.sqrt(w*hh);ratio=w/hh
    rows.append({'image_id':iid,'annotation_id':a['id'],'category_id':c,'true_train_count':freq[c],'owner_query':q,'owner_iou':float(best[j]),'owner_class':pr,'pred_train_count':freq[pr],'true_minus_pred_logit':float(logits[q,c]-logits[q,pr]),'actual_TP50':a['id'] in hits[50],'actual_TP75':a['id'] in hits[75],'sqrt_area640':float(area),'ratio640':float(ratio),'neighbor_max_iou':float(neigh[j]),'area_bin':int(np.digitize(area,[4,8,16,32,64,128])),'ratio_bin':int(np.digitize(ratio,[.5,1,2])),'neighbor_bin':int(np.digitize(neigh[j],[.1,.3,.5]))})
  assert len(rows)==38759
  summaries={str(t):summarize(rows,t,lo,hi) for t in [.5,.75]};rank=read(rankp)
  summaries['native_class_AP']=rank['per_class_native_AP'];summaries['native_AP']=rank['native_positiveID'];summaries['actual_TP']={str(t):len(hits[t]) for t in [50,75]}
  save(OUT/f'seed{seed}_rows.json',rows);save(OUT/f'seed{seed}_results.json',summaries);result['seeds'][str(seed)]=summaries
 for p,h in bindings.items():assert sha(p)==h,('input_changed',p)
 result['both_direction_gates']=all(s['0.5']['direction_gate'] for s in result['seeds'].values());result['seconds']=time.monotonic()-start
 save(OUT/'results.json',result);save(OUT/'sha256.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
 print(json.dumps({'groups':definitions['train_valid_instances'],'low3':lo,'high3':hi,'seeds':{s:{'groups':r['0.5']['groups'],'standardized':{k:v for k,v in r['0.5']['common_support'].items() if k!='cell_rows'},'direction_gate':r['0.5']['direction_gate']} for s,r in result['seeds'].items()},'both_gate':result['both_direction_gates'],'seconds':result['seconds']}))
if __name__=='__main__':
 try:main()
 except Exception as e:
  if OUT.exists():save(OUT/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'scientific_result':'NA'})
  raise
