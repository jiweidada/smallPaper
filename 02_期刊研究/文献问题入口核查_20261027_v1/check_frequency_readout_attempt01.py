"""Frozen full native readout and diagnostic identity comparison. No model."""
import json,pathlib,hashlib,collections,time,traceback
import numpy as np
BASE=pathlib.Path(__file__).resolve().parent;ROOT=BASE.parent
E=ROOT/'data/00_实验计划与证据索引';CACHE=E/'正式132轮基线整图双问题诊断_20261008_v1/native_transform_v2'
PREV=ROOT/'具体问题重新筛选_20261027_v1';FREQ=BASE/'frequency_classification_attempt02'
PROTO=BASE/'05_频次线索_完整读出与身份竞争_追加协议.json';OUT=BASE/'frequency_readout_attempt01'
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(p,v):pathlib.Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def iou(a,b):
 wh=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]));it=wh.prod(-1)
 return it/np.maximum((a[:,2:]-a[:,:2]).prod(-1)[:,None]+(b[:,2:]-b[:,:2]).prod(-1)[None,:]-it,1e-300)
def summarize(rows,ids):
 r=[x for x in rows if x['category_id'] in ids];part=collections.Counter(x['stage'] for x in r);geometry=[x for x in r if x['has_geometry']];private=[x for x in r if x['has_private_geometry']]
 return {'class_ids':ids,'all_GT':len(r),'partition':dict(part),'geometry_available':len(geometry),'geometry_all_argmax_wrong':sum(not x['any_cover_top1_true'] for x in geometry),'geometry_all_argmax_wrong_actual_miss':sum(not x['any_cover_top1_true'] and not x['actual_TP'] for x in geometry),'private_geometry_available':len(private),'private_all_argmax_wrong':sum(not x['any_private_cover_top1_true'] for x in private),'private_all_argmax_wrong_actual_miss':sum(not x['any_private_cover_top1_true'] and not x['actual_TP'] for x in private),'owner_wrong':sum(x['owner_wrong'] for x in geometry),'owner_wrong_more_IoU_other_predclass':sum(x['owner_wrong'] and x['owner_closer_predclass_GT'] for x in geometry),'FP64_vs_native_geometry_cross':sum(x['geometry_threshold_differs_from04'] for x in r)}
def main():
 start=time.monotonic();OUT.mkdir(exist_ok=False);old=read(FREQ/'results.json');bindings={str(PROTO):sha(PROTO),str(pathlib.Path(__file__)):sha(__file__)}
 for p,h in old['inputs'].items():assert sha(p)==h
 valp=pathlib.Path('C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val/annotations/val.json');v=read(valp);images={i['id']:i for i in v['images']};anns=collections.defaultdict(list)
 for a in v['annotations']:anns[a['image_id']].append(a)
 result={'seeds':{},'inputs':bindings,'new_forwards':0,'new_CPU_evaluations':0,'training':0}
 for seed in [1,2]:
  cp=CACHE/f'seed{seed}';sourcesp=cp/'prediction_sources.json';predp=cp/'predictions.json';sources=read(sourcesp);preds=read(predp)
  bindings[str(sourcesp)]=sha(sourcesp);bindings[str(predp)]=sha(predp)
  seal={r['path']:r['sha256'] for r in read(PREV/'delivery_manifest.json')['inputs']+read(PREV/'delivery_manifest.json')['outputs']}
  assert bindings[str(sourcesp)]==seal[str(sourcesp)] and bindings[str(predp)]==seal[str(predp)]
  oldrows=read(FREQ/f'seed{seed}_rows.json');oldby={r['annotation_id']:r for r in oldrows};export=collections.defaultdict(dict)
  assert len(sources)==len(preds)==164400
  for s,d in zip(sources,preds):
   assert s['image_id']==d['image_id'] and s['class']==d['category_id'];export[s['image_id']][s['query'],s['class']]=s['row']
  hits={};caps={}
  with np.load(PREV/f'06_seed{seed}_fixed_rank_arrays.npz') as z:
   for th,ti in [(.5,0),(.75,5)]:
    h=set()
    for c in range(10):
     m=z[f'a0_c{c}_matches'][ti];ig=z[f'a0_c{c}_ignored'][ti];h.update(int(x)-1 for x in m[(m>0)&~ig])
    hits[th]=h
   caprows=set(np.concatenate([z[f'a0_c{c}_row'] for c in range(10)]).tolist())
  allrows={.5:[],.75:[]};parity=0
  for iid,im in sorted(images.items()):
   p=cp/'arrays'/f'{iid:06d}.npz';assert sha(p)==old['inputs'][str(p)]
   with np.load(p) as z:b=z['layer_boxes'][-1];logits=z['logits']
   xy=np.concatenate([b[:,:2]-.5*b[:,2:],b[:,:2]+.5*b[:,2:]],1).astype(np.float32)*np.array([im['width'],im['height'],im['width'],im['height']],np.float32)
   xywh=xy.copy();xywh[:,2:]-=xywh[:,:2]
   for (q,c),row in export[iid].items():assert np.array_equal(xywh[q],np.array(preds[row]['bbox'],np.float32));parity+=1
   boxes=xywh.astype(float);boxes[:,2:]+=boxes[:,:2]
   ga=sorted(anns[iid],key=lambda a:a['id']);g=np.array([a['bbox'] for a in ga],float);g[:,2:]+=g[:,:2];ov=iou(g,boxes);closest_gt=ov.argmax(0);predclass=logits.argmax(1)
   for j,a in enumerate(ga):
    c=a['category_id'];owner=int(ov[j].argmax());wrong=predclass[owner]!=c;other=[k for k,aa in enumerate(ga) if k!=j and aa['category_id']==int(predclass[owner])]
    closer=bool(other and max(ov[k,owner] for k in other)>ov[j,owner])
    for th in [.5,.75]:
     cover=np.flatnonzero(ov[j]>=th);private=np.array([q for q in cover if closest_gt[q]==j],int);passed=[int(q) for q in cover if float(logits[q,c])>np.log(.001/.999)]
     exported=[export[iid][q,c] for q in cover if (q,c) in export[iid]];capped=[r for r in exported if r in caprows];actual=a['id'] in hits[th]
     if actual:stage='actual_TP'
     elif not len(cover):stage='miss_no_geometry'
     elif not passed:stage='miss_trueclass_below_threshold'
     elif not exported:stage='miss_flatTop300'
     elif not capped:stage='miss_classCap100'
     else:stage='miss_native_matching_competition'
     assert not actual or len(cover),('actual_hit_no_canonical_cover',seed,iid,a['id'],th,float(ov[j].max()))
     allrows[th].append({'image_id':iid,'annotation_id':a['id'],'category_id':c,'stage':stage,'actual_TP':actual,'has_geometry':bool(len(cover)),'cover_queries':len(cover),'has_private_geometry':bool(len(private)),'private_cover_queries':len(private),'any_cover_top1_true':bool(np.any(predclass[cover]==c)),'any_private_cover_top1_true':bool(np.any(predclass[private]==c)),'owner_wrong':bool(wrong),'owner_closer_predclass_GT':closer,'geometry_threshold_differs_from04':bool((oldby[a['id']]['owner_iou']>=th)!=(len(cover)>0)),'native_trueclass_cover_exported':len(exported),'native_trueclass_cover_capped':len(capped)})
  per={'canonical_exported_box_exact_rows':parity,'thresholds':{}}
  for th,rows in allrows.items():
   assert len(rows)==38759
   per['thresholds'][str(th)]={name:summarize(rows,ids) for name,ids in [('low3',old['group_definitions']['low3']),('high3',old['group_definitions']['high3']),('all10',list(range(10)))]+[(f'class{c}',[c]) for c in range(10)]}
   save(OUT/f'seed{seed}_IoU{th}_rows.json',rows)
  result['seeds'][str(seed)]=per
 for p,h in bindings.items():assert sha(p)==h
 result['seconds']=time.monotonic()-start;save(OUT/'results.json',result);save(OUT/'sha256.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
 print(json.dumps({'seeds':{s:{k:r['thresholds']['0.5'][k] for k in ['low3','high3','all10']} for s,r in result['seeds'].items()},'seconds':result['seconds']}))
if __name__=='__main__':
 try:main()
 except Exception as e:
  if OUT.exists():save(OUT/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'scientific_result':'NA'})
  raise
