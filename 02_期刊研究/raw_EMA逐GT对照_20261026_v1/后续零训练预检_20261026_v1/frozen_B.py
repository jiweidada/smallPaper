"""Execute original B preflight; only valid equal-budget arms reach evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import sys,json,csv,time,copy,hashlib,importlib.util
from pathlib import Path
from collections import Counter,defaultdict
sys.dont_write_bytecode=True
import numpy as np
import torch
O=Path(__file__).resolve().parent;PAIR=O.parent;ROOT=PAIR.parent;PREV=ROOT/'问题优先诊断_20261026_v1'
spec=importlib.util.spec_from_file_location('prior_audit',PREV/'audit.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A);A.OUT=O;torch.set_num_threads(2)
S3=ROOT/'code/unit_instance_completion_20261021_v1/342_codex_stage3_20261022_v1'
VAL=Path(r'C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val/annotations/val.json')
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(n,j):
 with open(O/n,'x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvout(n,rr):
 with open(O/n,'x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def main():
 start=time.time();plan=PREV/'followup_plan.json';assert A.sha(plan)==(PREV/'followup_plan.sha256').read_text('utf-8').strip()=='c11bfc210fc0e556c10f76ac9996f980ecd59f8763e5126b50ccf532d6aaf311'
 ph=A.sha(O/'protocol.json');assert ph==(O/'protocol.sha256').read_text('utf-8').strip()
 inputs={};oldmanifest={r['path']:r['sha256'] for r in read(PREV/'input_manifest.json')['entries']}
 def track(p):
  p=Path(p);h=A.sha(p);inputs[str(p)]=h
  if str(p) in oldmanifest:assert h==oldmanifest[str(p)]
 for p in (plan,O/'protocol.json',Path(__file__),PREV/'results.json',PREV/'per_GT.csv',VAL,Path(A.__file__)):track(p)
 ds=read(VAL);ims=sorted(ds['images'],key=lambda x:x['id']);old=read(PREV/'results.json');oldrows=list(csv.DictReader(open(PREV/'per_GT.csv',encoding='utf-8-sig')))
 assert len(ims)==548 and len(ds['annotations'])==38759
 predictions={};gates=[];recipechecks={};bcandidates={}
 for seed in (1,2):
  pred=[];Acap=[];c_count=0;exportinvalidC=0;Bq_by_image={}
  for im in ims:
   if time.time()-start>900:raise TimeoutError('Frozen900s budget')
   iid=im['id'];p=S3/f'342_raw_val_seed{seed}/342_{iid:06d}.pt';track(p);r=torch.load(p,map_location='cpu',weights_only=True)
   pb=r['final300_boxes'];lg=r['final300_logits'];assert pb.shape==(300,4) and lg.shape==(300,10) and torch.isfinite(pb).all() and torch.isfinite(lg).all()
   scores=lg.sigmoid();av,af=scores.flatten().topk(300);af=af[av>.001];av=av[av>.001];aq=af//10;al=af%10;U=torch.unique(aq,sorted=True)
   positive=(pb[:,2:]>0).all(-1)&torch.isfinite(pb).all(-1)
   offered=int(((scores[U]>.001)&torch.isfinite(scores[U])&positive[U,None]).sum())
   gates.append({'seed':seed,'image_id':iid,'native_unique_queries':len(U),'offered_finite_positive_score_pairs':offered,'pass_600':offered>=600})
   bb=(A.edges_norm(pb)*torch.tensor([im['width'],im['height']]*2))
   cap=A.cap_indices(al.numpy())
   for ix in cap:
    q=int(aq[ix]);c=int(al[ix]);b=bb[q];xywh=torch.cat((b[:2],b[2:]-b[:2])).tolist()
    Acap.append({'image_id':iid,'category_id':c,'bbox':xywh,'score':float(av[ix])})
   cv,cf=scores.flatten().topk(600);cf=cf[cv>.001];c_count+=len(cf);exportinvalidC+=int((~positive[cf//10]).sum())
   if offered>=600:
    bv,bf=scores[U].flatten().topk(600);bq=U[bf//10];bl=bf%10
    assert (bv>.001).all() and positive[bq].all()
    for q,c,v in zip(bq.tolist(),bl.tolist(),bv.tolist()):
     b=bb[q];xywh=torch.cat((b[:2],b[2:]-b[:2])).tolist();pred.append({'image_id':iid,'category_id':c,'bbox':xywh,'score':v})
    Bq_by_image[iid]=len(A.cap_indices(bl.numpy()))
  oldp=ROOT/f'code/unit_instance_completion_20261021_v1/349_cpu_val_20261023_v1/349_seed{seed}_baseline_predictions.json';track(oldp);stored=read(oldp)
  assert len(Acap)==len(stored)
  for a,b in zip(Acap,stored):assert all(a[k]==b[k] for k in ('image_id','category_id','bbox','score'))
  assert len(Acap)==old['precheck'][str(seed)]['evaluated_rows_native']
  assert c_count==old['precheck'][str(seed)]['exported_rows_K600']
  recipechecks[str(seed)]={'A_archived_capped_predictions_exact':len(Acap),'C_recomputed_export_count':c_count,'C_invalid_exported_boxes':exportinvalidC,
   'C_metrics_and_hitsets':'Reused original verified frozen results, no new C evaluator call or unavailable-C-prediction-hash claim.'}
  predictions[seed]=pred;bcandidates[seed]=Bq_by_image
 bad=[r for r in gates if not r['pass_600']]
 csvout('B_preflight_per_image.csv',gates)
 write('B_preflight.json',{'status':'PASS' if not bad else 'NA_EQUAL_BUDGET_GATE_FAILED','protocol_sha256':ph,
  'original_plan_sha256':A.sha(plan),'images_per_seed':548,'invalid_images':bad,'failed_counts_per_seed':{str(s):sum(r['seed']==s for r in bad) for s in (1,2)},
  'min_offered_by_seed':{str(s):min(r['offered_finite_positive_score_pairs'] for r in gates if r['seed']==s) for s in (1,2)},
  'recipe_checks':recipechecks,'tie_policy':'B U sorted ascending query IDs, native torch FP32 topk600 on Ux10, no tie modification.',
  'model_forwards':0,'evaluator_calls_so_far':0,'elapsed_seconds':time.time()-start})
 if bad:
  assert all(A.sha(p)==h for p,h in inputs.items());write('B_input_manifest.json',inputs)
  write('B_results.json',{'status':'NA_EQUAL_BUDGET_GATE_FAILED','affected_images':len(bad),'no_padding_or_threshold_change':True,
   'new_forwards':0,'B_evaluator_calls':0,'training':0,'stops_only':'Exact frozen equal-budget B contrast; not G or innovation exploration.'})
  print('B N/A:',{s:sum(r['seed']==s for r in bad) for s in (1,2)},flush=True);return
 # The original joint gate has passed for every image; exactly two new B evaluations.
 positive=copy.deepcopy(ds)
 for a in positive['annotations']:a['id']+=1
 result={};perGT=[];grouped=[];imageTP={};classAP=[]
 for seed in (1,2):
  if time.time()-start>900:raise TimeoutError('Frozen900s budget')
  ev,met=A.evaluate(positive,predictions[seed],f'seed{seed}_B_positiveIDs');hit,dt,_=A.hits_from_evaluator(ev,offset=1)
  rr=[r for r in oldrows if int(r['seed'])==seed];a={th:{int(r['annotation_id']) for r in rr if r[f'hit{th}']=='1'} for th in (50,75)}
  c={th:{int(r['annotation_id']) for r in rr if r[f'K600_hit{th}']=='1'} for th in (50,75)}
  ma=old['metrics'][str(seed)]['positive_id_native300'];mc=old['metrics'][str(seed)]['positive_id_top600']
  kept=set()
  for e in ev.evalImgs:
   if e is not None and tuple(e['aRng'])==tuple(ev.params.areaRng[0]):kept.update(e['dtIds'])
  result[str(seed)]={'metrics_B':met,'B_minus_A_pp':{k:100*(met[k]-ma[k]) for k in A.METRICS},'C_minus_B_pp':{k:100*(mc[k]-met[k]) for k in A.METRICS},
   'exported_B':len(predictions[seed]),'evaluated_B':len(kept),'FP50_B':len(kept)-len(hit[50]),'TP50_B':len(hit[50]),'TP75_B':len(hit[75]),'sets':{}}
  for th in (50,75):
   result[str(seed)]['sets'][str(th)]={'B_new_vs_A':sorted(hit[th]-a[th]),'B_lost_vs_A':sorted(a[th]-hit[th]),
    'C_new_vs_B':sorted(c[th]-hit[th]),'C_lost_vs_B':sorted(hit[th]-c[th]),'B_new_intersect_C_new_vs_A':len((hit[th]-a[th])&(c[th]-a[th]))}
  for r in rr:
   aid=int(r['annotation_id']);row={k:r[k] for k in ('seed','image_id','annotation_id','category_id','area_original','short_side_640')}
   for th in (50,75):row.update({f'A{th}':int(aid in a[th]),f'B{th}':int(aid in hit[th]),f'C{th}':int(aid in c[th])})
   perGT.append(row)
  for typ,labels in [('class',list(range(10))),('area',['small','medium','large'])]:
   for label in labels:
    sub=[r for r in perGT if int(r['seed'])==seed and (int(r['category_id'])==label if typ=='class' else ('small' if float(r['area_original'])<1024 else 'medium' if float(r['area_original'])<9216 else 'large')==label)]
    grouped.append({'seed':seed,'type':typ,'group':label,'GT':len(sub),**{f'{br}{th}':sum(r[f'{br}{th}'] for r in sub) for br in 'ABC' for th in (50,75)}})
  for catindex,catid in enumerate(ev.params.catIds):
   p=ev.eval['precision'][:,:,catindex,0,2];valid=p[p>-1];classAP.append({'seed':seed,'category_id':int(catid),'B_AP':float(valid.mean()) if valid.size else None})
  imageTP[seed]=np.asarray([[sum(r[k] for r in perGT if int(r['seed'])==seed and int(r['image_id'])==im['id']) for k in ('A50','B50','C50','A75','B75','C75')] for im in ims])
  print('B evaluated seed',seed,'AP',met['AP'],'TP50',len(hit[50]),flush=True)
 rng=np.random.default_rng(20261026);boot=rng.integers(0,548,size=(1000,548));ngt=np.asarray([sum(a['image_id']==im['id'] for a in ds['annotations']) for im in ims]);bootden=ngt[boot].sum(1)
 for seed in (1,2):
  q=imageTP[seed];intervals={}
  for name,j,k in [('B_minus_A_50',1,0),('C_minus_B_50',2,1),('B_minus_A_75',4,3),('C_minus_B_75',5,4)]:
   vals=(q[:,j]-q[:,k])[boot].sum(1)/bootden;intervals[name]={'point':float((q[:,j]-q[:,k]).sum()/38759),'image_paired_percentile95':np.quantile(vals,[.025,.975]).tolist()}
  result[str(seed)]['TP_rate_bootstrap']=intervals
  result[str(seed)]['class_hypothesis_joint_gate']=all(result[str(seed)]['B_minus_A_pp'][k]>1e-4 for k in ('AP','APs')) and result[str(seed)]['TP50_B']>old['precheck'][str(seed)]['TP50_native']
  result[str(seed)]['absent_query_joint_gate']=all(result[str(seed)]['C_minus_B_pp'][k]>1e-4 for k in ('AP','APs')) and old['precheck'][str(seed)]['TP50_K600']>result[str(seed)]['TP50_B']
 csvout('B_per_GT.csv',perGT);csvout('B_by_class_area.csv',grouped);csvout('B_class_AP.csv',classAP)
 write('B_results.json',{'status':'COMPLETE','protocol_sha256':ph,'seeds':result,'joint_class_hypothesis_signal':all(result[str(s)]['class_hypothesis_joint_gate'] for s in (1,2)),
  'joint_absent_query_signal':all(result[str(s)]['absent_query_joint_gate'] for s in (1,2)),'bootstrap_shared_indices_sha256':hashlib.sha256(boot.tobytes()).hexdigest(),
  'new_model_forwards':0,'B_evaluator_calls':2,'training':0,'elapsed_seconds':time.time()-start})
 assert all(A.sha(p)==h for p,h in inputs.items());write('B_input_manifest.json',inputs);write('B_verification.json',{'status':'PASS','unchanged_inputs':len(inputs),'protocol_unchanged':A.sha(O/'protocol.json')==ph,'rows':len(perGT)})
if __name__=='__main__':main()
