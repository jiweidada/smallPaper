"""Fixed-cache B/C by cap diagnostic; no model or head construction/execution."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import sys,json,csv,copy,hashlib,time,io,contextlib,importlib.util,traceback
from pathlib import Path
from collections import defaultdict
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
torch.set_num_threads(2)
HERE=Path(__file__).resolve().parent;OUT=HERE/'attempt01';ROOT=HERE.parent.parent
PAIR=ROOT/'raw_EMA逐GT对照_20261026_v1';REP=PAIR/'同后端CPU重放_20261026_v1'
TRANSFER=PAIR/'rawBC操作地图与EMA迁移_20261026_v1'
VAL=Path(r'C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val/annotations/val.json')
spec=importlib.util.spec_from_file_location('prior_audit',ROOT/'问题优先诊断_20261026_v1/audit.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A);torch.set_num_threads(2)
MET=A.METRICS
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(n,j):
 with open(OUT/n,'x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvout(n,rr):
 with open(OUT/n,'x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def metrics(ev,cap):
 p=ev.params;mi=p.maxDets.index(cap);P=ev.eval['precision'];R=ev.eval['recall']
 def avg(z):
  z=z[z>-1];return float(z.mean()) if z.size else -1.
 def ap(area=0,t=None):
  q=P[:,:,:,area,mi];return avg(q if t is None else q[t])
 def ar(area=0,m=mi):return avg(R[:,:,area,m])
 values=[ap(),ap(t=0),ap(t=5),ap(1),ap(2),ap(3),ar(m=p.maxDets.index(1)),ar(m=p.maxDets.index(10)),ar(),ar(1),ar(2),ar(3)]
 return dict(zip(MET,values))
def main():
 start=time.monotonic();assert not OUT.exists();OUT.mkdir();ph=A.sha(HERE/'protocol.json');assert ph==(HERE/'protocol.sha256').read_text('utf-8').strip()
 inputs={};calls=0;phase='inputs';parity=[];result={};hits={};gtrows=[];group_rows=[]
 def track(p):
  p=Path(p);h=A.sha(p);inputs[str(p)]=h;return h
 def budget():
  if time.monotonic()-start>900:raise TimeoutError('Frozen900-second budget')
  if sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())>200*1024**2:raise RuntimeError('Frozen output budget')
 try:
  for p in (HERE/'protocol.json',Path(__file__),Path(A.__file__),VAL,REP/'complete.json',REP/'preflight.json',TRANSFER/'raw_operations_v1/results.json',TRANSFER/'EMA_transfer_v1/results.json',PAIR/'后续零训练预检_20261026_v1/B_per_GT.csv',TRANSFER/'EMA_transfer_v1/per_GT.csv'):track(p)
  for n in ('coco.py','cocoeval.py','mask.py','_mask.cp310-win_amd64.pyd'):track(A.EVALROOT/'pycocoeval'/n)
  prior={'raw':read(TRANSFER/'raw_operations_v1/results.json'),'EMA':read(TRANSFER/'EMA_transfer_v1/results.json')}
  completed=read(REP/'complete.json');pref=read(REP/'preflight.json')
  declared={r['path']:r['sha256'] for r in pref['entries']+completed['entries']};assert len(declared)==2192
  ds=read(VAL);ims=sorted(ds['images'],key=lambda x:x['id']);anns=sorted(ds['annotations'],key=lambda x:x['id']);ids=[a['id'] for a in anns];ai={a:i for i,a in enumerate(ids)}
  assert len(ims)==548 and len(anns)==38759 and len(ai)==38759
  assert all(a['iscrowd']==0 and a['bbox'][2]>0 and a['bbox'][3]>0 for a in anns)
  pos=copy.deepcopy(ds)
  for a in pos['annotations']:a['id']+=1
  oldhits={}
  for state,path in [('raw',PAIR/'后续零训练预检_20261026_v1/B_per_GT.csv'),('EMA',TRANSFER/'EMA_transfer_v1/per_GT.csv')]:
   with open(path,encoding='utf-8-sig') as f:rr=list(csv.DictReader(f))
   assert len(rr)==77518
   for seed in (1,2):
    selected=[r for r in rr if int(r['seed'])==seed];assert len(selected)==38759
    for arm in ('B','C'):
     for th in (50,75):oldhits[state,seed,arm,th]={int(r['annotation_id']) for r in selected if r[f'{arm}{th}']=='1'}
  for state in ('raw','EMA'):
   result[state]={}
   for seed in (1,2):
    predictions={'B':[],'C':[]};phase=f'{state} seed{seed} reconstruct'
    for im in ims:
     budget();p=REP/f'seed{seed}_{state}/image_{im["id"]:06d}.npz';assert track(p)==declared[str(p)]
     with np.load(p,allow_pickle=False) as z:
      assert int(z['seed'])==seed and str(z['branch'])==state and int(z['image_id'])==im['id']
      pb=torch.from_numpy(z['final_boxes']);lg=torch.from_numpy(z['final_logits'])
     assert pb.shape==(300,4) and lg.shape==(300,10) and torch.isfinite(pb).all() and torch.isfinite(lg).all()
     s=lg.sigmoid();av,af=s.flatten().topk(300);aq=af[av>.001]//10;U=torch.unique(aq,sorted=True)
     positive=(pb[:,2:]>0).all(-1);offered=int(((s[U]>.001)&positive[U,None]).sum());assert offered>=600
     bv,bf=s[U].flatten().topk(600);cv,cf=s.flatten().topk(600)
     boxes=A.edges_norm(pb)*torch.tensor([im['width'],im['height']]*2)
     for arm,v,q,c in [('B',bv,U[bf//10],bf%10),('C',cv,cf//10,cf%10)]:
      assert (v>.001).all() and positive[q].all()
      for qi,ci,score in zip(q.tolist(),c.tolist(),v.tolist()):
       b=boxes[qi];xywh=torch.cat((b[:2],b[2:]-b[:2])).tolist();predictions[arm].append({'image_id':im['id'],'category_id':ci,'bbox':xywh,'score':score})
    result[state][str(seed)]={'arms':{}}
    for arm in ('B','C'):
     budget();phase=f'{state} seed{seed} {arm} evaluate';log=io.StringIO()
     with contextlib.redirect_stdout(log):
      gt=A.COCO();gt.dataset=copy.deepcopy(pos);gt.createIndex();dt=gt.loadRes(copy.deepcopy(predictions[arm]))
      ev=A.COCOeval(gt,dt,'bbox',using_tiny_metrice=False,using_vtiny_metrice=False,max_dets=600)
      ev.params.imgIds=sorted(gt.getImgIds());ev.params.maxDets=[1,10,100,600]
      ev.evaluate();ev.accumulate();ev.summarize()
     calls+=1
     with open(OUT/f'{state}_s{seed}_{arm}_eval.txt','x',encoding='utf-8') as f:f.write(log.getvalue())
     mm={cap:metrics(ev,cap) for cap in (100,600)}
     assert max(abs(mm[100][k]-float(v)) for k,v in zip(MET,ev.stats))<1e-12
     oldarm=prior[state]['seeds'][str(seed)][arm] if state=='raw' else prior[state]['seeds'][str(seed)]['arms'][arm]
     error=max(abs(mm[100][k]-oldarm['metrics'][k]) for k in MET);assert error<1e-12,(phase,error)
     hh={cap:np.zeros((len(anns),10),dtype=bool) for cap in (100,600)};evaluated={100:0,600:0};seen=[]
     for e in ev.evalImgs:
      if e is None or tuple(e['aRng'])!=tuple(ev.params.areaRng[0]):continue
      assert not np.asarray(e['gtIgnore']).any();seen.extend(int(x)-1 for x in e['gtIds'])
      for cap in (100,600):
       n=min(cap,len(e['dtIds']));evaluated[cap]+=n
       for t in range(10):
        for j in range(n):
         aid=int(e['dtMatches'][t,j])
         if aid>0 and not e['dtIgnore'][t,j]:
          original=aid-1;assert original in ai;hh[cap][ai[original],t]=True
     assert len(seen)==len(ids) and set(seen)==set(ids)
     assert hh[100].sum(0).tolist()==oldarm['all10_counts']
     assert evaluated[100]==oldarm['evaluated'] and evaluated[600]==len(predictions[arm])==328800
     assert not (hh[100]&~hh[600]).any()
     for th,t in ((50,0),(75,5)):assert {ids[j] for j in np.flatnonzero(hh[100][:,t])}==oldhits[state,seed,arm,th]
     parity.append({'state':state,'seed':seed,'arm':arm,'max12_metric_error':error,'all10_counts_exact':True,'TP50_75_ID_sets_exact':True,'cap100_evaluated_count_exact':True})
     hits[state,seed,arm]=hh
     with open(OUT/f'{state}_s{seed}_{arm}_GT_hits.npz','xb') as f:np.savez_compressed(f,annotation_id=np.asarray(ids),hit100=hh[100],hit600=hh[600])
     result[state][str(seed)]['arms'][arm]={str(cap):{'metrics':mm[cap],'TP_all10':hh[cap].sum(0).tolist(),'evaluated':evaluated[cap],'FP50':evaluated[cap]-int(hh[cap][:,0].sum())} for cap in (100,600)}
     print(f'{phase} PASS cap100 parity; AP100={mm[100]["AP"]*100:.6f}, AP600={mm[600]["AP"]*100:.6f}; elapsed={time.monotonic()-start:.1f}s',flush=True)
     del ev,gt,dt
    for cap in (100,600):
     mb=result[state][str(seed)]['arms']['B'][str(cap)]['metrics'];mc=result[state][str(seed)]['arms']['C'][str(cap)]['metrics'];hb=hits[state,seed,'B'][cap];hc=hits[state,seed,'C'][cap]
     d={k:100*(mc[k]-mb[k]) for k in MET};new={};lost={}
     for th,t in ((50,0),(75,5)):
      new[str(th)]=[ids[j] for j in np.flatnonzero(~hb[:,t]&hc[:,t])];lost[str(th)]=[ids[j] for j in np.flatnonzero(hb[:,t]&~hc[:,t])]
     result[state][str(seed)][f'C_minus_B_cap{cap}']={'delta_pp':d,'new_IDs':new,'lost_IDs':lost,
      'joint_gate':d['AP']>1e-4 and d['APs']>1e-4 and int(hc[:,0].sum())>int(hb[:,0].sum())}
    for j,a in enumerate(anns):
     r={'state':state,'seed':seed,'image_id':a['image_id'],'annotation_id':a['id'],'category_id':a['category_id'],'area_original':a['area']}
     for arm in ('B','C'):
      for cap in (100,600):
       for th,t in ((50,0),(75,5)):r[f'{arm}_cap{cap}_hit{th}']=int(hits[state,seed,arm][cap][j,t])
     gtrows.append(r)
    write(f'{state}_s{seed}_done.json',result[state][str(seed)])
  for state in ('raw','EMA'):
   for seed in (1,2):
    rr=[r for r in gtrows if r['state']==state and r['seed']==seed]
    for typ,labels in [('class',list(range(10))),('area',['small','medium','large'])]:
     for label in labels:
      sub=[r for r in rr if (r['category_id']==label if typ=='class' else ('small' if r['area_original']<1024 else 'medium' if r['area_original']<9216 else 'large')==label)]
      group_rows.append({'state':state,'seed':seed,'group_type':typ,'group':label,'GT':len(sub),**{f'{arm}_cap{cap}_TP{th}':sum(r[f'{arm}_cap{cap}_hit{th}'] for r in sub) for arm in ('B','C') for cap in (100,600) for th in (50,75)}})
  csvout('per_GT.csv',gtrows);csvout('by_class_area.csv',group_rows)
  assert calls==8;budget();unchanged=all(A.sha(p)==h for p,h in inputs.items());assert unchanged
  write('results.json',{'status':'COMPLETE','protocol_sha256':ph,'conditions':result,
   'cap100_necessity_refuted_for_measured_joint_benefit':all(result[state][str(seed)]['C_minus_B_cap600']['joint_gate'] for state in ('raw','EMA') for seed in (1,2)),
   'meaning':'Within-cap readout-rule contrasts. Cap600 is diagnostic, not native benchmark AP; no additive attribution or training claim.',
   'operations':{'evaluator_calls':calls,'new_model_forwards':0,'head_calls':0,'checkpoint_loads':0,'model_builds':0,'training':0},'elapsed_seconds':time.monotonic()-start})
  write('parity.json',parity);write('input_manifest.json',inputs)
  write('verification.json',{'status':'PASS','cap100_parities':len(parity),'unchanged_inputs':len(inputs),'all_replay_NPZ_SHA_exact':True,'protocol_unchanged':A.sha(HERE/'protocol.json')==ph,'GT_rows':len(gtrows),'operations':{'evaluator_calls':calls,'forwards':0,'head_calls':0,'training':0}})
  print('COMPLETE',time.monotonic()-start,flush=True)
 except Exception as e:
  write('failure.json',{'status':'FAILED_PRESERVED','phase':phase,'error':repr(e),'traceback':traceback.format_exc(),'calls':calls,'completed_parity':parity,'elapsed_seconds':time.monotonic()-start,'protocol_sha256':ph,'model_forwards':0,'training':0})
  raise
if __name__=='__main__':main()
