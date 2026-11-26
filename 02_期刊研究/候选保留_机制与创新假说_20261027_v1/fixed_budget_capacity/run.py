"""GT-oracle graph capacity at fixed300 outputs; not an inference method."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import sys,json,csv,time,hashlib,importlib.util,traceback
from pathlib import Path
from collections import defaultdict
sys.dont_write_bytecode=True
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;O=HERE/'attempt01';ROOT=HERE.parent.parent;PAIR=ROOT/'raw_EMA逐GT对照_20261026_v1';REP=PAIR/'同后端CPU重放_20261026_v1'
VAL=Path(r'C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val/annotations/val.json')
spec=importlib.util.spec_from_file_location('prior_audit',ROOT/'问题优先诊断_20261026_v1/audit.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A);torch.set_num_threads(2)
from pycocoeval import mask as maskUtils
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(n,j):
 with open(O/n,'x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvout(n,rr):
 with open(O/n,'x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def main():
 start=time.monotonic();assert not O.exists();O.mkdir();ph=A.sha(HERE/'protocol.json');assert ph==(HERE/'protocol.sha256').read_text('utf-8').strip();inputs={};phase='inputs'
 def track(p):inputs[str(p)]=A.sha(p)
 try:
  for p in (HERE/'protocol.json',Path(__file__),Path(A.__file__),VAL,REP/'complete.json',REP/'preflight.json'):track(p)
  for n in ('mask.py','_mask.cp310-win_amd64.pyd'):track(A.EVALROOT/'pycocoeval'/n)
  cp=read(REP/'complete.json');pref=read(REP/'preflight.json');declared={r['path']:r['sha256'] for r in cp['entries']+pref['entries']};assert len(declared)==2192
  ds=read(VAL);ims=sorted(ds['images'],key=lambda x:x['id']);by=defaultdict(list)
  for a in ds['annotations']:by[a['image_id']].append(a)
  assert len(ims)==548 and len(ds['annotations'])==38759
  actual={}
  for state,path in [('raw',PAIR/'后续零训练预检_20261026_v1/B_per_GT.csv'),('EMA',PAIR/'rawBC操作地图与EMA迁移_20261026_v1/EMA_transfer_v1/per_GT.csv')]:
   track(path)
   with open(path,encoding='utf-8-sig') as f:rr=list(csv.DictReader(f))
   assert len(rr)==77518
   for r in rr:
    for th in (50,75):actual[state,int(r['seed']),int(r['image_id']),th]=actual.get((state,int(r['seed']),int(r['image_id']),th),0)+int(r[f'A{th}'])
  rows=[];classes=[];summary={};invalid=[]
  for state in ('raw','EMA'):
   summary[state]={}
   for seed in (1,2):
    totals={th:{'GT':0,'A_actual':0,'A_pool_oracle':0,'full300_oracle':0,'images':548,'budget_binding_images':0,'Ngt_gt300_images':0,'Ngt_gt300_GT':0} for th in (50,75)}
    for im in ims:
     if time.monotonic()-start>300:raise TimeoutError('Frozen300-second budget')
     phase=f'{state} seed{seed} image{im["id"]}';p=REP/f'seed{seed}_{state}/image_{im["id"]:06d}.npz';track(p);assert inputs[str(p)]==declared[str(p)]
     with np.load(p,allow_pickle=False) as z:pb=torch.from_numpy(z['final_boxes']);lg=torch.from_numpy(z['final_logits'])
     scores=lg.sigmoid();av,af=scores.flatten().topk(300);af=af[av>.001];aq=(af//10).numpy();ac=(af%10).numpy()
     assert (pb[aq,2:]>0).all() and torch.isfinite(pb).all() and torch.isfinite(lg).all()
     inv=int((pb[:,2:]<=0).any(-1).sum())
     if inv:invalid.append({'state':state,'seed':seed,'image_id':im['id'],'nonpositive_raw_queries':inv})
     e=A.edges_norm(pb)*torch.tensor([im['width'],im['height']]*2);xywh=torch.cat((e[:,:2],e[:,2:]-e[:,:2]),-1).numpy().astype(np.float64)
     aa=by[im['id']];g=np.asarray([a['bbox'] for a in aa],dtype=np.float64).reshape(-1,4);gl=np.asarray([a['category_id'] for a in aa]);M=np.asarray(maskUtils.iou(xywh,g,[0]*len(aa)),dtype=np.float64).reshape(300,len(aa))
     assert np.isfinite(M).all() and (M>=0).all() and (M<=1).all()
     for th in (50,75):
      fullsum=Asum=0
      for c in range(10):
       ix=np.flatnonzero(gl==c);eligible=scores[:,c].numpy()>.001;mask=(M[:,ix]>=th/100)&eligible[:,None]
       mf=A.maxmatch(mask);qa=aq[ac==c];ma=A.maxmatch(mask[qa]);fc=min(100,mf);acap=min(100,ma)
       fullsum+=fc;Asum+=acap;classes.append({'state':state,'seed':seed,'image_id':im['id'],'IoU':th/100,'class':c,'GT':len(ix),'full_local_match':mf,'full_cap100_bound':fc,'A_pool_local_match':ma,'A_pool_cap100_bound':acap})
      fullbound=min(300,fullsum);Abound=min(300,Asum);a=actual[state,seed,im['id'],th];n=len(aa)
      assert 0<=a<=Abound<=fullbound<=min(n,300),(phase,th,a,Abound,fullbound,n)
      row={'state':state,'seed':seed,'image_id':im['id'],'IoU':th/100,'GT':n,'A_actual':a,'A_pool_oracle':Abound,'full300_oracle':fullbound,'class_bounds_sum_before_total300':fullsum,'total300_binds':int(fullsum>300),'GT_more_than300':int(n>300)};rows.append(row)
      t=totals[th]
      for k,v in [('GT',n),('A_actual',a),('A_pool_oracle',Abound),('full300_oracle',fullbound),('budget_binding_images',int(fullsum>300)),('Ngt_gt300_images',int(n>300)),('Ngt_gt300_GT',n if n>300 else 0)]:t[k]+=v
    assert totals[50]['GT']==totals[75]['GT']==38759
    summary[state][str(seed)]={str(th):v for th,v in totals.items()}
    print(state,seed,json.dumps(summary[state][str(seed)]),flush=True)
  csvout('per_image_bounds.csv',rows);csvout('per_image_class_bounds.csv',classes)
  write('results.json',{'status':'COMPLETE','protocol_sha256':ph,'conditions':summary,'invalid_raw_rows_preserved':invalid,
   'meaning':'GT-oracle capacity with arbitrary nonunique matching, NOT native attainable TP or AP. No perGT recoverable labels or class allocation invented under binding total budget.',
   'operations':{'evaluator_calls':0,'model_forwards':0,'head_calls':0,'training':0},'elapsed_seconds':time.monotonic()-start})
  assert all(A.sha(p)==h for p,h in inputs.items());write('input_manifest.json',inputs);write('verification.json',{'status':'PASS','per_image_rows':len(rows),'all_bound_order_invariants':True,'unchanged_inputs':len(inputs),'protocol_unchanged':A.sha(HERE/'protocol.json')==ph,'new_forwards':0,'new_evaluations':0})
 except Exception as e:
  write('failure.json',{'status':'FAILED_PRESERVED','phase':phase,'error':repr(e),'traceback':traceback.format_exc(),'protocol_sha256':ph,'elapsed_seconds':time.monotonic()-start});raise
if __name__=='__main__':main()
