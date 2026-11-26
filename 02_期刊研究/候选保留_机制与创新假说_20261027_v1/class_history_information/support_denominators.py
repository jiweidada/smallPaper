"""Complete image denominators for already-frozen primary M2; no new test."""
import sys,json,csv,gzip,itertools,time,importlib.util
from pathlib import Path
from collections import defaultdict
sys.dont_write_bytecode=True
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;O=HERE/'support_denominators_v1'
spec=importlib.util.spec_from_file_location('frozen_class',HERE/'run.py');H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H);torch.set_num_threads(2)
def main():
 assert not O.exists();O.mkdir();start=time.monotonic();prior=H.read(HERE/'attempt01/results.json');data=defaultdict(list);inputs={}
 def track(p):inputs[str(p)]=H.A.sha(p)
 for p in (Path(__file__),HERE/'protocol.json',HERE/'run.py',HERE/'attempt01/results.json',HERE/'attempt01/verification.json',H.PRES/'B_union_C_pair_features.csv.gz'):track(p)
 with gzip.open(H.PRES/'B_union_C_pair_features.csv.gz','rt',encoding='utf-8') as f:
  for (state,seed,iid),group in itertools.groupby(csv.DictReader(f),key=lambda r:(r['state'],int(r['seed']),int(r['image_id']))):
   rr=list(group);p=H.REP/f'seed{seed}_{state}/image_{iid:06d}.npz';track(p)
   with np.load(p,allow_pickle=False) as z:lg=torch.from_numpy(z['final_logits']);pl=torch.from_numpy(z['pre_logits'])
   s=lg.sigmoid();sp=s.numpy();av,af=s.flatten().topk(300);af=af[av>.001].numpy();U=np.unique(af//10);bv,bi=s[torch.from_numpy(U)].flatten().topk(600);bf=U[bi.numpy()//10]*10+bi.numpy()%10
   union=np.asarray([int(r['flat_pair']) for r in rr]);q=union//10;c=union%10;extra=np.isin(union,np.setdiff1d(bf,af));assert extra.sum()==300
   top2=np.sort(sp,axis=1)[:,-2:];margin=top2[:,1]-top2[:,0];soft=torch.softmax(lg,1);certainty=(1+(soft*soft.log()).sum(1)/np.log(10)).numpy()
   sb=np.asarray([int(r['score_band']) for r in rr]);mb=np.searchsorted([.003,.01,.03,.1,.3,.5,.7,.9],sp.max(1)[q],side='right');gb=np.searchsorted([.01,.03,.1,.3],margin[q],side='right');cb=np.searchsorted([.25,.50,.75],certainty[q],side='right')
   rank=np.asarray([1+np.sum(sp[q0]>sp[q0,c0])+np.sum((sp[q0]==sp[q0,c0])&(np.arange(10)<c0)) for q0,c0 in zip(q,c)]);rb=np.searchsorted([2,3,5],rank,side='right')
   st=(((((c*9+sb)*4+rb)*9+mb)*5+gb)*4+cb).astype(np.int64)
   fd=np.asarray([int(r['final_neighbor_band']) for r in rr]);pd=np.asarray([int(r['pre_edge_band']) for r in rr]);pi=np.asarray([int(r['pre_maxIoU_band']) for r in rr]);st=((st*5+fd)*5+pd)*6+pi
   x=(pl.argmax(1).numpy()[q]==c).astype(int);y=np.asarray([int(r['GT_has_edge50']) for r in rr])
   data[state,seed].append(np.stack((np.full(extra.sum(),iid),st[extra],x[extra],y[extra]),1))
 results={};rows=[]
 for (state,seed),parts in data.items():
  z=np.concatenate(parts);iid,st,x,y=z.T;u,ind=np.unique(st,return_inverse=True);ct=np.bincount(ind*4+x*2+y,minlength=len(u)*4).reshape(-1,4);good=(ct[:,:2].sum(1)>0)&(ct[:,2:].sum(1)>0);keep=good[ind]
  rd,meta=H.P.estimate(ct);old=prior['conditions'][f'{state}_s{seed}']['domains']['B_minus_A']['M2'];assert abs(rd-old['risk_difference'])<1e-14 and meta['overlap_candidates']==old['overlap_candidates'] and meta['overlap_strata']==old['overlap_strata']
  r={'full_images':548,'images_with_at_least_one_supported_candidate':int(len(np.unique(iid[keep]))),'full_candidate_pairs':len(z),'supported_candidate_pairs':int(keep.sum()),'supported_strata':int(good.sum()),'weighted_RD_exactly_reproduced':rd};results[f'{state}_s{seed}']=r
  for image in np.unique(iid):
   b=iid==image;rows.append({'state':state,'seed':seed,'image_id':int(image),'full_pairs':int(b.sum()),'supported_pairs':int((b&keep).sum()),'supported_pre_argmax':int((b&keep&(x==1)).sum()),'supported_qualified50':int((b&keep&(y==1)).sum())})
 assert all(H.A.sha(p)==h for p,h in inputs.items())
 with open(O/'per_image.csv','x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (O/'results.json').write_text(json.dumps({'status':'PASS','meaning':'Denominator completion only; all frozen primary M2 risk differences reproduced, no new bins/statistical tests or output rule.','conditions':results,'unchanged_inputs':len(inputs),'model_forwards':0,'evaluator_calls':0,'training':0,'elapsed_seconds':time.monotonic()-start},ensure_ascii=False,indent=2),encoding='utf-8')
 (O/'input_manifest.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(results,ensure_ascii=False))
if __name__=='__main__':main()
