"""One frozen equal-budget output swap, without detector execution."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import sys,json,csv,copy,time,io,contextlib,traceback,importlib.util
from pathlib import Path
from collections import Counter
sys.dont_write_bytecode=True
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;O=HERE/'attempt01';ROOT=HERE.parent.parent
spec=importlib.util.spec_from_file_location('previous_cap',HERE.parent/'cap_dependency/run.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C);A=C.A;torch.set_num_threads(2)
PAIR=C.PAIR;REP=C.REP;VAL=C.VAL
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(n,j):
 with open(O/n,'x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvout(n,rr):
 with open(O/n,'x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def band(s):return np.searchsorted([.003,.01,.03,.1,.3,.5,.7,.9],s,side='right')
def main():
 start=time.monotonic();assert not O.exists();O.mkdir();ph=A.sha(HERE/'protocol.json');assert ph==(HERE/'protocol.sha256').read_text('utf-8').strip();phase='inputs';inputs={};calls=0;result={};swaprows=[];gtrows=[]
 def track(p):inputs[str(p)]=A.sha(p)
 def budget():
  if time.monotonic()-start>900:raise TimeoutError('Frozen900-second budget')
 try:
  hp=HERE.parent/'class_history_information';priorpath=REP/'analysis_v1/results.json';oldgtpath=REP/'analysis_v1/per_GT_stages_and_transitions.csv'
  for p in (HERE/'protocol.json',Path(__file__),Path(C.__file__),Path(A.__file__),VAL,REP/'preflight.json',REP/'complete.json',priorpath,oldgtpath,hp/'protocol.json',hp/'attempt01/results.json',hp/'attempt01/verification.json'):track(p)
  assert read(hp/'attempt01/verification.json')['status']=='PASS' and read(hp/'attempt01/results.json')['strict_information_advance_gate_pass']
  for n in ('coco.py','cocoeval.py','mask.py','_mask.cp310-win_amd64.pyd'):track(A.EVALROOT/'pycocoeval'/n)
  comp=read(REP/'complete.json');pref=read(REP/'preflight.json');declared={r['path']:r['sha256'] for r in comp['entries']+pref['entries']};prior=read(priorpath)
  ds=read(VAL);ims=sorted(ds['images'],key=lambda x:x['id']);anns=sorted(ds['annotations'],key=lambda x:x['id']);ids=np.asarray([a['id'] for a in anns]);ai={int(a):i for i,a in enumerate(ids)}
  pos=copy.deepcopy(ds)
  for a in pos['annotations']:a['id']+=1
  with open(oldgtpath,encoding='utf-8-sig') as f:old=list(csv.DictReader(f))
  oldhits={(state,seed,th):{int(r['annotation_id']) for r in old if int(r['seed'])==seed and r[f'{state}_hit{th}']=='1'} for state in ('raw','EMA') for seed in (1,2) for th in (50,75)}
  for state in ('raw','EMA'):
   for seed in (1,2):
    predictions={'A':[],'H':[],'S':[]};swaps=0;different=0;capN=0
    for im in ims:
     budget();iid=im['id'];phase=f'{state} seed{seed} image{iid}';p=REP/f'seed{seed}_{state}/image_{iid:06d}.npz';track(p);assert inputs[str(p)]==declared[str(p)]
     with np.load(p,allow_pickle=False) as z:pb=torch.from_numpy(z['final_boxes']);lg=torch.from_numpy(z['final_logits']);pl=torch.from_numpy(z['pre_logits'])
     prob=lg.sigmoid();flatprob=prob.flatten().numpy();av,af=prob.flatten().topk(300);assert (av>.001).all();af=af.numpy();U=np.unique(af//10);bv,bi=prob[torch.from_numpy(U)].flatten().topk(600);bf=(U[bi.numpy()//10]*10+bi.numpy()%10);assert (bv>.001).all() and set(af)<=set(bf)
     argpre=pl.argmax(1).numpy();counts=Counter((af%10).tolist());aset=set(af.tolist());extra=np.asarray([f for f in bf if f not in aset]);bands=band(flatprob)
     h=af.copy();ss=af.copy();info={'state':state,'seed':seed,'image_id':iid,'swapped':0,'different_H_S':0,'donor':-1,'history_added':-1,'score_added':-1,'class':-1,'score_band':-1}
     for new in extra:
      q,c=divmod(int(new),10)
      if counts[c]>100 or int(argpre[q])!=c:continue
      donors=[int(f) for f in af if f%10==c and bands[f]==bands[new] and int(argpre[f//10])!=c]
      if not donors:continue
      donor=donors[-1];snew=next(int(f) for f in extra if f%10==c and bands[f]==bands[new])
      survivors=[int(f) for f in af if f!=donor]
      def ordered(add):
       fs=np.asarray(survivors+[int(add)]);return fs[np.argsort(-flatprob[fs],kind='stable')]
      h=ordered(new);ss=ordered(snew);swaps+=1;different+=int(new!=snew)
      info.update({'swapped':1,'different_H_S':int(new!=snew),'donor':donor,'history_added':int(new),'score_added':snew,'class':c,'score_band':int(bands[new]),'donor_score':float(flatprob[donor]),'H_added_score':float(flatprob[new]),'S_added_score':float(flatprob[snew])})
      assert donor in set(af[A.cap_indices(af%10)].tolist()) and new in set(h[A.cap_indices(h%10)].tolist()) and snew in set(ss[A.cap_indices(ss%10)].tolist())
      break
     # Fixed schema also for no-swap images.
     for k in ('donor_score','H_added_score','S_added_score'):info.setdefault(k,None)
     swaprows.append(info);basecap=len(A.cap_indices(af%10));capN+=basecap
     e=A.edges_norm(pb)*torch.tensor([im['width'],im['height']]*2)
     for arm,ff in (('A',af),('H',h),('S',ss)):
      assert len(ff)==300 and len(set(ff.tolist()))==300 and Counter((ff%10).tolist())==counts and len(A.cap_indices(ff%10))==basecap
      assert (pb[ff//10,2:]>0).all()
      for f0 in ff:
       q,c=divmod(int(f0),10);b=e[q];xywh=torch.cat((b[:2],b[2:]-b[:2])).tolist();predictions[arm].append({'image_id':iid,'category_id':c,'bbox':xywh,'score':float(flatprob[f0])})
    arms={};hits={}
    for arm in ('A','H','S'):
     budget();phase=f'{state} seed{seed} arm{arm} eval';log=io.StringIO()
     with contextlib.redirect_stdout(log):
      gt=A.COCO();gt.dataset=copy.deepcopy(pos);gt.createIndex();dt=gt.loadRes(copy.deepcopy(predictions[arm]));ev=A.COCOeval(gt,dt,'bbox',using_tiny_metrice=False,using_vtiny_metrice=False,max_dets=100);ev.params.imgIds=sorted(gt.getImgIds());ev.evaluate();ev.accumulate();ev.summarize()
     calls+=1
     with open(O/f'{state}_s{seed}_{arm}_eval.txt','x',encoding='utf-8') as f:f.write(log.getvalue())
     hh=np.zeros((len(anns),10),dtype=bool);evaluated=0;seen=[]
     for e in ev.evalImgs:
      if e is None or tuple(e['aRng'])!=tuple(ev.params.areaRng[0]):continue
      assert not np.asarray(e['gtIgnore']).any();seen.extend(int(a)-1 for a in e['gtIds']);n=len(e['dtIds']);evaluated+=n
      for t in range(10):
       for j in range(n):
        aid=int(e['dtMatches'][t,j])
        if aid>0 and not e['dtIgnore'][t,j]:hh[ai[aid-1],t]=True
     assert len(seen)==len(anns) and set(seen)==set(ids.tolist()) and evaluated==capN
     mm=dict(zip(C.MET,map(float,ev.stats)));hits[arm]=hh
     arms[arm]={'metrics':mm,'TP_all10':hh.sum(0).tolist(),'evaluated':evaluated,'exported':len(predictions[arm]),'FP50':evaluated-int(hh[:,0].sum())}
     if arm=='A':
      oldsummary=prior['summaries'][f'{seed}_{state}'];assert arms[arm]['TP_all10']==oldsummary['GT_hits_all10']
      for th,t in ((50,0),(75,5)):assert set(ids[np.flatnonzero(hh[:,t])].tolist())==oldhits[state,seed,th]
      arms[arm]['metrics_vs_prior_original_ID_convention_delta_pp']={k:100*(mm[k]-oldsummary['metrics'][k]) for k in C.MET}
     with open(O/f'{state}_s{seed}_{arm}_GT_hits.npz','xb') as f:np.savez_compressed(f,annotation_id=ids,hit=hh)
     print(phase,'AP',mm['AP']*100,'TP50',int(hh[:,0].sum()),'elapsed',time.monotonic()-start,flush=True);del gt,dt,ev
    contrasts={}
    for left,right in (('H','A'),('H','S'),('S','A')):
     ll=hits[left];rr=hits[right];d={k:100*(arms[left]['metrics'][k]-arms[right]['metrics'][k]) for k in C.MET}
     contrasts[left+'_minus_'+right]={'delta_pp':d,**{f'{kind}{th}_IDs':ids[np.flatnonzero((~rr[:,t]&ll[:,t]) if kind=='gained' else (rr[:,t]&~ll[:,t]))].tolist() for th,t in ((50,0),(75,5)) for kind in ('gained','lost')}}
    ha=contrasts['H_minus_A'];hs=contrasts['H_minus_S'];gate=ha['delta_pp']['AP']>.0001 and ha['delta_pp']['APs']>.0001 and len(ha['gained50_IDs'])>len(ha['lost50_IDs']) and hs['delta_pp']['AP']>0 and hs['delta_pp']['APs']>0 and len(hs['gained50_IDs'])>=len(hs['lost50_IDs']) and different>0
    result[f'{state}_s{seed}']={'arms':arms,'contrasts':contrasts,'images':548,'GT':38759,'images_swapped':swaps,'different_H_S_images':different,'strict_advance_gate_pass':bool(gate)}
    for j,a in enumerate(anns):gtrows.append({'state':state,'seed':seed,'image_id':a['image_id'],'annotation_id':a['id'],'class':a['category_id'],**{f'{arm}{th}':int(hits[arm][j,t]) for arm in ('A','H','S') for th,t in ((50,0),(75,5))}})
    write(f'{state}_s{seed}_done.json',result[f'{state}_s{seed}'])
  budget();assert calls==12 and all(A.sha(p)==h for p,h in inputs.items());csvout('per_image_swaps.csv',swaprows);csvout('per_GT.csv',gtrows)
  passed=all(z['strict_advance_gate_pass'] for z in result.values())
  write('results.json',{'status':'COMPLETE','protocol_sha256':ph,'conditions':result,'strict_readout_advance_gate_pass':passed,'decision':'advance this readout only; novelty still unproven' if passed else 'stop this fixed one-swap rule; semantic-information premise remains separate','operations':{'evaluator_calls':calls,'model_forwards':0,'head_calls':0,'checkpoint_loads':0,'training':0},'elapsed_seconds':time.monotonic()-start})
  write('input_manifest.json',inputs);write('verification.json',{'status':'PASS','all2192_source_NPZ_bound':True,'A_all10_counts_and_exact50_75_hitsets':True,'all_swaps_identical_budget_classcounts_and_capcounts':True,'all_swapped_pairs_actually_evaluated':True,'GT_rows':len(gtrows),'unchanged_inputs':len(inputs),'protocol_unchanged':A.sha(HERE/'protocol.json')==ph,'evaluator_calls':calls})
 except Exception as e:
  write('failure.json',{'status':'FAILED_PRESERVED','phase':phase,'error':repr(e),'traceback':traceback.format_exc(),'protocol_sha256':ph,'calls':calls,'elapsed_seconds':time.monotonic()-start});raise
if __name__=='__main__':main()
