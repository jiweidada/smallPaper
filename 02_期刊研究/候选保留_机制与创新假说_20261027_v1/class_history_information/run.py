"""One class-history information probe, following a failed geometric proxy."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import sys,json,csv,gzip,itertools,time,traceback,importlib.util
from pathlib import Path
from collections import defaultdict,Counter
sys.dont_write_bytecode=True
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;O=HERE/'attempt01';ROOT=HERE.parent.parent
PSRC=HERE.parent/'pre_witness_information';PRES=PSRC/'attempt01'
spec=importlib.util.spec_from_file_location('frozen_pre_probe',PSRC/'run.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P);A=P.A;torch.set_num_threads(2)
PAIR=P.PAIR;REP=P.REP;TRANS=P.TRANS;VAL=P.VAL
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(n,j):
 with open(O/n,'x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvout(n,rr):
 with open(O/n,'x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def sparse_add(records,ii,st,x,y,keep):
 code=(st*4+x*2+y)[keep];u,n=np.unique(code,return_counts=True);records.extend((ii,int(a),int(b)) for a,b in zip(u,n))
def analyze(rr,rng):
 if not rr:return P.analyze_sparse([],0,rng)
 z=np.asarray(rr,dtype=np.int64);uni=np.unique(z[:,1]//4);z[:,1]=np.searchsorted(uni,z[:,1]//4)*4+z[:,1]%4
 out=P.analyze_sparse(z.tolist(),len(uni),rng);out['observed_strata_without_merging']=len(uni);return out
def main():
 start=time.monotonic();assert not O.exists();O.mkdir();phase='inputs';inputs={};ph=A.sha(HERE/'protocol.json');assert ph==(HERE/'protocol.sha256').read_text('utf-8').strip()
 def track(p):inputs[str(p)]=A.sha(p)
 try:
  feature=PRES/'B_union_C_pair_features.csv.gz';rawgt=PAIR/'后续零训练预检_20261026_v1/B_per_GT.csv';rawpair=TRANS/'raw_operations_v1/pair_operation_map.csv.gz'
  for p in (HERE/'protocol.json',Path(__file__),PSRC/'run.py',PSRC/'protocol.json',PRES/'verification.json',PRES/'results.json',PRES/'input_manifest.json',feature,VAL,rawgt,rawpair,Path(A.__file__),REP/'complete.json',REP/'preflight.json'):track(p)
  assert read(PRES/'verification.json')['status']=='PASS'
  presrc=read(PRES/'input_manifest.json');assert inputs[str(feature)]==A.sha(feature)
  comp=read(REP/'complete.json');pref=read(REP/'preflight.json');declared={r['path']:r['sha256'] for r in comp['entries']+pref['entries']}
  for p,h in comp['source_sha256'].items():
   if p.endswith('replay.py') or p.endswith('dfine_decoder.py'):track(p);assert inputs[p]==h
  ds=read(VAL);ims=sorted(ds['images'],key=lambda x:x['id']);imidx={im['id']:(i,im) for i,im in enumerate(ims)};by=defaultdict(list)
  for a in ds['annotations']:by[a['image_id']].append(a)
  with open(rawgt,encoding='utf-8-sig') as f:gtflags={(int(r['seed']),int(r['annotation_id'])):r for r in csv.DictReader(f)}
  actual=defaultdict(list)
  with gzip.open(rawpair,'rt',encoding='utf-8-sig') as f:
   for r in csv.DictReader(f):
    seed=int(r['seed']);iid=int(r['image_id']);flat=int(r['query'])*10+int(r['class'])
    for th in (50,75):
     for arm in ('B','C'):
      aid=int(r[f'{arm}_match_GT{th}'])
      if aid<0:continue
      z=gtflags[seed,aid];aa,bb,cc=(int(z[f'{x}{th}']) for x in ('A','B','C'))
      if arm=='B':actual[seed,iid,th,'B_new_vsA' if not aa else 'B_common_A'].append(flat)
      if arm=='C' and not bb:actual[seed,iid,th,'C_new_vsB'].append(flat)
      if arm=='B' and bb and not cc:actual[seed,iid,th,'B_lost_toC'].append(flat)
  records=defaultdict(list);totals=defaultdict(Counter);witness=defaultdict(Counter);rows=[];seen=set()
  with gzip.open(feature,'rt',encoding='utf-8') as f:
   groups=itertools.groupby(csv.DictReader(f),key=lambda r:(r['state'],int(r['seed']),int(r['image_id'])))
   for (state,seed,iid),rr in groups:
    if time.monotonic()-start>900:raise TimeoutError('Frozen900-second budget')
    phase=f'{state} seed{seed} image{iid}';assert (state,seed,iid) not in seen;seen.add((state,seed,iid));fr=list(rr);ii,im=imidx[iid]
    p=REP/f'seed{seed}_{state}/image_{iid:06d}.npz';track(p);assert inputs[str(p)]==declared[str(p)]==presrc[str(p)]
    with np.load(p,allow_pickle=False) as z:lg=torch.from_numpy(z['final_logits']);pl=torch.from_numpy(z['pre_logits']);pb=torch.from_numpy(z['final_boxes'])
    s=lg.sigmoid();av,af=s.flatten().topk(300);af=af[av>.001];U=torch.unique(af//10,sorted=True);bv,bf=s[U].flatten().topk(600);cv,cf=s.flatten().topk(600);assert (bv>.001).all() and (cv>.001).all()
    af=af.numpy();bf=(U[bf//10]*10+bf%10).numpy();cf=cf.numpy();union=np.union1d(bf,cf);assert np.array_equal(union,np.asarray([int(r['flat_pair']) for r in fr])) and set(af)<=set(bf)
    extra=np.isin(union,np.setdiff1d(bf,af));assert extra.sum()==len(bf)-len(af)==300
    q=union//10;c=union%10;sp=s.numpy();argpre=pl.argmax(1).numpy();x=(argpre[q]==c).astype(int)
    y50=np.asarray([int(r['GT_has_edge50']) for r in fr]);crit=np.asarray([int(r['GT_critical50']) for r in fr])
    aa=by[iid];gl=np.asarray([a['category_id'] for a in aa]);g=np.asarray([a['bbox'] for a in aa],dtype=np.float64).reshape(-1,4);M=P.iou(P.boxes(pb,im),g)
    y75=((M[q]>=.75)&(c[:,None]==gl[None,:])).any(1).astype(int)
    ycheck=((M[q]>=.50)&(c[:,None]==gl[None,:])).any(1).astype(int);assert np.array_equal(y50,ycheck)
    top2=np.sort(sp,axis=1)[:,-2:];margin=top2[:,1]-top2[:,0];soft=torch.softmax(lg,1);certainty=(1+(soft*soft.log()).sum(1)/np.log(10)).numpy()
    scoreband=np.asarray([int(r['score_band']) for r in fr]);mxband=np.searchsorted([.003,.01,.03,.1,.3,.5,.7,.9],sp.max(1)[q],side='right')
    mgband=np.searchsorted([.01,.03,.1,.3],margin[q],side='right');ctband=np.searchsorted([.25,.50,.75],certainty[q],side='right')
    rank=np.asarray([1+np.sum(sp[q0]>sp[q0,c0])+np.sum((sp[q0]==sp[q0,c0])&(np.arange(10)<c0)) for q0,c0 in zip(q,c)])
    rb=np.searchsorted([2,3,5],rank,side='right')
    st1=(((((c*9+scoreband)*4+rb)*9+mxband)*5+mgband)*4+ctband).astype(np.int64)
    fd=np.asarray([int(r['final_neighbor_band']) for r in fr]);pd=np.asarray([int(r['pre_edge_band']) for r in fr]);pi=np.asarray([int(r['pre_maxIoU_band']) for r in fr])
    st2=((st1*5+fd)*5+pd)*6+pi
    masks={'B_minus_A':extra,'B_union_C':np.ones(len(union),dtype=bool),'GT_qualified':y50.astype(bool)}
    for domain,keep in masks.items():
     for level,st in (('M1',st1),('M2',st2)):
      yy=crit if domain=='GT_qualified' else y50;sparse_add(records[state,seed,domain,level],ii,st,x,yy,keep)
     z=totals[state,seed,domain];z['candidate_pairs']+=int(keep.sum());z['pre_argmax_candidate']+=int(x[keep].sum());z['GT_qualified50']+=int(y50[keep].sum());z['GT_qualified75']+=int(y75[keep].sum());z['GT_critical50']+=int(crit[keep].sum());z['pre_argmax_and_qualified50']+=int((x[keep]*y50[keep]).sum());z['pre_argmax_and_qualified75']+=int((x[keep]*y75[keep]).sum());z['pre_argmax_and_critical50']+=int((x[keep]*crit[keep]).sum())
     rows.append({'state':state,'seed':seed,'image_id':iid,'domain':domain,'GT':len(aa),'pairs':int(keep.sum()),'pre_argmax_candidate':int(x[keep].sum()),'qualified50':int(y50[keep].sum()),'pre_argmax_qualified50':int((x[keep]*y50[keep]).sum())})
    if state=='raw':
     lookup={int(f):j for j,f in enumerate(union)}
     for th in (50,75):
      for label in ('B_new_vsA','B_common_A','C_new_vsB','B_lost_toC'):
       z=witness[seed,th,label]
       for f0 in actual[seed,iid,th,label]:
        j=lookup[f0];z['actual_GT_witnesses']+=1;z['pre_argmax_candidate']+=int(x[j]);z['final_argmax_candidate']+=int(rank[j]==1);z['B_minus_A_pair']+=int(extra[j])
    if ii==547:print(state,seed,'cache features done',time.monotonic()-start,flush=True)
  assert len(seen)==2192 and all(z['candidate_pairs']==164400 for (st,ss,dom),z in totals.items() if dom=='B_minus_A')
  summary={}
  for ci,(state,seed) in enumerate((('raw',1),('raw',2),('EMA',1),('EMA',2))):
   rng=np.random.default_rng(20261127+ci);res={}
   for domain in ('B_minus_A','B_union_C','GT_qualified'):
    res[domain]={'counts':dict(totals[state,seed,domain]),'M1':analyze(records[state,seed,domain,'M1'],rng),'M2':analyze(records[state,seed,domain,'M2'],rng),'outcome':'critical50 among qualified' if domain=='GT_qualified' else 'qualified50'}
   primary=res['B_minus_A']['M2'];gate=primary['image_bootstrap95'] is not None and primary['image_bootstrap95'][0]>0
   summary[f'{state}_s{seed}']={'domains':res,'information_gate_pass':bool(gate)}
   print(state,seed,'primary M2',primary['risk_difference'],primary['image_bootstrap95'],'elapsed',time.monotonic()-start,flush=True)
  assert all(A.sha(p)==h for p,h in inputs.items())
  csvout('per_image.csv',rows);passed=all(z['information_gate_pass'] for z in summary.values())
  write('results.json',{'status':'COMPLETE','protocol_sha256':ph,'conditions':summary,'raw_actual_witnesses':{f's{s}_{th}_{label}':dict(z) for (s,th,label),z in witness.items()},'strict_information_advance_gate_pass':passed,'decision':'information-premise-only passes; novelty and actual intervention remain separate' if passed else 'stop advancement of this binary class-history premise under frozen gate; no information-absence theorem','operations':{'model_forwards':0,'head_calls':0,'checkpoint_loads':0,'evaluator_calls':0,'training':0},'elapsed_seconds':time.monotonic()-start})
  write('input_manifest.json',inputs);write('verification.json',{'status':'PASS','all2192_images':True,'all_union_pair_ID_lists_exact':True,'all_trueclass_edge50_labels_exact':True,'B_minus_A_300pairs_each_image':True,'unchanged_inputs':len(inputs),'protocol_unchanged':A.sha(HERE/'protocol.json')==ph})
 except Exception as e:
  write('failure.json',{'status':'FAILED_PRESERVED','phase':phase,'error':repr(e),'traceback':traceback.format_exc(),'protocol_sha256':ph,'elapsed_seconds':time.monotonic()-start});raise
if __name__=='__main__':main()
