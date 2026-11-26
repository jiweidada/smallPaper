"""One frozen pre-observation proxy; no detector execution or fitted model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import sys,json,csv,gzip,time,traceback,importlib.util
from pathlib import Path
from collections import defaultdict,Counter
sys.dont_write_bytecode=True
import numpy as np
import torch
HERE=Path(__file__).resolve().parent; O=HERE/'attempt01'; ROOT=HERE.parent.parent
PAIR=ROOT/'raw_EMA逐GT对照_20261026_v1'; REP=PAIR/'同后端CPU重放_20261026_v1'
TRANS=PAIR/'rawBC操作地图与EMA迁移_20261026_v1'
VAL=Path(r'C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val/annotations/val.json')
spec=importlib.util.spec_from_file_location('prior_audit',ROOT/'问题优先诊断_20261026_v1/audit.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A);torch.set_num_threads(2)
from pycocoeval import mask as maskUtils

def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(n,j):
 with open(O/n,'x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvout(n,rr):
 with open(O/n,'x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def boxes(pb,im):
 e=A.edges_norm(pb)*torch.tensor([im['width'],im['height']]*2)
 return torch.cat((e[:,:2],e[:,2:]-e[:,:2]),-1).numpy().astype(np.float64)
def iou(a,b):return np.asarray(maskUtils.iou(a,b,[0]*len(b)),dtype=np.float64).reshape(len(a),len(b))
def critical(mask):
 m=A.maxmatch(mask);out=[];checks=0
 if 0<m<=100:
  for q in np.flatnonzero(mask.any(1)):
   old=mask[q].copy();mask[q]=False;mr=A.maxmatch(mask);mask[q]=old;checks+=1
   assert m-1<=mr<=m
   if mr<m:out.append(int(q))
 return m,out,checks
def degbin(x):return np.searchsorted([1,2,4,8],x,side='right')
def estimate(counts):
 d,c,b,a=counts.T;n1=a+b;n0=c+d;n=n1+n0;good=(n1>0)&(n0>0)
 if not good.any():return None,{'overlap_strata':0,'overlap_candidates':0,'overlap_proxy1':0,'overlap_proxy0':0}
 w=n1[good]*n0[good]/n[good]
 rd=float(np.sum(w*(a[good]/n1[good]-c[good]/n0[good]))/np.sum(w))
 return rd,{'overlap_strata':int(good.sum()),'overlap_candidates':int(n[good].sum()),'overlap_proxy1':int(n1[good].sum()),'overlap_proxy0':int(n0[good].sum()),'weight_sum':float(w.sum())}
def analyze_sparse(records,nstrata,rng):
 if not records:return {'risk_difference':None,'image_bootstrap95':None,'valid_bootstrap_replicates':0,'counts_proxy0_false_true_proxy1_false_true':[0,0,0,0],'total_candidates':0,'overlap_strata':0,'overlap_candidates':0,'overlap_proxy1':0,'overlap_proxy0':0}
 rr=np.asarray(records,dtype=np.int64);im,code,cnt=rr.T
 ct=np.bincount(code,weights=cnt,minlength=nstrata*4).reshape(nstrata,4)
 rd,meta=estimate(ct);boot=[]
 if rd is not None:
  for _ in range(500):
   iw=rng.multinomial(548,np.full(548,1/548))
   cc=np.bincount(code,weights=cnt*iw[im],minlength=nstrata*4).reshape(nstrata,4)
   z,_=estimate(cc)
   if z is not None:boot.append(z)
 interval=np.quantile(boot,[.025,.975]).tolist() if boot else None
 return {'risk_difference':rd,'image_bootstrap95':interval,'valid_bootstrap_replicates':len(boot),'counts_proxy0_false_true_proxy1_false_true':ct.sum(0).astype(int).tolist(),'total_candidates':int(ct.sum()),**meta}

def main():
 start=time.monotonic();assert not O.exists();O.mkdir();phase='inputs';inputs={};checks=Counter()
 ph=A.sha(HERE/'protocol.json');assert ph==(HERE/'protocol.sha256').read_text('utf-8').strip()
 def track(p):inputs[str(p)]=A.sha(p)
 try:
  pri=HERE.parent/'matching_and_witness/attempt01/results.json'
  rawgt=PAIR/'后续零训练预检_20261026_v1/B_per_GT.csv'
  rawpairs=TRANS/'raw_operations_v1/pair_operation_map.csv.gz'
  oldpaths={'raw':TRANS/'raw_operations_v1/results.json','EMA':TRANS/'EMA_transfer_v1/results.json'}
  for p in (HERE/'protocol.json',Path(__file__),Path(A.__file__),VAL,pri,rawgt,rawpairs,REP/'complete.json',REP/'preflight.json',*oldpaths.values()):
   track(p)
  for n in ('mask.py','_mask.cp310-win_amd64.pyd'):track(A.EVALROOT/'pycocoeval'/n)
  oldarms={k:read(p) for k,p in oldpaths.items()}
  prior=read(pri);pref=read(REP/'preflight.json');comp=read(REP/'complete.json')
  for p,h in comp['source_sha256'].items():
   if p.endswith('replay.py') or p.endswith('dfine_decoder.py'):
    track(p);assert inputs[p]==h
  declared={r['path']:r['sha256'] for r in pref['entries']+comp['entries']}
  ds=read(VAL);ims=sorted(ds['images'],key=lambda x:x['id']);assert len(ims)==548 and len(ds['annotations'])==38759
  by=defaultdict(list)
  for a in ds['annotations']:by[a['image_id']].append(a)
  with open(rawgt,encoding='utf-8-sig') as f:gg=list(csv.DictReader(f))
  flags={(int(r['seed']),int(r['annotation_id'])):r for r in gg};assert len(flags)==77518
  groups=defaultdict(lambda:defaultdict(list))
  with gzip.open(rawpairs,'rt',encoding='utf-8-sig') as f:
   for r in csv.DictReader(f):
    seed=int(r['seed']);iid=int(r['image_id']);flat=int(r['query'])*10+int(r['class'])
    for th in (50,75):
     for arm in ('B','C'):
      aid=int(r[f'{arm}_match_GT{th}'])
      if aid<0:continue
      z=flags[seed,aid];a,b,c=(int(z[f'{x}{th}']) for x in ('A','B','C'))
      if arm=='B' and not a and b:groups[seed,iid,th]['B_new_vsA'].append(flat)
      if arm=='C' and not b and c:groups[seed,iid,th]['C_new_vsB'].append(flat)
      if arm=='B' and b and not c:groups[seed,iid,th]['B_lost_toC'].append(flat)
      if arm=='C' and b and c:groups[seed,iid,th]['common_C'].append(flat)
  imgrows=[];classrows=[];summary={};witness=defaultdict(Counter)
  out=gzip.open(O/'B_union_C_pair_features.csv.gz','xt',encoding='utf-8',newline='')
  fields=['state','seed','image_id','flat_pair','score_band','is_argmax','final_neighbor_band','pre_edge_band','pre_maxIoU_band','proxy_critical','GT_critical50','GT_critical75','GT_has_edge50','qualified_unsaturated50','B_export','C_export']
  writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
  for ci,(state,seed) in enumerate((('raw',1),('raw',2),('EMA',1),('EMA',2))):
   condition=Counter();m1records=[];m2records=[];qrecords=[];urecords=[]
   for ii,im in enumerate(ims):
    if time.monotonic()-start>900:raise TimeoutError('Frozen900-second budget')
    iid=im['id'];phase=f'{state} seed{seed} image{iid}';p=REP/f'seed{seed}_{state}/image_{iid:06d}.npz';track(p);assert inputs[str(p)]==declared[str(p)]
    with np.load(p,allow_pickle=False) as z:
     pb=torch.from_numpy(z['final_boxes']);lg=torch.from_numpy(z['final_logits']);preb=torch.from_numpy(z['pre_boxes']);prelg=torch.from_numpy(z['pre_logits'])
    assert pb.shape==preb.shape==(300,4) and lg.shape==prelg.shape==(300,10)
    assert all(torch.isfinite(x).all() for x in (pb,lg,preb,prelg))
    s=lg.sigmoid();ps=prelg.sigmoid().numpy();eligible=s.numpy()>.001
    av,af=s.flatten().topk(300);af=af[av>.001];U=torch.unique(af//10,sorted=True)
    bv,bf=s[U].flatten().topk(600);cv,cf=s.flatten().topk(600);assert (bv>.001).all() and (cv>.001).all()
    bf=(U[bf//10]*10+bf%10).numpy();cf=cf.numpy();union=np.union1d(bf,cf)
    bset=set(bf.tolist());cset=set(cf.tolist())
    for arm,fl in (('B',bf),('C',cf)):
     condition[arm+'_exported']+=len(fl);condition[arm+'_evaluated']+=len(A.cap_indices(fl%10))
    fxy=boxes(pb,im);pxy=boxes(preb,im);pp=iou(pxy,pxy);ff=iou(fxy,fxy);fp=iou(fxy,pxy)
    validpre=(pxy[:,2:]>0).all(1);units=[];proxy_masks=[];pranks=[];local_proxy=[]
    for c in range(10):
     qs=np.flatnonzero((ps[:,c]>.001)&validpre);order=qs[np.lexsort((qs,-ps[qs,c]))];kept=[]
     while len(order):
      q=int(order[0]);kept.append(q);order=order[1:];order=order[pp[q,order]<=.50]
     u=np.asarray(kept,dtype=np.int64);units.append(u)
     pm=(fp[:,u]>=.50)&eligible[:,c,None];m,crit,nn=critical(pm)
     proxy_masks.append(pm);pranks.append(m);local_proxy.extend(q*10+c for q in crit);checks['proxy_single_deletions']+=nn
    psum=sum(min(100,m) for m in pranks);pcap=min(300,psum)
    pcrit=set(local_proxy) if psum<=300 else set()
    # When the full global sum exceeds300 every single deletion leaves its truncated rank unchanged.
    if psum>300:checks['images_global_proxy_saturated']+=1
    aa=by[iid];g=np.asarray([a['bbox'] for a in aa],dtype=np.float64).reshape(-1,4);gl=np.asarray([a['category_id'] for a in aa]);fg=iou(fxy,g)
    truth={};ranks={};gtclassranks={};qualified={}
    for th in (50,75):
     tc=set();full=0;cm=[];has=np.zeros((300,10),dtype=bool)
     for c in range(10):
      gm=(fg[:,gl==c]>=th/100)&eligible[:,c,None];m,cc,nn=critical(gm);full+=min(100,m);tc.update(q*10+c for q in cc);checks['GT_single_deletions']+=nn;cm.append(m);has[:,c]=gm.any(1)
     assert full<=300;truth[th]=tc;ranks[th]=full;gtclassranks[th]=cm;qualified[th]=has
     condition[f'GT_critical{th}']+=len(tc)
    q=union//10;c=union%10
    scoreband=np.searchsorted([.003,.01,.03,.1,.3,.5,.7,.9],s.numpy()[q,c],side='right')
    arg=(s.argmax(1).numpy()[q]==c).astype(int)
    degree=np.asarray([int(((ff[q0]>=.50)&eligible[:,c0]).sum())-1 for q0,c0 in zip(q,c)])
    assert (degree>=0).all();fb=degbin(degree)
    predegree=np.asarray([int(proxy_masks[c0][q0].sum()) for q0,c0 in zip(q,c)])
    premax=np.asarray([float(fp[q0,units[c0]].max()) if len(units[c0]) else 0 for q0,c0 in zip(q,c)])
    predeg=degbin(predegree);premaxband=np.searchsorted([.50,.60,.70,.80,.90],premax,side='right')
    xp=np.asarray([int(f in pcrit) for f in union]);yt=np.asarray([int(f in truth[50]) for f in union])
    qf=qualified[50][q,c];uf=qf&np.asarray([gtclassranks[50][c0]<=100 and pranks[c0]<=100 and psum<=300 for c0 in c])
    st1=(((c*9+scoreband)*2+arg)*5+fb).astype(int);st2=((st1*5+predeg)*6+premaxband).astype(int)
    for st,records in ((st1,m1records),(st2,m2records)):
     code=st*4+xp*2+yt;uc,ct=np.unique(code,return_counts=True);records.extend((ii,int(k),int(n)) for k,n in zip(uc,ct))
    for ok,records in ((qf,qrecords),(uf,urecords)):
     code=(st2*4+xp*2+yt)[ok];uc,ct=np.unique(code,return_counts=True);records.extend((ii,int(k),int(n)) for k,n in zip(uc,ct))
    for j,f in enumerate(union):
     writer.writerow(dict(zip(fields,[state,seed,iid,int(f),int(scoreband[j]),int(arg[j]),int(fb[j]),int(predeg[j]),int(premaxband[j]),int(xp[j]),int(yt[j]),int(f in truth[75]),int(qf[j]),int(uf[j]),int(f in bset),int(f in cset)])))
    for c0 in range(10):classrows.append({'state':state,'seed':seed,'image_id':iid,'class':c0,'eligible_final_pairs':int(eligible[:,c0].sum()),'pre_units':len(units[c0]),'proxy_class_rank':pranks[c0],'proxy_class100_truncated':int(pranks[c0]>100),'GT_class_rank50':gtclassranks[50][c0],'GT_class_rank75':gtclassranks[75][c0],'GT_class100_truncated50':int(gtclassranks[50][c0]>100),'GT_class100_truncated75':int(gtclassranks[75][c0]>100),'proxy_global300_truncated':int(psum>300)})
    rr={'state':state,'seed':seed,'image_id':iid,'GT':len(aa),'eligible_final_pairs':int(eligible.sum()),'B_union_C_pairs':len(union),'pre_units':sum(map(len,units)),'excluded_nonpositive_pre_pairs':int(((ps>.001)&~validpre[:,None]).sum()),'proxy_uncapped_classsum':psum,'proxy_capacity':pcap,'proxy_critical_full':len(pcrit),'proxy_critical_union':int(xp.sum()),'truth_critical50_full':len(truth[50]),'truth_critical75_full':len(truth[75]),'truth_critical50_union':int(yt.sum()),'truth_and_proxy50_union':int((yt*xp).sum()),'proxy_global_saturated':int(psum>300)}
    imgrows.append(rr)
    for k,v in rr.items():
     if isinstance(v,(int,np.integer)) and k not in ('seed','image_id'):condition[k]+=int(v)
    if state=='raw':
     for th in (50,75):
      for label,fl in groups[seed,iid,th].items():
       zz=witness[seed,th,label]
       for f in fl:
        zz['actual_GT_witnesses']+=1;zz['true_critical']+=int(f in truth[th]);zz['proxy_critical']+=int(f in pcrit);zz['both']+=int(f in pcrit and f in truth[th])
   assert condition['GT']==38759
   for arm in ('B','C'):
    old=oldarms[state]['seeds'][str(seed)][arm] if state=='raw' else oldarms[state]['seeds'][str(seed)]['arms'][arm]
    assert condition[arm+'_exported']==old['exported'] and condition[arm+'_evaluated']==old['evaluated']
   for th in (50,75):assert condition[f'GT_critical{th}']==prior['conditions'][f'{state}_s{seed}_{th}_A']['full_critical_pair_nodes']
   rng=np.random.default_rng(20261027+ci)
   m1=analyze_sparse(m1records,900,rng);m2=analyze_sparse(m2records,27000,rng);mq=analyze_sparse(qrecords,27000,rng);mu=analyze_sparse(urecords,27000,rng)
   gate=all(z['image_bootstrap95'] is not None and z['image_bootstrap95'][0]>0 for z in (m2,mq,mu))
   summary[f'{state}_s{seed}']={'counts':dict(condition),'M1':m1,'M2':m2,'M2_GT_qualified':mq,'M2_GT_qualified_unsaturated':mu,'information_gate_pass':bool(gate)}
   print(state,seed,'M2',m2['risk_difference'],m2['image_bootstrap95'],'proxy/full',condition['proxy_critical_full'],'elapsed',time.monotonic()-start,flush=True)
  out.close();csvout('per_image.csv',imgrows);csvout('per_image_class.csv',classrows)
  for seed in (1,2):
   for th in (50,75):
    old=prior['raw_actual_pair_witnesses'][f's{seed}_{th}']
    for label,oldlabel in (('C_new_vsB','C_new'),('B_lost_toC','C_lost'),('common_C','common_C')):
     assert witness[seed,th,label]['actual_GT_witnesses']==old[oldlabel+'_GT']
     assert witness[seed,th,label]['true_critical']==old[oldlabel+'_critical_matched_pair']
  assert len(imgrows)==2192 and all(A.sha(p)==h for p,h in inputs.items())
  passed=all(z['information_gate_pass'] for z in summary.values())
  write('results.json',{'status':'COMPLETE','protocol_sha256':ph,'conditions':summary,'raw_actual_witnesses':{f's{s}_{th}_{label}':dict(z) for (s,th,label),z in witness.items()},'checks':dict(checks),'strict_information_advance_gate_pass':passed,'decision':'advance_information_only' if passed else 'stop_this_frozen_pre_proxy_advancement; no absence-of-information theorem or closure of other research lines','operations':{'model_forwards':0,'head_calls':0,'checkpoint_loads':0,'training':0,'evaluator_calls':0},'elapsed_seconds':time.monotonic()-start})
  write('input_manifest.json',inputs);write('verification.json',{'status':'PASS','all2192_images':True,'prior_GT_critical_aggregate_all8_exact':True,'unchanged_inputs':len(inputs),'protocol_unchanged':A.sha(HERE/'protocol.json')==ph,'output_MiB':sum(p.stat().st_size for p in O.rglob('*') if p.is_file())/2**20})
 except Exception as e:
  write('failure.json',{'status':'FAILED_PRESERVED','phase':phase,'error':repr(e),'traceback':traceback.format_exc(),'protocol_sha256':ph,'elapsed_seconds':time.monotonic()-start});raise
if __name__=='__main__':main()
