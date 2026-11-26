"""Frozen lineage information test: matching marginals and identity nulls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import sys,json,csv,gzip,time,traceback,importlib.util
from pathlib import Path
from collections import defaultdict,Counter
sys.dont_write_bytecode=True
import numpy as np
import torch
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.optimize import _lsap
HERE=Path(__file__).resolve().parent;O=HERE/'attempt01';ROOT=HERE.parent
OLD=ROOT/'候选保留_机制与创新假说_20261027_v1'
spec=importlib.util.spec_from_file_location('frozen_pre',OLD/'pre_witness_information/run.py');F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F);A=F.A;torch.set_num_threads(2)
PAIR=F.PAIR;REP=F.REP;VAL=F.VAL
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(n,j):
 with open(O/n,'x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvout(n,rr):
 with open(O/n,'x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def scorebin(x):return np.searchsorted([.003,.01,.03,.1,.3,.5,.7,.9],x,side='right')
def degbin(x):return np.searchsorted([1,2,4,8],x,side='right')
def cmi(groups,y,j,nz):
 n=np.bincount(groups*6+j*2+y,minlength=nz*6).reshape(nz,3,2).astype(float)
 n_z=n.sum((1,2));n_zj=n.sum(2);n_zy=n.sum(1);N=n_z.sum()
 if N==0:return 0.,{'informative_pairs':0,'informative_strata':0,'positive_labels':0,'negative_labels':0},np.zeros(nz,dtype=bool)
 def xl(a):return np.sum(a[a>0]*np.log2(a[a>0]))
 hy=(xl(n_z)-xl(n_zy))/N;v=(xl(n)+xl(n_z)-xl(n_zj)-xl(n_zy))/N
 assert v>=-1e-12
 good=(n_zy>0).all(1)&((n_zj>0).sum(1)>=2)
 meta={'informative_pairs':int(n_z[good].sum()),'informative_strata':int(good.sum()),'positive_labels':int(n_zy[good,1].sum()),'negative_labels':int(n_zy[good,0].sum()),'HY_given_controls_upper_bound_bits':float(max(0,hy)),'singleton_strata':int((n_z==1).sum()),'outcome_mixed_strata':int((n_zy>0).all(1).sum()),'outcome_mixed_pairs':int(n_z[(n_zy>0).all(1)].sum()),'largest_stratum_weight':float(n_z.max()/N)}
 return float(max(0,v)),meta,good
def perm_j(pre,fin,q,pis):
 # Sets were frozen before relabeling; anchor fix changes only pi images.
 n=len(q);out=np.zeros((len(pis),n),dtype=np.uint8);edge_i,edge_r=np.where(pre & fin.any(1)[:,None]);edge_q=q[edge_i]
 if not len(edge_i):return out
 for k,pi in enumerate(pis):
  inv=np.argsort(pi);mapped=pi[edge_r].copy();fix=edge_r==inv[edge_q];mapped[fix]=pi[edge_q[fix]]
  assert not np.any(mapped==edge_q)
  jj=np.bincount(edge_i,weights=fin[edge_i,mapped],minlength=n);out[k]=np.minimum(2,jj).astype(np.uint8)
 return out
def permutations(seed,state,iid,blocks,kind):
 rng=np.random.default_rng(np.random.SeedSequence([20261027,seed,state,iid,kind]));out=[]
 for _ in range(199):
  p=np.arange(300)
  for b in blocks:p[b]=rng.permutation(b)
  assert len(np.unique(p))==300;out.append(p)
 return np.asarray(out)
def summarize(data):
 iid=np.concatenate([r['iid'] for r in data]);st=np.concatenate([r['strata'] for r in data]);y=np.concatenate([r['y'] for r in data]);jr=np.concatenate([r['real'] for r in data]);jg=np.concatenate([r['geo'] for r in data]);strong=np.concatenate([r['strong'] for r in data],axis=1);weak=np.concatenate([r['weak'] for r in data],axis=1);exactmove=np.concatenate([r['exactmove'] for r in data])
 unique,groups=np.unique(st,return_inverse=True);real,meta,good=cmi(groups,y,jr,len(unique));geo,_,_=cmi(groups,y,jg,len(unique));ss=[cmi(groups,y,j,len(unique))[0] for j in strong];ww=[cmi(groups,y,j,len(unique))[0] for j in weak]
 inf=good[groups] if len(groups) else np.zeros(0,dtype=bool);meta['informative_images']=int(len(np.unique(iid[inf])))
 yct=np.bincount(groups*2+y,minlength=len(unique)*2).reshape(-1,2);ymixed=(yct>0).all(1);meta['outcome_mixed_images']=int(len(np.unique(iid[ymixed[groups]])))
 eq=np.array_equal(np.bincount(groups*6+jr*2+y,minlength=len(unique)*6),np.bincount(groups*6+jg*2+y,minlength=len(unique)*6))
 feasible=meta['informative_pairs']>=100 and meta['informative_images']>=20 and min(meta['positive_labels'],meta['negative_labels'])>=10
 p95=float(np.quantile(ss,.95));n=int(len(y));passed=feasible and real>p95+1e-12 and real>geo+1e-12
 return {'qualified_pairs':n,'images_with_qualified_pairs':int(len(np.unique(iid))),'label1':int(y.sum()),'label0':int(n-y.sum()),'observed_strata':len(unique),'J_counts_0_1_2plus':np.bincount(jr,minlength=3).tolist(),'geometry_J_counts':np.bincount(jg,minlength=3).tolist(),'real_geometry_J_different_pairs':int((jr!=jg).sum()),'real_geometry_joint_tables_identical':bool(eq),'strongJ_exact_bin_movable_pairs':int(exactmove.sum()),'strongJ_exact_movable_and_Ymixed_pairs':int((exactmove&ymixed[groups]).sum()),'strongJ_ever_different_from_real_pairs':int((strong!=jr[None,:]).any(0).sum()),'real_CMI_bits':real,'geometry_CMI_bits':geo,'strong_null_CMI_bits_199':ss,'weak_null_CMI_bits_199':ww,'strong_null95_bits':p95,'weak_null95_bits':float(np.quantile(ww,.95)),'information_feasible':bool(feasible),'advance_gate_pass':bool(passed),**meta}, {'image_id':iid,'strata_code':st,'label':y,'Jreal':jr,'Jgeo':jg,'Jstrong':strong,'Jweak':weak,'exact_strong_bin_mobility':exactmove}
def main():
 start=time.monotonic();assert not O.exists();O.mkdir();phase='inputs';inputs={};checks=Counter();ph=A.sha(HERE/'protocol.json');assert ph==(HERE/'protocol.sha256').read_text('utf-8').strip()
 def track(p):inputs[str(p)]=A.sha(p)
 def budget():
  if time.monotonic()-start>900:raise TimeoutError('Frozen900-second execution budget')
  if sum(p.stat().st_size for p in O.iterdir() if p.is_file())>150*2**20:raise RuntimeError('Frozen150MiB output budget')
 try:
  parent=OLD/'Hlineage_下一最小零训练检验_冻结计划.json';assert A.sha(parent)==read(HERE/'protocol.json')['parent_plan_sha256']
  priorfile=OLD/'matching_and_witness/attempt01/per_image_arm.csv'
  for p in (HERE/'protocol.json',Path(__file__),parent,OLD/'pre_witness_information/run.py',Path(A.__file__),Path(_lsap.__file__),VAL,REP/'complete.json',REP/'preflight.json',priorfile):track(p)
  for n in ('mask.py','_mask.cp310-win_amd64.pyd'):track(A.EVALROOT/'pycocoeval'/n)
  comp=read(REP/'complete.json');pref=read(REP/'preflight.json');declared={r['path']:r['sha256'] for r in comp['entries']+pref['entries']}
  for p,h in comp['source_sha256'].items():
   if p.endswith('dfine_decoder.py') or p.endswith('replay.py'):track(p);assert inputs[p]==h
  with open(priorfile,encoding='utf-8-sig') as f:prior={(r['state'],int(r['seed']),int(r['image_id'])):r for r in csv.DictReader(f) if r['arm']=='A' and float(r['IoU'])==.5}
  ds=read(VAL);ims=sorted(ds['images'],key=lambda x:x['id']);by=defaultdict(list)
  for a in ds['annotations']:by[a['image_id']].append(a)
  assert len(ims)==548 and len(ds['annotations'])==38759
  summary={};imgrows=[];classrows=[]
  with gzip.open(O/'candidate_features.csv.gz','xt',encoding='utf-8',newline='') as f:
   fields=['state','seed','image_id','domain','flat_pair','qualified','true_marginal','Npre','Nfinal','Jreal','Jgeo','strong_ever_moves_J','exact_strong_bin_mobility','Jstrong_lower','Jstrong_upper','stratum_code'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
   for si,state in enumerate(('raw','EMA')):
    for seed in (1,2):
     data={'delete':[],'insert':[]};counts={'delete':Counter(),'insert':Counter()};totalGT=0
     for ii,im in enumerate(ims):
      budget();iid=im['id'];phase=f'{state} seed{seed} image{iid}';p=REP/f'seed{seed}_{state}/image_{iid:06d}.npz';track(p);assert inputs[str(p)]==declared[str(p)]
      with np.load(p,allow_pickle=False) as z:pb=torch.from_numpy(z['final_boxes']);lg=torch.from_numpy(z['final_logits']);preb=torch.from_numpy(z['pre_boxes']);prelg=torch.from_numpy(z['pre_logits'])
      s=lg.sigmoid();ps=prelg.sigmoid();sv=s.numpy();pv=ps.numpy();arg=s.argmax(1).numpy();parg=ps.argmax(1).numpy();av,af=s.flatten().topk(300);af=af[av>.001].numpy();assert len(af)==300
      U=np.unique(af//10);bv,bi=s[torch.from_numpy(U)].flatten().topk(600);bf=U[bi.numpy()//10]*10+bi.numpy()%10;assert (bv>.001).all() and set(af)<=set(bf)
      capped=af[A.cap_indices(af%10)];ac=Counter((af%10).tolist());extra=np.asarray([x for x in bf if x not in set(af.tolist())]);assert len(extra)==300
      fbox=F.boxes(pb,im);pbox=F.boxes(preb,im);fp=F.iou(fbox,pbox);ff=F.iou(fbox,fbox);assert np.isfinite(fp).all() and (fp>=0).all() and (fp<=1).all()
      aa=by[iid];totalGT+=len(aa);gl=np.asarray([a['category_id'] for a in aa]);g=np.asarray([a['bbox'] for a in aa],dtype=np.float64).reshape(-1,4);M=F.iou(fbox,g)
      base={};baseflat={};baseclass={}
      for c in range(10):
       cc=capped[capped%10==c];mat=M[cc//10][:,gl==c]>=.50;base[c]=mat;baseflat[c]=cc;baseclass[c]=A.maxmatch(mat)
      rank=sum(baseclass.values());assert rank==int(prior[state,seed,iid]['same_cap_pool_max']) and len(capped)==int(prior[state,seed,iid]['retained_cap100']) and rank<=300;checks['nativeA_cap_rank_exact']+=1
      domainflats={'delete':np.asarray([x for x in capped if ac[x%10]<=99],dtype=np.int64),'insert':np.asarray([x for x in extra if ac[x%10]<=99],dtype=np.int64)}
      flats=np.concatenate(list(domainflats.values()));q=flats//10;c=flats%10;n=len(flats);domain=np.asarray(['delete']*len(domainflats['delete'])+['insert']*len(domainflats['insert']))
      pre=(fp[q]>=.50)&(parg[None,:]==c[:,None]);fin=(ff[q]>=.50)&(arg[None,:]==c[:,None]);pre[np.arange(n),q]=False;fin[np.arange(n),q]=False
      npre=pre.sum(1);nfin=fin.sum(1);jr=np.minimum(2,(pre&fin).sum(1)).astype(np.uint8)
      cx=np.clip(np.floor(pb[:,0].numpy()*4).astype(int),0,3);cy=np.clip(np.floor(pb[:,1].numpy()*4).astype(int),0,3);conf=np.searchsorted([.1,.5],sv.max(1),side='right');block=(arg*16+cy*4+cx)*3+conf;blocks=[np.flatnonzero(block==v) for v in np.unique(block)]
      low=np.zeros(n,dtype=int);up=np.zeros(n,dtype=int)
      for b in blocks:
       nb=len(b)-np.isin(q,b).astype(int);aa0=pre[:,b].sum(1);bb0=fin[:,b].sum(1);low+=np.maximum(0,aa0+bb0-nb);up+=np.minimum(aa0,bb0)
      lowbin=np.minimum(2,low);upbin=np.minimum(2,up);exactmove=lowbin!=upbin
      sp=permutations(seed,si,iid,blocks,1);wp=permutations(seed,si,iid,[np.arange(300)],0)
      js=perm_j(pre,fin,q,sp);jw=perm_j(pre,fin,q,wp)
      assert (js>=lowbin[None,:]).all() and (js<=upbin[None,:]).all() and (js[:,~exactmove]==jr[None,~exactmove]).all();checks['exact_strong_bound_checks']+=n
      cost=301*(parg[:,None]!=arg[None,:])+(1-fp.T);ri,ci=linear_sum_assignment(cost);assert np.array_equal(ri,np.arange(300));geo=ci;jg=perm_j(pre,fin,q,np.asarray([geo]))[0]
      checks['geo_query_ID_different']+=int((geo!=np.arange(300)).sum());checks['geo_query_ID_total']+=300
      qualified=((M[q]>=.50)&(c[:,None]==gl[None,:])).any(1);y=np.zeros(n,dtype=np.uint8)
      for j in np.flatnonzero(qualified):
       c0=int(c[j]);flat=int(flats[j]);mat=base[c0];old=baseclass[c0]
       if domain[j]=='delete':
        k=int(np.flatnonzero(baseflat[c0]==flat)[0]);tmp=np.delete(mat,k,0);r=A.maxmatch(tmp);delta=rank-min(300,rank-old+r)
       else:
        row=M[q[j],gl==c0]>=.50;r=A.maxmatch(np.vstack((mat,row)));delta=min(300,rank-old+r)-rank
       assert delta in (0,1);y[j]=delta;checks['true_marginal_match_checks']+=1
      maxp=sv.max(1);top=np.sort(sv,axis=1)[:,-2:];margin=top[:,1]-top[:,0];soft=torch.softmax(lg,1);certainty=(1+(soft*soft.log()).sum(1)/np.log(10)).numpy()
      strata=c.copy().astype(np.int64)
      for value,baseN in ((scorebin(sv[q,c]),9),(scorebin(.5*(pv[q,c]+sv[q,c])),9),(scorebin(np.maximum(pv[q,c],sv[q,c])),9),(degbin(npre),5),(degbin(nfin),5),(scorebin(maxp[q]),9),(np.searchsorted([.01,.03,.1,.3],margin[q],side='right'),5),(np.searchsorted([.25,.5,.75],certainty[q],side='right'),4)):strata=strata*baseN+value
      movable=np.zeros(300,dtype=bool)
      for b in blocks:
       if len(b)>1:movable[b]=True
      for dom in ('delete','insert'):
       keep=domain==dom;qual=keep&qualified;z=counts[dom];z['candidate_pairs']+=int(keep.sum());z['qualified_pairs']+=int(qual.sum());z['marginal_positive']+=int(y[keep].sum());z['images_with_candidates']+=int(keep.any());z['images_with_qualified']+=int(qual.any());z['both_neighbor_nonempty_qualified']+=int((qual&(npre>0)&(nfin>0)).sum());z['realJ_nonzero_qualified']+=int((qual&(jr>0)).sum());z['strongJ_ever_changes_qualified']+=int((qual&(js!=jr[None,:]).any(0)).sum())
       data[dom].append({'iid':np.full(qual.sum(),iid,dtype=np.int32),'strata':strata[qual],'y':y[qual],'real':jr[qual],'geo':jg[qual],'strong':js[:,qual],'weak':jw[:,qual],'exactmove':exactmove[qual]})
       imgrows.append({'state':state,'seed':seed,'image_id':iid,'domain':dom,'GT':len(aa),'candidate_pairs':int(keep.sum()),'qualified_pairs':int(qual.sum()),'positive_marginal':int(y[keep].sum()),'both_neighborhoods_nonempty_qualified':int((qual&(npre>0)&(nfin>0)).sum()),'Jnonzero_qualified':int((qual&(jr>0)).sum()),'strongJ_ever_changes_qualified':int((qual&(js!=jr[None,:]).any(0)).sum()),'strong_movable_query_IDs':int(movable.sum()),'strong_singleton_blocks':sum(len(b)==1 for b in blocks),'geo_diff_query_IDs':int((geo!=np.arange(300)).sum())})
      for j,flat in enumerate(flats):writer.writerow(dict(zip(fields,[state,seed,iid,domain[j],int(flat),int(qualified[j]),int(y[j]),int(npre[j]),int(nfin[j]),int(jr[j]),int(jg[j]),int((js[:,j]!=jr[j]).any()),int(exactmove[j]),int(low[j]),int(up[j]),int(strata[j])])))
      if (ii+1)%128==0:print(phase,'done',ii+1,'elapsed',time.monotonic()-start,flush=True)
     assert totalGT==38759
     for dom in ('delete','insert'):
      budget();stats,arrays=summarize(data[dom]);assert stats['qualified_pairs']==counts[dom]['qualified_pairs'];key=f'{state}_s{seed}_{dom}';summary[key]={'full_counts':dict(counts[dom]),'primary_qualified':stats}
      with open(O/f'{key}_primary_arrays.npz','xb') as f2:np.savez_compressed(f2,**arrays)
      print(key,'real',stats['real_CMI_bits'],'geo',stats['geometry_CMI_bits'],'strong95',stats['strong_null95_bits'],'informative',stats['informative_pairs'],stats['informative_images'],'gate',stats['advance_gate_pass'],'elapsed',time.monotonic()-start,flush=True)
  budget();assert checks['nativeA_cap_rank_exact']==2192 and all(A.sha(p)==h for p,h in inputs.items());csvout('per_image.csv',imgrows)
  passed=all(z['primary_qualified']['advance_gate_pass'] for z in summary.values())
  write('results.json',{'status':'COMPLETE','protocol_sha256':ph,'parent_plan_sha256':A.sha(parent),'conditions':summary,'checks':dict(checks),'strict_information_advance_gate_pass':passed,'decision':'information gate only passes; novelty/actual intervention separate' if passed else 'frozen Hlineage does not advance; report failed comparisons versus unidentifiable strata separately','runtime':{'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__},'operations':{'model_forwards':0,'head_calls':0,'checkpoint_loads':0,'evaluator_calls':0,'training':0},'elapsed_seconds':time.monotonic()-start})
  write('input_manifest.json',inputs);write('verification.json',{'status':'PASS','all2192_source_NPZ_SHA_bound':True,'all2192_nativeA_cap_rank_exact':True,'all0or1_marginal_labels':True,'anchor_fixed_permutation_and_set_freeze':True,'unchanged_inputs':len(inputs),'protocol_unchanged':A.sha(HERE/'protocol.json')==ph,'output_MiB':sum(p.stat().st_size for p in O.rglob('*') if p.is_file())/2**20})
 except Exception as e:
  write('failure.json',{'status':'FAILED_PRESERVED','phase':phase,'error':repr(e),'traceback':traceback.format_exc(),'protocol_sha256':ph,'elapsed_seconds':time.monotonic()-start});raise
if __name__=='__main__':main()
