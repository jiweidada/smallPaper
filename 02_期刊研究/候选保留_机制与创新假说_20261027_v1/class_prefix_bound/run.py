"""Certificate/ILP oracle for fixed-pool class-prefix capacity; no learning."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import sys,json,csv,gzip,time,warnings,importlib.util,traceback
from pathlib import Path
from collections import defaultdict,Counter
sys.dont_write_bytecode=True
import numpy as np
import torch
from scipy.sparse import csr_matrix,coo_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.optimize import milp,Bounds,LinearConstraint
HERE=Path(__file__).resolve().parent;O=HERE/'attempt01';ROOT=HERE.parent.parent;PAIR=ROOT/'raw_EMA逐GT对照_20261026_v1';REP=PAIR/'同后端CPU重放_20261026_v1'
VAL=Path(r'C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val/annotations/val.json')
FREE=HERE.parent/'fixed_budget_capacity/attempt01'
spec=importlib.util.spec_from_file_location('prior_audit',ROOT/'问题优先诊断_20261026_v1/audit.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A);torch.set_num_threads(2)
from pycocoeval import mask as maskUtils
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(n,j):
 with open(O/n,'x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False)
def closure(edges,s):
 out=set()
 for flat,g in edges:
  q,c=divmod(flat,10);out.add(flat)
  out.update(q*10+int(k) for k in np.flatnonzero(s[q]>s[q,c]))
 return out
def validate(edges,sel,s,M,gl,th,limit=True):
 assert len({g for f,g in edges})==len(edges) and len({f for f,g in edges})==len(edges)
 assert all(f in sel and s[f//10,f%10]>.001 and gl[g]==f%10 and M[f//10,g]>=th for f,g in edges)
 assert all(v<=100 for v in Counter(f%10 for f,g in edges).values())
 assert closure([(f,-1) for f in sel],s)<=sel
 if limit:assert len(sel)<=300
def solve(targets,s,M,gl,th,free,deadline):
 all_edges=[(f,g) for f,gg in targets.items() for g in gg];ys=sorted(closure(all_edges,s));yi={f:j for j,f in enumerate(ys)};ny=len(ys);ne=len(all_edges)
 if not ne:return [],set(),{'status':0,'kind':'empty'},0
 ri=[];ci=[];vv=[];ub=[]
 def add(items,upper):
  r=len(ub);ub.append(upper)
  for i,v in items:ri.append(r);ci.append(i);vv.append(v)
 bygt=defaultdict(list);bypair=defaultdict(list);byclass=defaultdict(list)
 for j,(f,g) in enumerate(all_edges):bygt[g].append(ny+j);bypair[f].append(ny+j);byclass[f%10].append(ny+j)
 for js in bygt.values():add([(j,1) for j in js],1)
 for f,js in bypair.items():add([(j,1) for j in js]+[(yi[f],-1)],0)
 for js in byclass.values():add([(j,1) for j in js],100)
 for f in ys:
  q,c=divmod(f,10)
  for k in np.flatnonzero(s[q]>s[q,c]):add([(yi[f],1),(yi[q*10+int(k)],-1)],0)
 add([(j,1) for j in range(ny)],300)
 c=np.r_[np.zeros(ny),-np.ones(ne)];mat=coo_matrix((vv,(ri,ci)),shape=(len(ub),ny+ne)).tocsc()
 sec=max(.05,min(3.,deadline-time.monotonic()))
 with warnings.catch_warnings(record=True) as ww:
  warnings.simplefilter('always');res=milp(c,integrality=np.ones(ny+ne),bounds=Bounds(np.zeros(ny+ne),np.ones(ny+ne)),constraints=LinearConstraint(mat,-np.inf,np.asarray(ub)),options={'time_limit':sec,'mip_rel_gap':0.,'presolve':True,'threads':2})
 meta={'status':int(res.status),'message':str(res.message),'warnings':[str(w.message) for w in ww],'variables':ny+ne,'constraints':len(ub),'seconds_limit':sec}
 edges=[];sel=set()
 if res.x is not None:
  assert np.max(np.abs(res.x-np.rint(res.x)))<1e-5
  z=np.rint(res.x).astype(int);sel={f for f,j in yi.items() if z[j]};edges=[e for j,e in enumerate(all_edges) if z[ny+j]];validate(edges,sel,s,M,gl,th)
 dual=getattr(res,'mip_dual_bound',None);upper=free
 if dual is not None and np.isfinite(dual):meta['minimize_dual_bound']=float(dual);upper=min(free,int(np.floor(-dual+1e-5)))
 if res.status==0:upper=len(edges)
 assert len(edges)<=upper<=free
 return edges,sel,meta,upper
def main():
 start=time.monotonic();assert not O.exists();O.mkdir();ph=A.sha(HERE/'protocol.json');assert ph==(HERE/'protocol.sha256').read_text('utf-8').strip();inputs={};phase='inputs';solver_count=0
 def track(p):inputs[str(p)]=A.sha(p)
 try:
  for p in (HERE/'protocol.json',Path(__file__),Path(A.__file__),VAL,FREE/'per_image_bounds.csv',FREE/'results.json',REP/'complete.json',REP/'preflight.json'):track(p)
  for n in ('mask.py','_mask.cp310-win_amd64.pyd'):track(A.EVALROOT/'pycocoeval'/n)
  with open(FREE/'per_image_bounds.csv',encoding='utf-8-sig') as f:prior={(r['state'],int(r['seed']),int(r['image_id']),int(round(float(r['IoU'])*100))):r for r in csv.DictReader(f)}
  assert len(prior)==4384;cp=read(REP/'complete.json');pref=read(REP/'preflight.json');declared={r['path']:r['sha256'] for r in cp['entries']+pref['entries']}
  ds=read(VAL);ims=sorted(ds['images'],key=lambda x:x['id']);by=defaultdict(list)
  for a in ds['annotations']:by[a['image_id']].append(a)
  rows=[];tot=defaultdict(Counter);examples=[]
  with gzip.open(O/'certificates.jsonl.gz','xt',encoding='utf-8') as cert:
   for state in ('raw','EMA'):
    for seed in (1,2):
     for im in ims:
      if time.monotonic()-start>600:raise TimeoutError('Frozen600-second budget')
      phase=f'{state} s{seed} image{im["id"]}';p=REP/f'seed{seed}_{state}/image_{im["id"]:06d}.npz';track(p);assert inputs[str(p)]==declared[str(p)]
      with np.load(p,allow_pickle=False) as z:pb=torch.from_numpy(z['final_boxes']);lg=torch.from_numpy(z['final_logits'])
      s=lg.sigmoid().numpy();e=A.edges_norm(pb)*torch.tensor([im['width'],im['height']]*2);xywh=torch.cat((e[:,:2],e[:,2:]-e[:,:2]),-1).numpy().astype(np.float64)
      aa=by[im['id']];g=np.asarray([a['bbox'] for a in aa],dtype=np.float64).reshape(-1,4);gl=np.asarray([a['category_id'] for a in aa]);M=np.asarray(maskUtils.iou(xywh,g,[0]*len(aa)),dtype=np.float64).reshape(300,len(aa))
      for th in (50,75):
       t=th/100;old=prior[state,seed,im['id'],th];free=int(old['full300_oracle']);assert not int(old['total300_binds']);edges=[];targets={}
       for c in range(10):
        gi=np.flatnonzero(gl==c);mask=(M[:,gi]>=t)&(s[:,c,None]>.001)
        match=maximum_bipartite_matching(csr_matrix(mask),perm_type='column');pairs=[(q*10+c,int(gi[j])) for q,j in enumerate(match) if j>=0][:100];edges.extend(pairs)
        for q in np.flatnonzero(mask.any(1)):targets[int(q)*10+c]=[int(gi[j]) for j in np.flatnonzero(mask[q])]
       assert len(edges)==free;sel=closure(edges,s);validate(edges,sel,s,M,gl,t,limit=False);initial_cost=len(sel);upper=free;meta=None
       if initial_cost<=300:status='EQUAL_CONSTRUCTIVE';lower=free
       else:
        partial=[];ps=set()
        for ed in edges:
         candidate=closure(partial+[ed],s)
         if len(candidate)<=300:partial.append(ed);ps=candidate
        lower=max(len(partial),int(old['A_actual']));edges=partial;sel=ps;validate(edges,sel,s,M,gl,t)
        if time.monotonic()-start<540:
         solver_count+=1;ee,ss,meta,upper=solve(targets,s,M,gl,t,free,start+540)
         if len(ee)>len(edges):edges,sel=ee,ss
         lower=max(lower,len(ee));assert lower<=upper
         if lower==free:status='EQUAL_MILP_WITNESS'
         elif upper<free:status='STRICT_RELAXED_GAP' if lower==upper else 'STRICT_GAP_EXACT_UNKNOWN'
         else:status='UNKNOWN_SOLVER_LIMIT'
        else:status='UNKNOWN_BUDGET_RESERVE'
        examples.append({'state':state,'seed':seed,'image_id':im['id'],'IoU':t,'initial_closure_cost':initial_cost,'status':status,'solver':meta,'lower':lower,'upper':upper,'Mfree':free})
       assert 0<=int(old['A_actual'])<=lower<=upper<=free
       row={'state':state,'seed':seed,'image_id':im['id'],'IoU':t,'GT':len(aa),'A_actual':int(old['A_actual']),'Mfree':free,'initial_closure_cost':initial_cost,'prefix_lower':lower,'prefix_upper':upper,'status':status};rows.append(row)
       z=tot[state,seed,th];z['images']+=1;z['GT']+=len(aa);z['Mfree']+=free;z['prefix_lower']+=lower;z['prefix_upper']+=upper;z[status]+=1;z['largest_initial_closure']=max(z['largest_initial_closure'],initial_cost)
       cert.write(json.dumps({'state':state,'seed':seed,'image_id':im['id'],'IoU':t,'status':status,'selected_flat':sorted(sel),'matched_flat_originalGT':[[f,aa[j]['id']] for f,j in edges]},ensure_ascii=False)+'\n')
     print(state,seed,'complete',time.monotonic()-start,flush=True)
  assert len(rows)==4384 and all(z['GT']==38759 and z['images']==548 for z in tot.values())
  with open(O/'per_image.csv','x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  sums={f'{s}_s{seed}_{th}':dict(z) for (s,seed,th),z in tot.items()};all_equal=all(r['prefix_lower']==r['Mfree'] for r in rows)
  write('results.json',{'status':'COMPLETE','protocol_sha256':ph,'conditions':sums,'all_cases_equal_certified':all_equal,'fallback_cases':examples,
   'meaning':'Optimistic fixed-eligible-pool prefix/matching-cap relaxation. Equality refutes this capacity-bound explanation only, not real greedy optimality or full trained-method reachability.',
   'operations':{'solver_calls':solver_count,'evaluator_calls':0,'model_forwards':0,'head_calls':0,'training':0},'elapsed_seconds':time.monotonic()-start})
  assert all(A.sha(p)==h for p,h in inputs.items());write('input_manifest.json',inputs);write('verification.json',{'status':'PASS','all4384_rows_present':True,'Mfree_exactly_reproduced':True,'all_constructed_matching_prefix_certificates_valid':True,'unchanged_inputs':len(inputs),'protocol_unchanged':A.sha(HERE/'protocol.json')==ph})
 except Exception as e:
  write('failure.json',{'status':'FAILED_PRESERVED','phase':phase,'error':repr(e),'traceback':traceback.format_exc(),'solver_calls':solver_count,'protocol_sha256':ph,'elapsed_seconds':time.monotonic()-start});raise
if __name__=='__main__':main()
