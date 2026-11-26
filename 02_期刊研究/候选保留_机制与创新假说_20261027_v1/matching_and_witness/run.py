"""Same-capped-pool matching and full-graph critical-node checks; no eval/train."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import sys,json,csv,gzip,time,importlib.util,traceback
from pathlib import Path
from collections import defaultdict,Counter
sys.dont_write_bytecode=True
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;O=HERE/'attempt01';ROOT=HERE.parent.parent;PAIR=ROOT/'raw_EMA逐GT对照_20261026_v1';REP=PAIR/'同后端CPU重放_20261026_v1';FREE=HERE.parent/'fixed_budget_capacity/attempt01'
TRANS=PAIR/'rawBC操作地图与EMA迁移_20261026_v1'
VAL=Path(r'C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val/annotations/val.json')
spec=importlib.util.spec_from_file_location('prior_audit',ROOT/'问题优先诊断_20261026_v1/audit.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A);torch.set_num_threads(2)
from pycocoeval import mask as maskUtils
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(n,j):
 with open(O/n,'x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvout(n,rr):
 with open(O/n,'x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def main():
 start=time.monotonic();assert not O.exists();O.mkdir();ph=A.sha(HERE/'protocol.json');assert ph==(HERE/'protocol.sha256').read_text('utf-8').strip();inputs={};phase='inputs';deletions=0
 def track(p):inputs[str(p)]=A.sha(p)
 try:
  raws=PAIR/'后续零训练预检_20261026_v1/B_per_GT.csv';emas=TRANS/'EMA_transfer_v1/per_GT.csv';pairs=TRANS/'raw_operations_v1/pair_operation_map.csv.gz';imageA=REP/'analysis_v1/per_image.csv'
  for p in (HERE/'protocol.json',Path(__file__),Path(A.__file__),VAL,FREE/'per_image_bounds.csv',REP/'complete.json',REP/'preflight.json',raws,emas,pairs,imageA,TRANS/'raw_operations_v1/results.json',TRANS/'EMA_transfer_v1/results.json'):track(p)
  for n in ('mask.py','_mask.cp310-win_amd64.pyd'):track(A.EVALROOT/'pycocoeval'/n)
  with open(FREE/'per_image_bounds.csv',encoding='utf-8-sig') as f:freeidx={(r['state'],int(r['seed']),int(r['image_id']),int(round(float(r['IoU'])*100))):r for r in csv.DictReader(f)}
  with open(imageA,encoding='utf-8-sig') as f:Acounts={(r['branch'],int(r['seed']),int(r['image_id'])):r for r in csv.DictReader(f)}
  allGT={}
  for state,path in [('raw',raws),('EMA',emas)]:
   with open(path,encoding='utf-8-sig') as f:rr=list(csv.DictReader(f))
   assert len(rr)==77518
   for r in rr:allGT[state,int(r['seed']),int(r['annotation_id'])]=r
  actualpair={}
  with gzip.open(pairs,'rt',encoding='utf-8-sig') as f:
   for r in csv.DictReader(f):
    seed=int(r['seed']);iid=int(r['image_id']);flat=int(r['query'])*10+int(r['class'])
    for arm in ('B','C'):
     for th in (50,75):
      aid=int(r[f'{arm}_match_GT{th}'])
      if aid>=0:
       k=seed,arm,th,aid;assert k not in actualpair;actualpair[k]=(iid,flat)
  prior={'raw':read(TRANS/'raw_operations_v1/results.json'),'EMA':read(TRANS/'EMA_transfer_v1/results.json')}
  pref=read(REP/'preflight.json');comp=read(REP/'complete.json');declared={r['path']:r['sha256'] for r in pref['entries']+comp['entries']}
  ds=read(VAL);by=defaultdict(list)
  for a in ds['annotations']:by[a['image_id']].append(a)
  ims=sorted(ds['images'],key=lambda x:x['id']);rows=[];classrows=[];summary=defaultdict(Counter);witness=defaultdict(Counter);sourcecounts=defaultdict(Counter)
  for state in ('raw','EMA'):
   for seed in (1,2):
    for im in ims:
     if time.monotonic()-start>600:raise TimeoutError('Frozen600-second budget')
     iid=im['id'];phase=f'{state} seed{seed} image{iid}';p=REP/f'seed{seed}_{state}/image_{iid:06d}.npz';track(p);assert inputs[str(p)]==declared[str(p)]
     with np.load(p,allow_pickle=False) as z:pb=torch.from_numpy(z['final_boxes']);lg=torch.from_numpy(z['final_logits'])
     s=lg.sigmoid();av,af=s.flatten().topk(300);af=af[av>.001];U=torch.unique(af//10,sorted=True);bv,bf=s[U].flatten().topk(600);cv,cf=s.flatten().topk(600)
     assert (bv>.001).all() and (cv>.001).all();flats={'A':af.numpy(),'B':(U[bf//10]*10+bf%10).numpy(),'C':cf.numpy()};exports={a:set(v.tolist()) for a,v in flats.items()};caps={}
     for arm,f in flats.items():
      ix=A.cap_indices(f%10);caps[arm]=f[ix];assert (pb[caps[arm]//10,2:]>0).all()
      sourcecounts[state,seed,arm]['exported']+=len(f);sourcecounts[state,seed,arm]['evaluated']+=len(ix)
     assert len(flats['A'])==int(Acounts[state,seed,iid]['exported_pairs']) and len(caps['A'])==int(Acounts[state,seed,iid]['cap_pairs'])
     e=A.edges_norm(pb)*torch.tensor([im['width'],im['height']]*2);xywh=torch.cat((e[:,:2],e[:,2:]-e[:,:2]),-1).numpy().astype(np.float64)
     aa=by[iid];g=np.asarray([a['bbox'] for a in aa],dtype=np.float64).reshape(-1,4);gl=np.asarray([a['category_id'] for a in aa]);M=np.asarray(maskUtils.iou(xywh,g,[0]*len(aa)),dtype=np.float64).reshape(300,len(aa))
     for th in (50,75):
      crit=set();full=0;local_info=[]
      for c in range(10):
       gi=np.flatnonzero(gl==c);mask=(M[:,gi]>=th/100)&(s[:,c,None].numpy()>.001);m=A.maxmatch(mask);mc=min(100,m);full+=mc;c_crit=[]
       if 0<m<=100:
        for q in np.flatnonzero(mask.any(1)):
         original=mask[q].copy();mask[q]=False;mr=A.maxmatch(mask);mask[q]=original;deletions+=1;assert m-1<=mr<=m
         if mr<m:crit.add(int(q)*10+c);c_crit.append(int(q)*10+c)
       local_info.append({'state':state,'seed':seed,'image_id':iid,'IoU':th/100,'class':c,'GT':len(gi),'full_class_rank':m,'capped_class_rank':mc,'critical_nodes':len(c_crit)})
      old=freeidx[state,seed,iid,th];assert full==int(old['full300_oracle']) and full<=300
      flags={arm:np.asarray([int(allGT[state,seed,a['id']][f'{arm}{th}']) for a in aa],dtype=bool) for arm in ('A','B','C')}
      for arm,f in caps.items():
       q=f//10;cl=f%10;mask=(M[q]>=th/100)&(cl[:,None]==gl[None,:]);bound=A.maxmatch(mask);actual=int(flags[arm].sum());missing_export=len(crit-exports[arm]);missing_cap=len(crit-set(f.tolist()))
       assert 0<=actual<=bound<=full and missing_export<=missing_cap<=full-bound
       mm=int(((~flags[arm])&mask.any(0)).sum());gap=bound-actual;assert gap<=mm
       r={'state':state,'seed':seed,'image_id':iid,'IoU':th/100,'arm':arm,'GT':len(aa),'exported':len(flats[arm]),'retained_cap100':len(f),'actual_TP':actual,'same_cap_pool_max':bound,'greedy_cardinality_gap':gap,'unmatched_GT_with_qual_cap_pair':mm,'full_pool_bound':full,'full_critical_pair_nodes':len(crit),'critical_missing_export':missing_export,'critical_missing_cap':missing_cap,'capacity_loss_not_individual_critical':full-bound-missing_cap};rows.append(r)
       z=summary[state,seed,th,arm]
       for k in ('GT','actual_TP','same_cap_pool_max','greedy_cardinality_gap','unmatched_GT_with_qual_cap_pair','full_pool_bound','full_critical_pair_nodes','critical_missing_export','critical_missing_cap','capacity_loss_not_individual_critical'):z[k]+=r[k]
       z['images']+=1
       for info in local_info:info[arm+'_critical_kept_cap']=sum(f0 in set(f.tolist()) for f0 in crit if f0%10==info['class'])
      classrows.extend(local_info)
      if state=='raw':
       nw=np.flatnonzero(~flags['B']&flags['C']);ls=np.flatnonzero(flags['B']&~flags['C']);common=np.flatnonzero(flags['B']&flags['C']);z=witness[seed,th]
       for label,js,arm in [('C_new',nw,'C'),('C_lost',ls,'B'),('common_C',common,'C'),('common_B',common,'B')]:
        for j in js:
         aid=aa[j]['id'];mid,flat=actualpair[seed,arm,th,aid];assert mid==iid and flat in set(caps[arm].tolist());z[label+'_GT']+=1;z[label+'_critical_matched_pair']+=int(flat in crit)
    print(state,seed,'done',time.monotonic()-start,flush=True)
  for state in ('raw','EMA'):
   for seed in (1,2):
    for arm in ('B','C'):
     old=prior[state]['seeds'][str(seed)][arm] if state=='raw' else prior[state]['seeds'][str(seed)]['arms'][arm]
     assert sourcecounts[state,seed,arm]['exported']==old['exported'] and sourcecounts[state,seed,arm]['evaluated']==old['evaluated']
     assert summary[state,seed,50,arm]['actual_TP']==old['TP50'] and summary[state,seed,75,arm]['actual_TP']==old['TP75']
  assert len(rows)==13152 and all(z['GT']==38759 and z['images']==548 for z in summary.values())
  csvout('per_image_arm.csv',rows);csvout('per_image_class.csv',classrows)
  write('results.json',{'status':'COMPLETE','protocol_sha256':ph,'conditions':{f'{st}_s{s}_{th}_{a}':dict(z) for (st,s,th,a),z in summary.items()},'raw_actual_pair_witnesses':{f's{s}_{th}':dict(z) for (s,th),z in witness.items()},'single_node_deletion_checks':deletions,
   'meaning':'Fixed-cap matching gaps and GT-graph single-node criticality; group bottlenecks remain separately possible. No AP gains or GT-free learned selector claimed.',
   'operations':{'evaluator_calls':0,'model_forwards':0,'head_calls':0,'training':0},'elapsed_seconds':time.monotonic()-start})
  assert all(A.sha(p)==h for p,h in inputs.items());write('input_manifest.json',inputs);write('verification.json',{'status':'PASS','actual_counts_and_export_cap_counts_exact':True,'all_graph_bound_order_and_critical_loss_invariants':True,'all13152_rows_present':True,'unchanged_inputs':len(inputs),'protocol_unchanged':A.sha(HERE/'protocol.json')==ph})
 except Exception as e:
  write('failure.json',{'status':'FAILED_PRESERVED','phase':phase,'error':repr(e),'traceback':traceback.format_exc(),'protocol_sha256':ph,'elapsed_seconds':time.monotonic()-start});raise
if __name__=='__main__':main()
