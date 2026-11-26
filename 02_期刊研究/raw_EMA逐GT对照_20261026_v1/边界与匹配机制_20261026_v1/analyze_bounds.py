"""Existing matching thresholds supply geometric lower bounds, not EMA box estimates."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,csv,gzip,hashlib,time,collections
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent;PAIR=OUT.parent;ROOT=PAIR.parent
PREV=ROOT/'问题优先诊断_20261026_v1'
BASE=ROOT/'data/00_实验计划与证据索引/本次24轮双基线模型下载_20261009_v1'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def csvread(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(n,j):
 with open(OUT/n,'x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvout(n,rows):
 if not rows:return
 with open(OUT/n,'x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(x):
 a=np.asarray(x,dtype=np.float64)
 return {'n':int(a.size),'mean':float(a.mean()),'quantiles':np.quantile(a,[0,.1,.25,.5,.75,.9,1]).tolist()} if a.size else {'n':0}
def summary(rows):
 known=[r for r in rows if r['IoU_improvement_lower_bound'] is not None]
 return {'n_population':len(rows),'n_bound_known':len(known),'n_bound_unknown':len(rows)-len(known),
  'lower_bound':stats([r['IoU_improvement_lower_bound'] for r in known]),
  'raw_maxIoU':stats([r['raw_maxIoU'] for r in rows]),
  'raw_distance_below50':stats([.5-r['raw_maxIoU'] for r in rows]),
  'h_histogram':dict(collections.Counter(str(r['EMA_highest_matching_threshold']) for r in known)),
  'bound_above_markers':{str(t):sum(r['IoU_improvement_lower_bound']>t for r in known) for t in (0,.01,.05,.1,.25)}}
def main():
 start=time.perf_counter();ph=sha(OUT/'protocol.json');assert ph==(OUT/'protocol.sha256').read_text('utf-8').strip()
 inputs={}
 def track(p):
  p=Path(p);inputs[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size};return inputs[str(p)]['sha256']
 for p in [PREV/'per_GT.csv',PAIR/'attempt02/per_GT_transition.csv',PAIR/'attempt02/results.json',PAIR/'attempt02/verification.json']:
  track(p)
 prev={(int(r['seed']),int(r['annotation_id'])):r for r in csvread(PREV/'per_GT.csv')}
 paired=csvread(PAIR/'attempt02/per_GT_transition.csv');pr=read(PAIR/'attempt02/results.json')
 assert len(prev)==len(paired)==77518
 allrows=[];witnesses=[];checks={};nonmonotonic={}
 for seed in (1,2):
  endpoint=BASE/f'seed{seed}/run/endpoint_epoch24';ep=read(endpoint/'summary.json');mf=read(endpoint/'manifest.json')
  for n in ('manifest.json','summary.json','matching_all_area.json.gz','GT_hits.npz'):track(endpoint/n)
  for n,h in mf['files'].items():assert sha(endpoint/n)==h
  th=np.asarray(ep['protocol']['iou_thresholds'],dtype=np.float64)
  z=np.load(endpoint/'GT_hits.npz',allow_pickle=False);aids=z['annotation_id'].astype(int);native=z['hits'].copy();ai={int(a):j for j,a in enumerate(aids)}
  hmat=np.zeros(native.shape,dtype=bool);native_replay=np.zeros_like(hmat);seen=[];record_for={}
  records=json.load(gzip.open(endpoint/'matching_all_area.json.gz','rt',encoding='utf-8'))
  for e in records:
   assert e['aRng']==ep['protocol']['area_ranges'][0] and e['maxDet']==100 and not any(e['gtIgnore'])
   gt=list(map(int,e['gtIds']));dt=list(map(int,e['dtIds']));dp={d:i for i,d in enumerate(dt)}
   gm=np.asarray(e['gtMatches'],dtype=np.int64).reshape(10,len(gt));dm=np.asarray(e['dtMatches'],dtype=np.int64).reshape(10,len(dt));di=np.asarray(e['dtIgnore'],dtype=bool).reshape(10,len(dt))
   seen.extend(gt);owners=[]
   for t in range(10):
    own={}
    for j,a in enumerate(gt):
     did=int(gm[t,j])
     if did>0:
      assert did in dp and did not in own and not di[t,dp[did]] and int(dm[t,dp[did]])==a
      own[did]=a;hmat[ai[a],t]=True
    for k,a in enumerate(dm[t]):
     if a!=0 and not di[t,k]:native_replay[ai[int(a)],t]=True
    owners.append(own)
   for j,a in enumerate(gt):record_for[a]=(e,j,gm,dp,owners)
  assert len(seen)==38759 and set(seen)==set(aids) and np.array_equal(native_replay,native)
  rr=[r for r in paired if int(r['seed'])==seed]
  for r in rr:
   aid=int(r['annotation_id']);v=prev[seed,aid];hm=hmat[ai[aid]]
   assert int(hm[0])==int(r['EMA_hit50']) and int(hm[5])==int(r['EMA_hit75'])
   inds=np.flatnonzero(hm);h=float(th[inds[-1]]) if len(inds) else None;raw=float(v['raw_max_iou'])
   row={'seed':seed,'image_id':int(r['image_id']),'annotation_id':aid,'category_id':int(r['category_id']),
        'size_group':r['size_group'],'short_side_640':float(r['short_side_640']),'raw_state50':r['raw_state50'],
        'raw_hit50':int(r['raw_hit50']),'EMA_hit50':int(r['EMA_hit50']),'transition50':r['transition50'],
        'raw_maxIoU':raw,'EMA_highest_matching_threshold':h,'IoU_improvement_lower_bound':h-raw if h is not None else None,
        'EMA_threshold_hit_bits':''.join('1' if b else '0' for b in hm),
        'EMA_miss50_but_higher_hit':int(not hm[0] and len(inds)>0)}
   if row['EMA_miss50_but_higher_hit']:
    e,j,gm,dp,owners=record_for[aid];t=int(inds[-1]);did=int(gm[t,j]);owner50=owners[0].get(did)
    witnesses.append({'seed':seed,'image_id':int(e['image_id']),'category_id':int(e['category_id']),
      'annotation_id':aid,'raw_state50':r['raw_state50'],'higher_threshold':float(th[t]),'higher_matched_detection_id':did,
      'detection_score':float(e['dtScores'][dp[did]]),'same_detection_owner_at50':owner50,
      'target_match_at50':int(gm[0,j]),'same_detection_list_score_order_class_and_cap':True,
      'owner_at50_is_other_GT':owner50 is not None and owner50!=aid})
   allrows.append(row)
  event=[w for w in witnesses if w['seed']==seed]
  # Nonmonotonicity may occur at any adjacent thresholds; count separately.
  rises=(~hmat[:,:-1])&hmat[:,1:]
  nonmonotonic[str(seed)]={'miss50_higher_hit_GT':len(event),'denominator_all_GT':38759,
    'among_EMA_miss50':pr['all_GT_primary'][str(seed)]['50']['EMA_miss'],
    'any_adjacent_miss_to_hit_GT':int(rises.any(1).sum()),'any_adjacent_miss_to_hit_events':int(rises.sum()),
    'all_witness_low_threshold_owners_other_GT':all(w['owner_at50_is_other_GT'] for w in event)}
  checks[str(seed)]={'all_GT':len(rr),'native_all10_columns_replayed_exact':True,'primary50and75_equal_paired_results':True}
  print(f'seed{seed}: complete existing-record bounds and matching witnesses',flush=True)
 populations={};groupstats=[]
 for seed in (1,2):
  rr=[r for r in allrows if r['seed']==seed]
  pops={'all_GT':rr,'rawG_all':[r for r in rr if r['raw_state50']=='G'],
   'rawG_EMA_hit50':[r for r in rr if r['raw_state50']=='G' and r['EMA_hit50']],
   'rawG_EMA_miss50':[r for r in rr if r['raw_state50']=='G' and not r['EMA_hit50']],
   'rawK_EMA_hit50':[r for r in rr if r['raw_state50']=='K' and r['EMA_hit50']],
   'rawT_EMA_hit50':[r for r in rr if r['raw_state50']=='T' and r['EMA_hit50']],
   'rawT_EMA_miss50':[r for r in rr if r['raw_state50']=='T' and not r['EMA_hit50']]}
  expected=2247 if seed==1 else 1527;assert len(pops['rawG_EMA_hit50'])==expected
  populations[str(seed)]={name:summary(items) for name,items in pops.items()}
  for name in ('rawG_EMA_hit50','rawK_EMA_hit50','rawT_EMA_hit50'):
   items=pops[name]
   for typ,labels in [('size',['[0,4)','[4,8)','[8,16)','[16,32)','[32,inf)']),('category',list(range(10)))]:
    for label in labels:
     sub=[r for r in items if r['size_group']==label] if typ=='size' else [r for r in items if r['category_id']==label]
     groupstats.append({'seed':seed,'population':name,'group_type':typ,'group':label,**summary(sub)})
 csvout('per_GT_bounds.csv',allrows);csvout('matching_witnesses.csv',witnesses)
 write('results.json',{'populations':populations,'group_statistics':groupstats,'matching_events':nonmonotonic,'matching_witnesses':witnesses,
  'verification':checks,'protocol_sha256':ph,'quantile_order':[0,.1,.25,.5,.75,.9,1],
  'interpretation':'Endpoint existence lower bounds only. Unknown means no bound; no EMA maxIoU upper bound or coordinate-change attribution.',
  'operations':{'model_imports':0,'forwards':0,'evaluator_calls':0,'training_steps':0,'GPU':False,'remote':False,'test_reads':0},
  'elapsed_seconds':time.perf_counter()-start})
 write('input_manifest.json',inputs)
 changed=[p for p,m in inputs.items() if sha(p)!=m['sha256']];assert not changed and sha(OUT/'protocol.json')==ph
 write('verification.json',{'input_files_rehashed':len(inputs),'changed_files':changed,'protocol_unchanged':True,'script_sha256':sha(__file__),'all_GT_rows':len(allrows)})
 print(json.dumps({'primary_G_recovery':{s:populations[s]['rawG_EMA_hit50'] for s in ('1','2')},'matching_events':nonmonotonic},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
