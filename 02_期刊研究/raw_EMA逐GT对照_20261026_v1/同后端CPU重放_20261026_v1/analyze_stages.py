"""Analyze the completed passive CPU replay; no model construction or forward."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import sys,json,csv,gzip,hashlib,time,importlib.util
from pathlib import Path
from collections import Counter,defaultdict
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
REPLAY=Path(__file__).resolve().parent;PAIR=REPLAY.parent;ROOT=PAIR.parent
OUT=REPLAY/'analysis_v1'
VAL=Path(r'C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val')
BASE=ROOT/'data/00_实验计划与证据索引/本次24轮双基线模型下载_20261009_v1'
spec=importlib.util.spec_from_file_location('frozen_audit',ROOT/'问题优先诊断_20261026_v1/audit.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
torch.set_num_threads(2)
STAGES=('encoder_initial_boxes','pre_boxes','final_boxes')
NAMES=('initial','pre','final')

def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(n,x):
 with open(OUT/n,'x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvout(n,rr):
 with open(OUT/n,'x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def groups(meta):
 d=int(meta['density32']);n=int(meta['image_n_gt']);area=float(meta['area_original'])
 return [('all','all'),('size',meta['size_group']),('class',meta['category_id']),
 ('area','small<1024' if area<1024 else 'medium[1024,9216)' if area<9216 else 'large>=9216'),
 ('density32','0' if d==0 else '1-4' if d<5 else '5-14' if d<15 else '15+'),
 ('image_GT','<=50' if n<=50 else '51-100' if n<=100 else '101-200' if n<=200 else '>200'),
 ('occlusion',meta['occlusion']),('truncation',meta['truncation'])]

def main():
 start=time.time();assert not OUT.exists();OUT.mkdir();A.OUT=OUT
 ph=A.sha(REPLAY/'analysis_protocol.json');assert ph==(REPLAY/'analysis_protocol.sha256').read_text('utf-8').strip()
 complete=read(REPLAY/'complete.json');preflight=read(REPLAY/'preflight.json')
 assert complete['status']=='COMPLETE_READONLY_CPU_REPLAY' and complete['total_global_forwards']==2192
 assert complete['model_parameter_updates']==0 and not complete['training'] and not complete['GPU']
 allentries=preflight['entries']+complete['entries'];assert len(allentries)==2192
 declared={r['path']:r['sha256'] for r in allentries};assert len(declared)==2192
 assert all(b['state_sha256_before']==b['state_sha256_after'] and b['all_grad_none'] and b['all_eval_CPU'] for b in preflight['branches']+complete['branches'])
 assert all(b['raw_max_abs_box']==b['raw_max_abs_logits']==0 for b in complete['branches'] if b['branch']=='raw')
 sources={}
 def track(p):
  p=Path(p);h=A.sha(p);sources[str(p)]=h;return h
 for p in (REPLAY/'analysis_protocol.json',REPLAY/'protocol.json',REPLAY/'replay.py',Path(__file__),REPLAY/'complete.json',REPLAY/'preflight.json',Path(A.__file__)):track(p)
 for p,h in {**complete['source_sha256'],**complete['input_sha256'],**preflight['input_sha256']}.items():assert track(p)==h
 ds=read(VAL/'annotations/val.json');track(VAL/'annotations/val.json')
 ims=sorted(ds['images'],key=lambda r:r['id']);anns=sorted(ds['annotations'],key=lambda r:r['id']);aids=[a['id'] for a in anns];ai={a:i for i,a in enumerate(aids)}
 by=defaultdict(list)
 for a in ds['annotations']:by[a['image_id']].append(a)
 assert len(ims)==548 and len(anns)==38759
 oldraw=list(csv.DictReader(open(ROOT/'问题优先诊断_20261026_v1/per_GT.csv',encoding='utf-8-sig')))
 track(ROOT/'问题优先诊断_20261026_v1/per_GT.csv')
 oldpair=list(csv.DictReader(open(PAIR/'attempt02/per_GT_transition.csv',encoding='utf-8-sig')));track(PAIR/'attempt02/per_GT_transition.csv')
 meta={(int(r['seed']),int(r['annotation_id'])):r for r in oldpair}
 oldrawidx={(int(r['seed']),int(r['annotation_id'])):r for r in oldraw}
 oldresults=read(ROOT/'问题优先诊断_20261026_v1/results.json');track(ROOT/'问题优先诊断_20261026_v1/results.json')
 records={};hits={};metrics={};summaries={};groupmap={};perimage=[];tokenids={};oldEMA={};oldmetrics={};verify={}
 for seed in (1,2):
  ep=BASE/f'seed{seed}/run/endpoint_epoch24';oldmetrics[seed]=read(ep/'summary.json')['native_stats'];track(ep/'summary.json')
  track(ep/'matching_all_area.json.gz');h0=np.zeros((len(anns),10),bool)
  with gzip.open(ep/'matching_all_area.json.gz','rt',encoding='utf-8') as f:captured=json.load(f)
  for e in captured:
   gm=np.asarray(e['gtMatches'],dtype=np.int64).reshape(10,len(e['gtIds']))
   assert not any(e['gtIgnore'])
   for j,aid in enumerate(e['gtIds']):h0[ai[aid]]=gm[:,j]>0
  oldEMA[seed]=h0
  for branch in ('raw','EMA'):
   rec=[];pred=[];invalid=Counter();imagecounts={}
   for pos,im in enumerate(ims):
    iid=im['id'];p=REPLAY/f'seed{seed}_{branch}/image_{iid:06d}.npz'
    assert track(p)==declared[str(p)]
    with np.load(p,allow_pickle=False) as z:
     assert int(z['image_id'])==iid and int(z['seed'])==seed and str(z['branch'])==branch
     boxes=[z[k] for k in STAGES];lg=z['final_logits'];tokenids[seed,branch,iid]=z['encoder_index']
    assert len(set(tokenids[seed,branch,iid]))==300
    aa=by[iid];g=np.asarray([a['bbox'] for a in aa],dtype=np.float64).reshape(-1,4);g[:,2:]+=g[:,:2]
    gl=np.asarray([a['category_id'] for a in aa],dtype=np.int64)
    edges=[(A.edges_norm(torch.from_numpy(b))*torch.tensor([im['width'],im['height']]*2)).numpy() for b in boxes]
    MM=[A.iou_np(b,g) for b in edges];maxs=[m.max(0) for m in MM];best=[m.argmax(0) for m in MM]
    for name,b in zip(NAMES,boxes):invalid[name]+=int((b[:,2:]<=0).any(-1).sum())
    scores=torch.from_numpy(lg).sigmoid();v,flat=scores.flatten().topk(300);keep=v>.001;v=v[keep];flat=flat[keep];qs=(flat//10).numpy();cl=(flat%10).numpy()
    cap=A.cap_indices(cl);m=MM[-1];same=scores.numpy()[:,gl]
    fm=np.where(same>.001,m,0).max(0);tm=np.where(cl[:,None]==gl[None,:],m[qs],0).max(0)
    cm=np.where(cl[cap,None]==gl[None,:],m[qs[cap]],0).max(0)
    invalid['exported_final_pairs']+=int((boxes[-1][qs,2:]<=0).any(-1).sum())
    for q,c,s in zip(qs,cl,v.tolist()):
     b=torch.tensor(edges[-1][q]);wh=torch.cat((b[:2],b[2:]-b[:2])).tolist()
     pred.append({'image_id':iid,'category_id':int(c),'bbox':wh,'score':s})
    imagecounts[iid]={'seed':seed,'branch':branch,'image_id':iid,'GT':len(aa),'exported_pairs':len(qs),'cap_pairs':len(cap)}
    for j,a in enumerate(aa):
     aid=a['id'];r={'seed':seed,'branch':branch,'image_id':iid,'annotation_id':aid,'category_id':a['category_id']}
     for name,mx,bestq in zip(NAMES,maxs,best):r[name+'_maxIoU']=float(mx[j]);r[name+'_best_query']=int(bestq[j])
     r['initial_best_query_final_IoU']=float(m[best[0][j],j]);r['pre_best_query_final_IoU']=float(m[best[1][j],j])
     for th in (50,75):
      t=th/100;r['geometry_bits'+str(th)]=''.join(str(int(mx[j]>=t)) for mx in maxs)
      r['state'+str(th)]='G' if maxs[-1][j]<t else 'S' if fm[j]<t else 'K' if tm[j]<t else 'C' if cm[j]<t else 'pending'
     if branch=='raw':assert r['final_maxIoU']==float(oldrawidx[seed,aid]['raw_max_iou'])
     rec.append(r)
    if (pos+1)%180==0:print(f'seed{seed} {branch} geometric analysis {pos+1}/548',flush=True)
   ev,metric=A.evaluate(ds,pred,f'seed{seed}_{branch}_native_originalIDs')
   if branch=='raw':assert max(abs(metric[k]-oldresults['metrics'][str(seed)]['stored_id_native300'][k]) for k in A.METRICS)<1e-10
   hh=np.zeros((len(anns),10),bool);seen=[]
   for e in ev.evalImgs:
    if e is None or tuple(e['aRng'])!=tuple(ev.params.areaRng[0]):continue
    assert not np.asarray(e['gtIgnore']).any();gm=e['gtMatches'];dpos={int(d):j for j,d in enumerate(e['dtIds'])}
    for j,aid in enumerate(e['gtIds']):
     seen.append(aid);hh[ai[aid]]=gm[:,j]>0
     for t,did in enumerate(gm[:,j]):
      if did>0:assert e['dtMatches'][t,dpos[int(did)]]==aid and not e['dtIgnore'][t,dpos[int(did)]]
   assert len(seen)==len(anns) and set(seen)==set(aids)
   group=defaultdict(Counter)
   for r in rec:
    aid=r['annotation_id'];md=meta[seed,aid]
    for th,t in ((50,0),(75,5)):
     hit=int(hh[ai[aid],t]);r['hit'+str(th)]=hit
     if r['state'+str(th)]=='pending':r['state'+str(th)]='T' if hit else 'M'
     assert not hit or r['state'+str(th)]=='T'
     if branch=='raw':assert r['state'+str(th)]==md['raw_state'+str(th)] and hit==int(md['raw_hit'+str(th)])
    for typ,label in groups({**md,'image_n_gt':oldrawidx[seed,aid]['image_n_gt']}):
     c=group[typ,str(label)];c['GT']+=1
     for th in (50,75):
      c[f'state{th}_'+r['state'+str(th)]]+=1;c[f'geometry_bits{th}_'+r['geometry_bits'+str(th)]]+=1
      c[f'initial_best_still_covered_final{th}']+=int(r['initial_best_query_final_IoU']>=th/100)
      c[f'pre_best_still_covered_final{th}']+=int(r['pre_best_query_final_IoU']>=th/100)
     c['initial_final_best_query_different']+=int(r['initial_best_query']!=r['final_best_query'])
     c['pre_final_best_query_different']+=int(r['pre_best_query']!=r['final_best_query'])
   for iid,c in imagecounts.items():
    subset=[a for a in by[iid]]
    c['hit50']=int(sum(hh[ai[a['id']],0] for a in subset));c['hit75']=int(sum(hh[ai[a['id']],5] for a in subset));perimage.append(c)
   records[seed,branch]={r['annotation_id']:r for r in rec};hits[seed,branch]=hh;metrics[seed,branch]=metric
   summaries[f'{seed}_{branch}']={'metrics':metric,'GT_hits_all10':hh.sum(0).tolist(),'invalid_boxes':dict(invalid),'exported_pairs':len(pred),'allGT':dict(group['all','all'])}
   for (typ,label),c in group.items():groupmap[seed,branch,typ,label]=dict(c)
   print(f'seed{seed} {branch} complete: AP={metric["AP"]*100:.6f} TP50={hh[:,0].sum()} TP75={hh[:,5].sum()}',flush=True)
   del rec,pred,ev
 paired=[];transitions={};context={};geometryflow={}
 for seed in (1,2):
  rh=hits[seed,'raw'];eh=hits[seed,'EMA'];oh=oldEMA[seed]
  for th,t in ((50,0),(75,5)):
   transitions[f'{seed}_{th}']={
    'GT':len(anns),'raw_hit':int(rh[:,t].sum()),'EMA_hit':int(eh[:,t].sum()),
    'recovered':int((~rh[:,t]&eh[:,t]).sum()),'persistent_miss':int((~rh[:,t]&~eh[:,t]).sum()),
    'reverse_loss':int((rh[:,t]&~eh[:,t]).sum()),'both_hit':int((rh[:,t]&eh[:,t]).sum())}
  context[str(seed)]={'allGT':len(anns),'old_EMA_hit_all10':oh.sum(0).tolist(),'CPU_EMA_hit_all10':eh.sum(0).tolist(),
   'old_miss_CPU_hit_all10':(~oh&eh).sum(0).tolist(),'old_hit_CPU_miss_all10':(oh&~eh).sum(0).tolist(),
   'AP_metrics_CPU_minus_archived_pp':{k:100*(metrics[seed,'EMA'][k]-oldmetrics[seed][j]) for j,k in enumerate(A.METRICS)}}
  flows=defaultdict(Counter)
  for a in anns:
   aid=a['id'];md=meta[seed,aid];rr=records[seed,'raw'][aid];er=records[seed,'EMA'][aid]
   pr={k:md[k] for k in ('seed','image_id','annotation_id','category_id','short_side_640','size_group','density32','occlusion','truncation')}
   for branch,r in (('raw',rr),('EMA',er)):
    for k,v in r.items():
     if k not in ('seed','branch','image_id','annotation_id','category_id'):pr[branch+'_'+k]=v
   for name in NAMES:pr['EMA_minus_raw_'+name+'_maxIoU']=er[name+'_maxIoU']-rr[name+'_maxIoU']
   for th,t in ((50,0),(75,5)):
    pr['old_EMA_hit'+str(th)]=int(oh[ai[aid],t])
    flows['all_'+str(th)][rr['state'+str(th)]+'->'+er['state'+str(th)]]+=1
    if rr['state'+str(th)]=='G' and er['hit'+str(th)]:
     flows['rawG_recovered_EMA_bits'+str(th)][er['geometry_bits'+str(th)]]+=1
   paired.append(pr)
  geometryflow[str(seed)]={k:dict(v) for k,v in flows.items()}
 tokenrows=[]
 for seed in (1,2):
  for im in ims:
   iid=im['id'];n=len(set(tokenids[seed,'raw',iid])&set(tokenids[seed,'EMA',iid]));tokenrows.append({'seed':seed,'image_id':iid,'selected_per_branch':300,'shared_encoder_token_indices':n})
 # Rectangular output: conditional fields are explicitly null when not applicable.
 allkeys=list(dict.fromkeys(k for r in paired for k in r))
 paired=[{k:r.get(k) for k in allkeys} for r in paired]
 csvout('per_GT_stages_and_transitions.csv',paired);csvout('per_image.csv',perimage);csvout('selected_token_overlap.csv',tokenrows)
 allcountkeys=sorted(set(k for c in groupmap.values() for k in c))
 csvout('grouped_failure_and_geometry.csv',[{'seed':s,'branch':b,'group_type':t,'group':g,**{k:c.get(k,0) for k in allcountkeys}} for (s,b,t,g),c in groupmap.items()])
 result={'analysis_protocol_sha256':ph,'summaries':summaries,'same_backend_transitions':transitions,'archived_vs_CPU_EMA':context,
  'state_flows_and_G_recovery_geometry':geometryflow,'selected_token_overlap':{str(s):A.quant([r['shared_encoder_token_indices'] for r in tokenrows if r['seed']==s]) for s in (1,2)},
  'count_rows':{'perGT':len(paired),'per_image':len(perimage),'source_inputs':len(sources)},'operations':{'new_forwards':0,'model_builds':0,'optimizer_steps':0,'native_evaluations':4},'elapsed_seconds':time.time()-start}
 write('results.json',result)
 unchanged=all(A.sha(p)==h for p,h in sources.items());assert unchanged
 write('input_manifest.json',sources)
 write('verification.json',{'status':'PASS','raw12_metrics_replayed_both_seeds':True,'raw_final_allGT_maxIoU_exact':True,'allGT_matching_reciprocity':True,
  'raw_partitions_and_hits_exact':True,'replay_all2192_files_match_declared_sha256':True,'all_tracked_inputs_unchanged':unchanged,'inputs':len(sources),'analysis_protocol_sha256':ph})
 print('COMPLETE',result['elapsed_seconds'],flush=True)
if __name__=='__main__':main()
