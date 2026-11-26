"""Spatial-token membership witnesses from already saved initial references."""
import sys,json,csv,hashlib,time,importlib.util
from pathlib import Path
from collections import defaultdict,Counter
sys.dont_write_bytecode=True
import numpy as np
O=Path(__file__).resolve().parent;PAIR=O.parent;ROOT=PAIR.parent;REP=PAIR/'同后端CPU重放_20261026_v1'
VAL=Path(r'C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val/annotations/val.json')
spec=importlib.util.spec_from_file_location('prior_audit',ROOT/'问题优先诊断_20261026_v1/audit.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(n,j):
 with open(O/n,'x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvout(n,rr):
 with open(O/n,'x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def edges(pb,im):
 x,y,w,h=pb.T;t=np.float32(2)
 return np.stack((x-w/t,y-h/t,x+w/t,y+h/t),-1)*np.asarray([im['width'],im['height']]*2,dtype=np.float32)
def main():
 start=time.time();ph=A.sha(O/'protocol.json');assert ph==(O/'protocol.sha256').read_text('utf-8').strip();inputs={}
 def track(p):inputs[str(p)]=A.sha(p)
 for p in (O/'protocol.json',Path(__file__),VAL,REP/'analysis_v1/per_GT_stages_and_transitions.csv'):track(p)
 ds=read(VAL);by=defaultdict(list)
 for a in ds['annotations']:by[a['image_id']].append(a)
 prior=list(csv.DictReader(open(REP/'analysis_v1/per_GT_stages_and_transitions.csv',encoding='utf-8-sig')))
 pi={(int(r['seed']),int(r['annotation_id'])):r for r in prior};rows=[];agg=defaultdict(Counter)
 for seed in (1,2):
  for im in sorted(ds['images'],key=lambda x:x['id']):
   data={}
   for b in ('raw','EMA'):
    p=REP/f'seed{seed}_{b}/image_{im["id"]:06d}.npz';track(p)
    with np.load(p,allow_pickle=False) as z:data[b]={'ids':z['encoder_index'],'bb':z['encoder_initial_boxes']}
   common=set(data['raw']['ids'])&set(data['EMA']['ids']);assert common
   aa=by[im['id']];g=np.asarray([a['bbox'] for a in aa],dtype=np.float64).reshape(-1,4);g[:,2:]+=g[:,:2]
   maxima={}
   for b in ('raw','EMA'):
    ids=data[b]['ids'];M=A.iou_np(edges(data[b]['bb'],im),g);mask=np.asarray([i in common for i in ids])
    maxima[b+'_common']=M[mask].max(0)
    maxima[b+'_only']=M[~mask].max(0) if (~mask).any() else np.zeros(len(aa))
   for j,a in enumerate(aa):
    aid=a['id'];p=pi[seed,aid];r={'seed':seed,'image_id':im['id'],'annotation_id':aid,'category_id':a['category_id'],'size_group':p['size_group']}
    for k,m in maxima.items():r[k+'_initial_maxIoU']=float(m[j])
    for b in ('raw','EMA'):assert max(r[b+'_common_initial_maxIoU'],r[b+'_only_initial_maxIoU'])==float(p[b+'_initial_maxIoU'])
    for th in (50,75):
     t=th/100;shared=r['EMA_common_initial_maxIoU']>=t;added=r['EMA_only_initial_maxIoU']>=t
     w='both' if shared and added else 'common_only' if shared else 'added_only' if added else 'no_witness'
     r[f'EMA_initial_witness{th}']=w;r[f'raw000_{th}']=int(p[f'raw_geometry_bits{th}']=='000')
     populations=['allGT','paired_final_'+str(p[f'raw_hit{th}'])+str(p[f'EMA_hit{th}'])]
     if r[f'raw000_{th}']:populations.append('raw000')
     if p[f'raw_state{th}']=='G':populations.append('rawG')
     if p[f'raw_state{th}']=='G' and p[f'EMA_hit{th}']=='1':populations.append('rawG_recovered')
     if float(p['raw_initial_maxIoU'])<t:populations.append('raw_initial_miss')
     for pop in populations:
      for typ,label in (('all','all'),('size',p['size_group']),('class',str(a['category_id']))):
       c=agg[seed,th,pop,typ,label];c['GT']+=1;c[w]+=1
    rows.append(r)
 out=[{'seed':s,'IoU':th/100,'population':pop,'group_type':typ,'group':label,**{k:c.get(k,0) for k in ('GT','no_witness','common_only','added_only','both')}} for (s,th,pop,typ,label),c in agg.items()]
 assert len(rows)==77518
 for r in out:assert sum(r[k] for k in ('no_witness','common_only','added_only','both'))==r['GT']
 maincounts=[r for r in out if r['group_type']=='all' and r['population'] in ('allGT','raw000','rawG_recovered')]
 csvout('shared_token_per_GT.csv',rows);csvout('shared_token_by_group.csv',out)
 write('shared_token_results.json',{'protocol_sha256':ph,'primary_counts':maincounts,'perGT_rows':len(rows),
  'meaning':'Initial geometry existence by shared/added spatial-token membership, no object identity or causal attribution to selection/features.',
  'operations':{'forwards':0,'model_builds':0,'head_calls':0,'evaluator_calls':0,'optimizer_steps':0},'elapsed_seconds':time.time()-start})
 assert all(A.sha(p)==h for p,h in inputs.items());write('shared_token_input_manifest.json',inputs)
 write('shared_token_verification.json',{'status':'PASS','allGT_rows':len(rows),'initial_full_pool_maxima_exactly_reproduce_prior':True,'all_group_sums':True,'unchanged_inputs':len(inputs),'protocol_sha256':ph})
 print(json.dumps(maincounts,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
