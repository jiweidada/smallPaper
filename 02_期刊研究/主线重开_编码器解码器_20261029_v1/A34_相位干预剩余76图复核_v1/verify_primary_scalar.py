"""Independent scalar native-event, weighted-prefix and paired-contrast checks."""
from pathlib import Path
import collections,itertools,json,math,sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def iou(a,b):
 overlap=max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))*max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
 return overlap/(a[2]*a[3]+b[2]*b[3]-overlap) if overlap else 0.
def prefix(entries,weights,budget):
 fp=0;threshold=math.inf
 for score,block in itertools.groupby(entries,key=lambda x:x[0]):
  items=list(block);additional=sum(int(weights[x[1]]) for x in items if x[4]==0)
  if fp+additional>budget:break
  fp+=additional;threshold=score
 return threshold,fp
cfg=read(D/'phase_protocol_FROZEN.json');result=read(D/'native_phase_analysis_result.json')
boot=np.load(D/'paired_phase_bootstrap.npz',allow_pickle=False);images=sorted(c['image_id'] for c in cfg['cases']);imidx={im:i for i,im in enumerate(images)}
draws=np.random.default_rng(cfg['metrics']['bootstrap_seed']).integers(0,len(images),size=(1000,len(images)));assert np.array_equal(draws,boot['draws'])
gt=[g for g in read(cfg['local_inputs']['GT']['path'])['annotations'] if g['image_id'] in imidx];gg=collections.defaultdict(list)
for g in gt:gg[(g['image_id'],g['category_id'])].append(g)
small=boot['small_GT_ids'].tolist();small_im=boot['small_image_index'];sep=boot['separate'];merge=boot['merge'];hits={};checks=0;original_counts={}
columns=[0,1,500,1000]
for name in cfg['output_groups']:
 pred=read(D/'filtered_predictions'/(name+'.json'));dd=collections.defaultdict(list)
 for i,p in enumerate(pred):p=dict(p,_id=i+1);dd[(p['image_id'],p['category_id'])].append(p)
 entries=[];matched=set();scores={}
 for key,rows in dd.items():
  for p in sorted(rows,key=lambda p:-p['score']):
   best=None;level=.5
   for g in gg[key]:
    if g['id'] in matched:continue
    overlap=iou(p['bbox'],g['bbox'])
    if overlap>=level:level=overlap;best=g['id']
   if best is not None:matched.add(best);scores[best]=p['score']
   elif p['bbox'][2]*p['bbox'][3]<0 or p['bbox'][2]*p['bbox'][3]>1e10:continue
   entries.append((p['score'],imidx[p['image_id']],p['category_id'],p['_id'],best or 0))
 entries.sort(key=lambda x:(-x[0],x[1],x[2],x[3]));hits[name]={}
 for col in columns:
  w=np.ones(len(images),dtype=int) if col==0 else np.bincount(draws[col-1],minlength=len(images))
  tau,fp=prefix(entries,w,cfg['metrics']['FP_budget'])
  assert tau==boot[name+'_thresholds'][col] and fp==boot[name+'_FP'][col],(name,col,tau,fp)
  h=np.asarray([scores.get(gid,-math.inf)>=tau for gid in small]);assert np.array_equal(h,boot[name+'_hits'][:,col]),(name,col)
  hits[name][col]=h;checks+=len(small)
  if col==0:original_counts[name]=dict(actual_FP=fp,small_TP=int(h.sum()))
 for k,val in original_counts[name].items():assert result['metrics'][name][k]==val
maxerror=0.
for branch in ['M','O']:
 for col in columns:
  w=np.ones(len(images),dtype=int) if col==0 else np.bincount(draws[col-1],minlength=len(images));gw=w[small_im]
  delta=[];excess=[]
  for i in range(len(small)):
   h={dy:int(hits[branch+'_'+str(dy)][col][i]) for dy in cfg['shifts_y']}
   delta.append((h[-8]+h[8]-h[-32]-h[32])/2)
   excess.append((abs(h[-8]-h[0])+abs(h[8]-h[0])-abs(h[-32]-h[0])-abs(h[32]-h[0]))/2)
  def mean(mask):
   denom=sum(int(gw[i]) for i in range(len(small)) if mask[i])
   return sum(delta[i]*int(gw[i]) for i in range(len(small)) if mask[i])/denom if denom else math.nan
  contrast=mean(sep)-mean(merge);flip=sum(excess[i]*int(gw[i]) for i in range(len(small)))/sum(int(x) for x in gw)
  for value,expected in [(contrast,boot[branch+'_contrast'][col]),(flip,boot[branch+'_flip_excess'][col])]:
   assert (math.isnan(value) and math.isnan(expected)) or abs(value-expected)<1e-12,(branch,col,value,expected)
   if math.isfinite(value):maxerror=max(maxerror,abs(value-expected))
 expected=np.nanquantile(boot[branch+'_contrast'][1:],[.025,.975]);assert np.allclose(expected,result['contrasts'][branch]['paired_reestimated_FP_bootstrap95'],rtol=0,atol=1e-15)
out=dict(status='PASS_ALL14_SCALAR_FIXED_FP_EVENTS_AND_PRIMARY_PAIRED_CONTRASTS',groups_verified=14,original_and_bootstrap_columns=columns,bootstrap_draws_seed_checked=1000,scalar_GT_hit_checks=checks,max_contrast_error=maxerror,new_model_calls=0,scope='Independent scalar numerical reaggregation using same frozenobservations;notindependent scientificreview,noveltyorcausalvalidation.',original_counts=original_counts)
with (D/'independent_primary_verification.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in out.items() if k!='original_counts'},ensure_ascii=False))
