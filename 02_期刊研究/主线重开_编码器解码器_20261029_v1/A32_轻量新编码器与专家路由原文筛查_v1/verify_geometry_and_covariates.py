"""Independent event/statistic verification plus explicitly post-hoc covariate sensitivity."""
import collections,hashlib,json,math,pathlib,sys,time
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D=pathlib.Path(__file__).resolve().parent
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(p,o):
 with pathlib.Path(p).open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2);f.write('\n')
def iou(a,b):
 area=max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))*max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
 return area/(a[2]*a[3]+b[2]*b[3]-area) if area else 0.
def contributions(rows,arm,mode,bins,images):
 groups=collections.defaultdict(list)
 for r in rows:
  size=sum(r['area_input']>=b for b in bins[1:-1])
  key=(r['image_id'],r['category_id'],size)
  if mode=='fine_area_occlusion_truncation':key+=(r['occlusion'],r['truncation'])
  groups[key].append(r)
 values={i:[0.,0.] for i in images};eligible=0;ns=0
 for key,ss in groups.items():
  exposed=[r for r in ss if r['same16_different8']];unexposed=[r for r in ss if not r['same16_different8']]
  if not exposed or not unexposed:continue
  n1=len(exposed);n0=len(unexposed);w=n1*n0/(n1+n0)
  numerator=w*(sum(r['miss'][arm] for r in exposed)/n1-sum(r['miss'][arm] for r in unexposed)/n0)
  values[key[0]][0]+=numerator;values[key[0]][1]+=w;eligible+=len(ss);ns+=1
 return values,eligible,ns
def scalar_rd(values,selected_images):
 denominator=sum(values[i][1] for i in selected_images)
 return sum(values[i][0] for i in selected_images)/denominator if denominator else math.nan
def main():
 start=time.monotonic();plan=read(D/'geometry_covariate_plan_FROZEN.json')
 for row in plan['inputs'].values():assert sha(row['path'])==row['sha256'],row['path']
 base=read(D/'geometry_exposure_plan_FROZEN.json');result=read(D/'geometry_result.json')
 gt=read(base['inputs']['GT']['path']);all_rows=read(D/'geometry_per_GT.json')['rows'];rows=[r for r in all_rows if r['phase']==[0,0]]
 assert len(rows)==4127
 annotations={g['id']:g for g in gt['annotations']};images=sorted(i['id'] for i in gt['images'])
 # Scalar prediction matching, independently written without importing the prior matcher.
 event_checks=0
 for arm in base['arms']:
  selected=[p for p in read(base['inputs'][arm]['path']) if p['score']>=base['thresholds'][arm]]
  pp=collections.defaultdict(list);gg=collections.defaultdict(list)
  for p in selected:pp[(p['image_id'],p['category_id'])].append(p)
  for g in gt['annotations']:gg[(g['image_id'],g['category_id'])].append(g)
  matched=set();nfp=0
  for key,dd in pp.items():
   for pred in sorted(dd,key=lambda p:-p['score']):
    best=None;overlap=.5
    for g in gg[key]:
     if g['id'] in matched:continue
     value=iou(pred['bbox'],g['bbox'])
     if value>=overlap:overlap=value;best=g['id']
    if best is None:nfp+=1
    else:matched.add(best)
  assert nfp==926
  for r in all_rows:
   assert r['miss'][arm]==(r['GT_id'] not in matched)
   event_checks+=1
 # Verify annotation joins and original-coordinate -> network-coordinate centers.
 raw=read(base['inputs']['raw_annotations']['path'])['cases'];source={};geometries={}
 for case in raw:
  geometries[case['image_id']]=case['geometry']
  for r in case['rows']:source[case['file_name']+':'+str(r['source_row'])]=r['values']
 flags=collections.Counter()
 for r in rows:
  g=annotations[r['GT_id']];v=source[g['stable_uid']]
  assert v[:4]==g['bbox'] and int(v[5])-1==r['category_id']
  r['truncation']=int(v[6]);r['occlusion']=int(v[7]);assert r['truncation']==v[6] and r['occlusion']==v[7]
  flags[(r['truncation'],r['occlusion'])]+=1
  geo=geometries[r['image_id']];x=(g['bbox'][0]+.5*g['bbox'][2])*geo['ratio']+geo['pad_ltrb'][0];y=(g['bbox'][1]+.5*g['bbox'][3])*geo['ratio']+geo['pad_ltrb'][1]
  assert [x,y]==r['center_input']
  assert abs(g['bbox'][2]*g['bbox'][3]*geo['ratio']**2-r['area_input'])<1e-10
 boot=np.load(D/'geometry_bootstrap.npz',allow_pickle=False)
 draws=np.random.default_rng(base['bootstrap_seed']).integers(0,100,size=(1000,100));assert np.array_equal(draws,boot['draws'])
 primary_error=0.
 for arm in base['arms']:
  values,eligible,ns=contributions(rows,arm,'coarse_area',[0,64,256,1024],images)
  value=scalar_rd(values,images);reference=result['primary_stratified'][arm]
  assert eligible==reference['eligible_GT'] and ns==reference['shared_strata']
  primary_error=max(primary_error,abs(value-reference['weighted_risk_difference']));assert primary_error<1e-12
  for draw in [0,499,999]:
   value=scalar_rd(values,[images[i] for i in draws[draw]]);assert abs(value-boot[arm][draw])<1e-12
 # Sensitivity was frozen only after seeing the initial association; never relabel it confirmatory.
 draws2=np.random.default_rng(plan['bootstrap_seed']).integers(0,100,size=(1000,100));sens={};arrays={'draws':draws2}
 for mode in plan['modes']:
  sens[mode]={}
  for arm in base['arms']:
   values,eligible,ns=contributions(rows,arm,mode,plan['finer_area_bins'],images)
   samples=np.asarray([scalar_rd(values,[images[i] for i in draw]) for draw in draws2])
   sens[mode][arm]=dict(weighted_risk_difference=scalar_rd(values,images),bootstrap95=np.nanquantile(samples,[.025,.975]).tolist(),eligible_GT=eligible,shared_strata=ns)
   arrays[mode+'__'+arm]=samples
 np.savez_compressed(D/'geometry_covariate_bootstrap.npz',**arrays)
 out=dict(status='PASS_INDEPENDENT_NATIVE_EVENTS_CENTERS_AND_PRIMARY_STATISTICS',new_model_calls=0,new_optimizer_updates=0,verified_small_GT=4127,independently_verified_head_GT_phase_labels=event_checks,primary_statistic_max_abs_error=primary_error,original_bootstrap_draws_checked=1000,original_bootstrap_draws_reaggregated=[0,499,999],posthoc_sensitivity=sens,visibility_flags=[dict(truncation=k[0],occlusion=k[1],count=v) for k,v in sorted(flags.items())],posthoc_scope=plan['scope'],elapsed_seconds=time.monotonic()-start)
 save(D/'geometry_independent_verification_and_covariates.json',out)
 print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
