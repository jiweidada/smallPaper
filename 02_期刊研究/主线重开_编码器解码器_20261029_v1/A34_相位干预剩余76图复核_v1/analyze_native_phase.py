"""Frozen local metric/contrast specification for fixed-pixel native phase diagnosis."""
import ast,collections,copy,hashlib,json,math,pathlib,sys
from fractions import Fraction
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D=pathlib.Path(__file__).resolve().parent
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(p,o):
 with pathlib.Path(p).open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2);f.write('\n')
def load_functions(path,names):
 tree=ast.parse(pathlib.Path(path).read_text(encoding='utf-8'))
 nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
 assert len(nodes)==len(names)
 scope=dict(np=np,math=math,THRESHOLDS=np.arange(.5,.96,.05),RECALL=np.linspace(0,1,101))
 exec(compile(ast.Module(body=nodes,type_ignores=[]),'<audited-own-metric-definitions>','exec'),scope)
 return scope
def exposure_flags(annotations,cases,dy):
 flags={}
 for case in cases:
  aa=[g for g in annotations if g['image_id']==case['image_id'] and g['_area_input']<1024]
  scale=Fraction(str(case['geometry']['ratio'])).limit_denominator(1000000)
  assert abs(float(scale)-case['geometry']['ratio'])<1e-12
  left,top=case['geometry']['pad_ltrb'][:2];groups=collections.defaultdict(list);fine={}
  for g in aa:
   x=(Fraction(str(g['bbox'][0]))+Fraction(str(g['bbox'][2]))/2)*scale+left
   y=(Fraction(str(g['bbox'][1]))+Fraction(str(g['bbox'][3]))/2)*scale+top+dy
   groups[(int(x//16),int(y//16))].append(g['id']);fine[g['id']]=(int(x//8),int(y//8))
  for members in groups.values():
   for gid in members:flags[gid]=any(fine[gid]!=fine[other] for other in members if other!=gid)
 return flags
def prefix_thresholds(entries,nimages,weights,budget):
 """Whole-score-block global prefixes; re-estimate thresholds for every resample."""
 entries=sorted(entries,key=lambda x:(-x[0],x[1],x[2],x[3]))
 if not entries:return np.full(weights.shape[1],np.inf),np.zeros(weights.shape[1],dtype=int)
 scores=np.asarray([x[0] for x in entries],dtype=float)
 event=np.zeros((len(entries),nimages),dtype=np.int32)
 for i,x in enumerate(entries):
  if x[4]==0:event[i,x[1]]=1
 ends=np.flatnonzero(np.r_[scores[1:]!=scores[:-1],True])
 prefix=np.vstack([np.zeros((1,nimages),dtype=np.int32),event.cumsum(axis=0,dtype=np.int32)[ends]])
 boundaries=np.r_[np.inf,scores[ends]]
 # Chunking bounds memory; integer counts keep the FP budget exact.
 thresholds=[];actual=[]
 for begin in range(0,weights.shape[1],100):
  w=weights[:,begin:begin+100];total=prefix@w;last=(total<=budget).sum(axis=0)-1
  thresholds.extend(boundaries[last].tolist());actual.extend(total[last,np.arange(w.shape[1])].tolist())
 return np.asarray(thresholds),np.asarray(actual,dtype=int)
def fixture():
 entries=[(.9,0,0,1,1),(.8,1,0,2,0),(.8,0,0,3,2),(.7,0,0,4,0)]
 weights=np.asarray([[1,2],[1,0]],dtype=int)
 tau,fp=prefix_thresholds(entries,2,weights,0)
 assert tau.tolist()==[.9,.8] and fp.tolist()==[0,0]
 tau,fp=prefix_thresholds(entries,2,weights,1)
 assert tau.tolist()==[.8,.8] and fp.tolist()==[1,0]
 tau,fp=prefix_thresholds([(.9,0,0,1,0)],1,np.ones((1,1),dtype=int),0)
 assert math.isinf(tau[0]) and fp[0]==0
 return dict(status='PASS_HAND_COMPUTED_TIES_AND_WEIGHTED_FP_PREFIX',model_calls=0)
def main():
 fixture();cfg=read(D/'phase_protocol_FROZEN.json');E=D/'downloaded_evidence'
 assert sha(__file__)==cfg['local_inputs']['analysis']['sha256']
 for row in cfg['local_inputs'].values():assert sha(row['path'])==row['sha256'],row['path']
 complete=read(D/'remote_complete.json');assert complete['counts']['whole_detector_forwards']==533
 native=read(E/'result.json');assert native['counts']==complete['counts']
 metric=load_functions(cfg['local_inputs']['matcher']['path'],['matrix_iou','match_image','category_precision','ap'])
 region=load_functions(cfg['local_inputs']['region_helper']['path'],['matlab_round','ignore_integral','region_keep'])
 selected={c['image_id'] for c in cfg['cases']};images=sorted(selected);imidx={im:i for i,im in enumerate(images)}
 original=read(cfg['local_inputs']['GT']['path']);ann=[g for g in original['annotations'] if g['image_id'] in selected]
 cases={c['image_id']:c for c in cfg['cases']};cats=list(range(10));small=sorted(g['id'] for g in ann if g['_area_input']<1024)
 gid_to_image={g['id']:imidx[g['image_id']] for g in ann};small_im=np.asarray([gid_to_image[gid] for gid in small])
 gg=collections.defaultdict(list)
 for g in ann:gg[(g['category_id'],g['image_id'])].append(g)
 raw=read(cfg['local_inputs']['raw_annotations']['path'])['cases'];integrals={}
 for case in raw:
  if case['image_id'] in selected:
   values=np.asarray([r['values'] for r in case['rows']],dtype=float)
   integrals[case['image_id']]=region['ignore_integral'](values,case['height'],case['width'])
 e0=exposure_flags(ann,cfg['cases'],0);e8=exposure_flags(ann,cfg['cases'],8)
 for dy in [-8]:assert exposure_flags(ann,cfg['cases'],dy)==e8
 for dy in [-32,-16,16,32]:assert exposure_flags(ann,cfg['cases'],dy)==e0
 separate=np.asarray([e0[x] and not e8[x] for x in small]);merge=np.asarray([not e0[x] and e8[x] for x in small])
 group_info=dict(small_GT=len(small),separate_GT=int(separate.sum()),merge_GT=int(merge.sum()),separate_images=len({small_im[i] for i in np.flatnonzero(separate)}),merge_images=len({small_im[i] for i in np.flatnonzero(merge)}))
 rng=np.random.default_rng(cfg['metrics']['bootstrap_seed']);draws=rng.integers(0,len(images),size=(cfg['metrics']['bootstrap_draws'],len(images)))
 weights=np.column_stack([np.ones(len(images),dtype=np.int32)]+[np.bincount(d,minlength=len(images)).astype(np.int32) for d in draws])
 assert np.all(weights.sum(0)==len(images));gt_weights=weights[small_im]
 arrays=dict(draws=draws,weights=weights,small_GT_ids=np.asarray(small,dtype=np.int64),small_image_index=small_im,separate=separate,merge=merge)
 metrics={};all_hits={};all_thresholds={};all_fp={}
 (D/'filtered_predictions').mkdir(exist_ok=False)
 for key in cfg['output_groups']:
  pp=read(E/(key+'_predictions_original.json'));group=collections.defaultdict(list)
  for pred in pp:group[pred['image_id']].append(pred)
  kept=[]
  for im in images:
   rows=group[im]
   if not rows:continue
   boxes=np.asarray([x['bbox'] for x in rows],dtype=float)
   mask=region['region_keep'](boxes,integrals[im])
   kept.extend(x for x,k in zip(rows,mask) if k)
  save(D/'filtered_predictions'/(key+'.json'),kept)
  dd=collections.defaultdict(list)
  for i,p in enumerate(kept):p['_id']=i+1;dd[(p['category_id'],p['image_id'])].append(p)
  all_records={};small_records={};entries=[];gt_scores={}
  for c in cats:
   for im in images:
    rr=metric['match_image'](dd[(c,im)],gg[(c,im)],[0,1e10]);all_records[(c,im)]=rr
    for j,score in enumerate(rr['scores']):
     if rr['ignore'][0,j]:continue
     gid=int(rr['matches'][0,j]);entries.append((float(score),imidx[im],c,int(rr['ids'][j]),gid))
     if gid:assert gid not in gt_scores;gt_scores[gid]=float(score)
    s=cases[im]['geometry']['ratio'];left,top=cases[im]['geometry']['pad_ltrb'][:2]
    ginput=copy.deepcopy(gg[(c,im)]);dinput=copy.deepcopy(dd[(c,im)])
    for x in ginput:
     x['bbox']=[x['bbox'][0]*s+left,x['bbox'][1]*s+top,x['bbox'][2]*s,x['bbox'][3]*s];x['area']=x['_area_input']
    for x in dinput:x['bbox']=[x['bbox'][0]*s+left,x['bbox'][1]*s+top,x['bbox'][2]*s,x['bbox'][3]*s]
    small_records[(c,im)]=metric['match_image'](dinput,ginput,[0,math.nextafter(1024.,-math.inf)])
  ap,ap50=metric['ap'](all_records,images,cats);aps,_=metric['ap'](small_records,images,cats)
  tau,fp=prefix_thresholds(entries,len(images),weights,cfg['metrics']['FP_budget'])
  score=np.asarray([gt_scores.get(gid,-np.inf) for gid in small]);hits=score[:,None]>=tau[None,:]
  all_hits[key]=hits;all_thresholds[key]=tau;all_fp[key]=fp
  metrics[key]=dict(AP_original_all=ap,AP50_original_all=ap50,AP_input_small=aps,FP_budget=cfg['metrics']['FP_budget'],actual_FP=int(fp[0]),threshold=None if math.isinf(tau[0]) else float(tau[0]),small_TP=int(hits[:,0].sum()),small_FN=len(small)-int(hits[:,0].sum()),retained_predictions=len(kept))
  arrays[key+'_hits']=hits;arrays[key+'_thresholds']=tau;arrays[key+'_FP']=fp
 def mean_for(values,mask):
  denominator=(gt_weights*mask[:,None]).sum(0)
  return np.divide((values*gt_weights*mask[:,None]).sum(0),denominator,out=np.full(weights.shape[1],np.nan),where=denominator>0)
 contrasts={};gates={}
 for branch in ['M','O']:
  hh={dy:all_hits[branch+'_'+str(dy)].astype(float) for dy in cfg['shifts_y']}
  difference=(hh[-8]+hh[8]-hh[-32]-hh[32])/2
  contrast=mean_for(difference,separate)-mean_for(difference,merge)
  flip8=(np.abs(hh[-8]-hh[0])+np.abs(hh[8]-hh[0]))/2
  flip32=(np.abs(hh[-32]-hh[0])+np.abs(hh[32]-hh[0]))/2
  excess=mean_for(flip8-flip32,np.ones(len(small),dtype=bool))
  valid=int(np.isfinite(contrast[1:]).sum());ci=np.nanquantile(contrast[1:],[.025,.975]).tolist()
  phases=[all_hits[branch+'_'+str(dy)][:,0] for dy in cfg['shifts_y']]
  union=np.any(np.stack(phases),axis=0);best_by_image=0
  for im in range(len(images)):
   mask=small_im==im;best_by_image+=max(int(h[mask].sum()) for h in phases)
  contrasts[branch]=dict(separation_minus_merge_contrast=float(contrast[0]),paired_reestimated_FP_bootstrap95=ci,valid_bootstrap_draws=valid,flip8_minus_flip32=float(excess[0]),flip8_minus_flip32_bootstrap95=np.nanquantile(excess[1:],[.025,.975]).tolist(),GT_oracle_union_hits=int(union.sum()),GT_oracle_best_phase_per_image_hits=best_by_image,GT_oracle_gap=int(union.sum())-best_by_image,oracle_scope='Descriptive only;not deployable recall,not a common-FP selection policy,not a proposedalgorithm.')
  arrays[branch+'_contrast']=contrast;arrays[branch+'_flip_excess']=excess
 gates['sample_support']=group_info['separate_GT']>=50 and group_info['merge_GT']>=50 and group_info['separate_images']>=5 and group_info['merge_images']>=5
 gates['O_directional_effect_atleast_002']=contrasts['O']['separation_minus_merge_contrast']>=.02
 gates['O_directional_CI_lower_positive']=contrasts['O']['paired_reestimated_FP_bootstrap95'][0]>0 and contrasts['O']['valid_bootstrap_draws']>=950
 gates['O_phase_flip_excess_atleast_002']=contrasts['O']['flip8_minus_flip32']>=.02
 gates['M_same_direction']=contrasts['M']['separation_minus_merge_contrast']>=0
 np.savez_compressed(D/'paired_phase_bootstrap.npz',**arrays)
 save(D/'native_phase_analysis_result.json',dict(status='COMPLETED_FROZEN_PHASE_DIAGNOSIS',images=len(images),groups=group_info,metrics=metrics,contrasts=contrasts,gates=gates,mechanism_question_promotion=all(gates.values()),new_algorithm_validated=False,new_optimizer_updates=0,scope=cfg['interpretation_limits']))
 print(json.dumps(dict(status='COMPLETED_FROZEN_PHASE_DIAGNOSIS',groups=group_info,contrasts=contrasts,gates=gates,mechanism_question_promotion=all(gates.values())),ensure_ascii=False))
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='--fixture':print(json.dumps(fixture()))
 else:main()
