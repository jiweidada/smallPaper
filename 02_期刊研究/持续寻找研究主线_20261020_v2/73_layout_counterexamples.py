"""Registered all-GT density control geometry and post-hoc cached set diagnostics."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import pathlib,sys,json,itertools,importlib.util,time
ROOT=pathlib.Path(__file__).resolve().parent;WORK=ROOT.parents[1]
sys.path.insert(0,str(WORK/'英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1'))
import common as n
torch,np=n.torch,n.np
OUT=ROOT/'74_layout_counterexamples'

def load_file(name,file):
 s=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def add(t,k,v):t[k]=t.get(k,0)+int(v)

def main():
 start=time.monotonic();OUT.mkdir(exist_ok=False);resources=n.resource_limits();torch.set_grad_enabled(False)
 g=load_file('geom51','51_mosaic_layout_geometry.py')
 geo=json.loads((ROOT/'51_mosaic_layout_geometry/01_rows.json').read_text(encoding='utf8'))
 tags=json.loads((ROOT/'49_augmentation_replay_audit_v4/05_tagged_source_identity.json').read_text(encoding='utf8'))
 ids={(r['seed'],r['position']):r for r in tags}
 chosen=json.loads((ROOT/'62_layout_rgb/00_selection.json').read_text(encoding='utf8'))
 keys={(r['seed'],r['position']) for r in chosen}
 newrows=[];geom={}
 for r in geo:
  key=r['seed'],r['position'];ident=ids[key]
  boxes=np.array(ident['boxes'],float);slots=np.array([x['source_slot'] for x in ident['instances']],int)
  m=boxes[:,2:].min(1)*640;small=m<4;counts=np.bincount(slots,minlength=4);tiny=np.bincount(slots[small],minlength=4)
  sizes=np.array([p['size'] for p in r['pieces']]);active=(sizes>0).all(1)
  candidates=[{'layout':'original','scales':np.ones(4)}]
  for layout in itertools.product([.4,.5,.6],repeat=3):
   s=np.ones(4);s[active]=np.min(g.cells(layout)[active]/sizes[active],axis=1)
   if (s[active]<.8-1e-9).any() or ((m>=4)&(m*s[slots]<4-1e-9)).any():continue
   candidates.append({'layout':list(layout),'scales':s})
  risk=g.choose(candidates,lambda c:(tiny*c['scales']).sum())
  assert risk['layout']==r['arms']['simple_risk']['layout'] and np.array_equal(risk['scales'],np.array(r['arms']['simple_risk']['scales']))
  allc=g.choose(candidates,lambda c:(counts*c['scales']).sum())
  row={'seed':key[0],'position':key[1],'in_detection_panel':key in keys,'pieces':r['pieces'],
       'layout':allc['layout'],'scales':allc['scales'].tolist(),'different_from_risk':allc['layout']!=risk['layout'],
       'original_sub4':int(small.sum()),'all_count_cross4':int((small&(m*allc['scales'][slots]>=4)).sum()),
       'risk_cross4':r['arms']['simple_risk']['crossed_to_ge4'],'GT':len(m)}
  newrows.append(row)
  for scope in ['all_events']+(['fixed_16_panel'] if key in keys else []):
   t=geom.setdefault(scope,{}).setdefault(str(key[0]),{})
   add(t,'events',1);add(t,'different_events',row['different_from_risk'])
   for k in ['GT','original_sub4','all_count_cross4','risk_cross4']:add(t,k,row[k])
 n.save_json(OUT/'01_all_count_geometry.json',{'state':'complete','summary':geom,'rows':newrows})

 cached=json.loads((ROOT/'64_frozen_layout_detection/01_rows.json').read_text(encoding='utf8'))
 resultrows=[];totals={};bounds=[0,4,8,12,float('inf')];binlabels=['lt4','4_to_8','8_to_12','ge12']
 cases={}
 for p in sorted((ROOT/'62_layout_rgb').glob('case*.pt')):
  c=torch.load(p,map_location='cpu',weights_only=False);cases[c['event']['seed'],c['event']['position']]=c
 for row in cached:
  assert n.psutil.virtual_memory().available/2**30>8 and time.monotonic()-start<1200
  case=cases[row['augmentation_seed'],row['position']];small=case['original_sub4'].numpy().astype(bool)
  r,o=row['arms']['simple_risk'],row['arms']['original'];neut=row['arms'].get('neutral',o)
  ri=np.array(r['perGT_class05'],bool);oi=np.array(o['perGT_class05'],bool)
  new=~oi&ri;lost=oi&~ri
  m=case['targets']['simple_risk']['boxes'][:,2:].min(1).values.numpy()*640
  slots=np.array([x['source_slot'] for x in case['source_instances']]);s=np.array(case['metadata']['simple_risk']['scale'])[slots]
  accept=bool(not lost.any() and r['all_geo05']>=o['all_geo05'] and (new&small).any())
  item={'model_seed':row['model_seed'],'augmentation_seed':row['augmentation_seed'],'position':row['position'],
    'strict_teacher_accept':accept,'GT':len(small),'sub4':int(small.sum()),'max_risk_size_of_original_sub4':float(m[small].max()) if small.any() else None,'bins':{},'neutral75_vs_risk_by_scale':{}}
  t=totals.setdefault(str(row['model_seed']),{'cases':0,'teacher_accepted_cases':0,'teacher_new_sub4':0,'teacher_lost_sub4':0,'teacher_all_new':0,'teacher_all_lost':0,'bins':{},'neutral75_vs_risk_by_scale':{}})
  add(t,'cases',1);add(t,'teacher_accepted_cases',accept)
  if accept:
   add(t,'teacher_new_sub4',(new&small).sum());add(t,'teacher_lost_sub4',(lost&small).sum());add(t,'teacher_all_new',new.sum());add(t,'teacher_all_lost',lost.sum())
  for lo,hi,label in zip(bounds[:-1],bounds[1:],binlabels):
   a=small&(m>=lo)&(m<hi)
   stat={'GT':int(a.sum()),'new_sub4_presence_class05':int((a&new).sum()),'lost_sub4_presence_class05':int((a&lost).sum())}
   item['bins'][label]=stat;tt=t['bins'].setdefault(label,{})
   for k,v in stat.items():add(tt,k,v)
  nr=np.array(neut['perGT_geo075'],bool);rr=np.array(r['perGT_geo075'],bool)
  for label,sel in [('shrink',s<1-1e-9),('identity_scale',np.abs(s-1)<=1e-9),('grow',s>1+1e-9)]:
   stat={'GT':int(sel.sum()),'neutral_only_geo075':int((sel&nr&~rr).sum()),'risk_only_geo075':int((sel&rr&~nr).sum())}
   item['neutral75_vs_risk_by_scale'][label]=stat;tt=t['neutral75_vs_risk_by_scale'].setdefault(label,{})
   for k,v in stat.items():add(tt,k,v)
  resultrows.append(item)
 decision={s:bool(t['teacher_accepted_cases']>0 and t['teacher_new_sub4']>0) for s,t in totals.items()}
 n.save_json(OUT/'02_cached_diagnostics.json',{'state':'complete','analysis_type':'posthoc_descriptive_and_prospective_strict_rule_screen','totals':totals,'rows':resultrows,
  'strict_teacher_rule_pass_per_model':decision,'strict_teacher_rule_pass_both':all(decision.values()),
  'max_risk_size_of_original_sub4':max(r['max_risk_size_of_original_sub4'] or 0 for r in resultrows),
  'scope':'Counts use per-GT existence, not arbitrary identities of maximum matching. No AP or training transfer.'})
 n.save_json(OUT/'03_execution.json',{'state':'complete','resource':resources,'seconds':time.monotonic()-start,'protocol_sha256':n.sha(ROOT/'72_固定曝光布局_反例与得失首检协议.txt'),
  'script_sha256':n.sha(__file__),'model_constructed':False,'model_forwards':0,'model_backwards':0,'optimizer_steps':0,'new_training':False,'heldout_reads':False,
  'sources':{f:n.sha(ROOT/f) for f in ['51_mosaic_layout_geometry/01_rows.json','49_augmentation_replay_audit_v4/05_tagged_source_identity.json','64_frozen_layout_detection/01_rows.json','62_layout_rgb/00_selection.json']}})
 n.save_json(OUT/'04_SHA256.json',{p.name:n.sha(p) for p in OUT.iterdir() if p.is_file()})
 print(json.dumps({'state':'complete','geometry':geom,'diagnostic_totals':totals,'strict_teacher_rule_pass_both':all(decision.values()),'max_risk_size_of_original_sub4':max(r['max_risk_size_of_original_sub4'] or 0 for r in resultrows)},ensure_ascii=False))

if __name__=='__main__':main()
