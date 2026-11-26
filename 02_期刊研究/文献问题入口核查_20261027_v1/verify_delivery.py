import json,pathlib,hashlib,collections,math
B=pathlib.Path(__file__).resolve().parent
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for v in iter(lambda:f.read(4*1024*1024),b''):h.update(v)
 return h.hexdigest()
checks={};bindings={}
def bind(mapping):
 for p,h in mapping.items():
  if p in bindings:assert bindings[p]==h,('conflicting_binding',p)
  bindings[p]=h
bn=read(B/'BN132_attempt02/results.json');bind(bn['source_hashes'])
for s,r in bn['seeds'].items():
 assert not r['advance_fixed_hypothesis']
 for name,v in r['comparisons'].items():
  for t in ['50','75']:
   z=v[t];assert z['new']+z['lost']+z['both']+z['neither']==1583
assert bn['image_equivalent_forwards']==128 and bn['CPU_evaluator_calls']==8
checks['BN_completed_attempt_counts_and_GTinvariants']=True
pixel=read(B/'label_pixel_support_attempt01/results.json');bind(pixel['inputs']);pr=read(B/'label_pixel_support_attempt01/rows.json')
for seed in [1,2]:
 rr=[r for r in pr if r['seed']==seed];t=pixel['totals'][str(seed)];assert len(rr)==t['initial_valid']
 for status in ['retained','drop_sanitize1','drop_sanitize2']:
  subset=[r for r in rr if r['status']==status];assert len(subset)==t[status]
  assert sum(r['zero_support'] for r in subset)==t.get(status+'_zero',0)
 assert not any(r['status']=='retained' and r['zero_support'] for r in rr)
assert pixel['gates']['2D_certificate_controls']==255
checks['pixel_all_occurrence_counts_and_zero_retained']=True
freq=read(B/'frequency_classification_attempt02/results.json');bind(freq['inputs'])
flow=read(B/'frequency_readout_attempt02/results.json');bind(flow['inputs'])
for seed in [1,2]:
 s=str(seed);fr=read(B/f'frequency_classification_attempt02/seed{seed}_rows.json');assert len(fr)==38759
 assert sum(r['actual_TP50'] for r in fr)==[29682,29745][seed-1]
 for th in ['0.5','0.75']:
  rr=read(B/f'frequency_readout_attempt02/seed{seed}_IoU{th}_rows.json');assert len(rr)==38759 and len({r['annotation_id'] for r in rr})==38759
  for name,ids in [('all10',list(range(10))),('low3',[7,6,8]),('high3',[3,0,9])]:
   tt=[r for r in rr if r['category_id'] in ids];q=flow['seeds'][s]['thresholds'][th][name]
   assert len(tt)==q['all_GT'] and dict(collections.Counter(r['stage'] for r in tt))==q['partition']
   assert sum(q['partition'].values())==q['all_GT']
 assert flow['seeds'][s]['canonical_exported_box_exact_rows']==164400
checks['allGT_class_and_full_readout_partitions']=True
hit=read(B/'05a_实际命中query归属_口径更正.json');bind(hit['inputs'])
for r in hit['seeds'].values():
 for x in r.values():assert x['TP_same_owner_nonargmax_trueclass']+x['TP_different_query']==x['owner_argmax_wrong_but_actual_TP']
checks['hit_query_language_correction']=True
grad=read(B/'gradient_sources_attempt01/results.json');bind(grad['inputs'])
assert grad['both_local_screens_pass']
for s,r in grad['seeds'].items():
 rows=read(B/f'gradient_sources_attempt01/seed{s}_class_batch_rows.json');assert len(rows)==80 and len(r['batches'])==8
 for name,ids in [('low3',[7,6,8]),('high3',[3,0,9]),('all10',list(range(10)))]:
  q=r['groups'][name]
  for field in ['support','self_down','cross_down','head_cross_down','background_down','positive_queries','target_instances']:
   value=sum(x[field] for x in rows if x['class'] in ids);assert abs(value-q[field])<1e-12
 for b in r['batches']:assert b['absolute_scalar_difference']<=b['scalar_tolerance_frozen']
checks['native_MAL_scalar_and_source_mass_totals']=True
idx=read(B/'09_质量正样本计量语义_结果.json');assert not idx['fixed_screen_supported'] and idx['consistent_strict_inversions']==[]
assert [idx['seeds'][s]['strict_inversions'] for s in ['1','2']]==[2,0]
for s,r in idx['seeds'].items():assert len(r['pairs'])==45 and len(r['classes'])==10
checks['derived_priority_screen_correctly_negative']=True
protocols=[]
for p in B.glob('*.sha256'):
 actual=sha(p.with_suffix('.json'));assert actual==p.read_text(encoding='utf8').strip();protocols.append(p.name)
checks['frozen_protocol_hashes']=protocols
for p,h in bindings.items():assert sha(p)==h,('source_changed',p)
checks['unique_completed_input_bindings_verified']=len(bindings)
for folder in ['BN132_attempt02','label_pixel_support_attempt01','frequency_classification_attempt02','frequency_readout_attempt02','gradient_sources_attempt01']:
 mf=B/folder/'sha256.json'
 if not mf.exists():continue
 for name,h in read(mf).items():assert sha(B/folder/name)==h,('completed_output_changed',folder,name)
checks['completed_output_hashes']=True
result={'status':'PASS','checks':checks,'limits':'Arithmetic/provenance verification only;not mechanism,method performance,novelty,independent validation or original-host reproduction','new_model_forwards':0,'training':0}
(B/'12_交付核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'status':'PASS','unique_inputs':len(bindings),'frozen_protocols':len(protocols)}))
