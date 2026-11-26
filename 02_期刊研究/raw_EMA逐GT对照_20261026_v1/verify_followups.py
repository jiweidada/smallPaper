"""Recount the two zero-forward checks from their delivered row tables."""
import csv,json,hashlib
from pathlib import Path
from collections import Counter
P=Path(__file__).resolve().parent;O=P/'后续零训练预检_20261026_v1'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def rows(n):return list(csv.DictReader(open(O/n,encoding='utf-8-sig')))
G=read(O/'shared_token_results.json');B=read(O/'B_results.json');ss=rows('shared_token_per_GT.csv');bb=rows('B_per_GT.csv')
assert len(ss)==len(bb)==77518
assert len({(r['seed'],r['annotation_id']) for r in ss})==len({(r['seed'],r['annotation_id']) for r in bb})==77518
for ref in G['primary_counts']:
 s=str(ref['seed']);t=int(round(ref['IoU']*100));pop=ref['population']
 if pop=='rawG_recovered':continue
 rr=[r for r in ss if r['seed']==s and (pop=='allGT' or r[f'raw000_{t}']=='1')]
 c=Counter(r[f'EMA_initial_witness{t}'] for r in rr);assert len(rr)==ref['GT']
 assert all(c[k]==ref[k] for k in ('no_witness','common_only','added_only','both'))
 for r in rr:
  a=float(r['EMA_common_initial_maxIoU'])>=ref['IoU'];b=float(r['EMA_only_initial_maxIoU'])>=ref['IoU']
  assert r[f'EMA_initial_witness{t}']==('both' if a and b else 'common_only' if a else 'added_only' if b else 'no_witness')
for s in ('1','2'):
 rr=[r for r in bb if r['seed']==s];q=B['seeds'][s]
 assert sum(int(r['B50']) for r in rr)==q['TP50_B'] and sum(int(r['B75']) for r in rr)==q['TP75_B']
 for t in (50,75):
  h={k:{int(r['annotation_id']) for r in rr if r[f'{k}{t}']=='1'} for k in 'ABC'};z=q['sets'][str(t)]
  assert sorted(h['B']-h['A'])==z['B_new_vs_A'];assert sorted(h['A']-h['B'])==z['B_lost_vs_A']
  assert sorted(h['C']-h['B'])==z['C_new_vs_B'];assert sorted(h['B']-h['C'])==z['C_lost_vs_B']
  assert len((h['B']-h['A'])&(h['C']-h['A']))==z['B_new_intersect_C_new_vs_A']
 group=rows('B_by_class_area.csv')
 for typ in ('class','area'):
  gg=[r for r in group if r['seed']==s and r['type']==typ];assert sum(int(r['GT']) for r in gg)==38759
  for k in 'ABC':
   for t in (50,75):assert sum(int(r[f'{k}{t}']) for r in gg)==len(h[k]) if t==75 else sum(int(r[f'{k}{t}']) for r in gg)==sum(int(r[f'{k}{t}']) for r in rr)
pre=read(O/'B_preflight.json');assert pre['status']=='PASS' and not pre['invalid_images']
out={'status':'PASS','shared_token_rows':len(ss),'B_rows':len(bb),'B_exact_new_lost_ID_sets_recounted':True,'class_area_partitions_sum_allGT':True,
 'shared_witness_labels_and_counts_recomputed':True,'model_forwards':0,'evaluator_calls':0,
 'output_sha256':{n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ('shared_token_per_GT.csv','B_per_GT.csv','shared_token_results.json','B_results.json')}}
with open(O/'independent_followup_counts.json','x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print('Independent zero-forward follow-up counts PASS')
