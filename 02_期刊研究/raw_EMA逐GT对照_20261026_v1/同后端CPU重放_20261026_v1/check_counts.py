"""Independent full-population counting from delivered per-GT rows."""
import csv,json,hashlib
from pathlib import Path
from collections import Counter
P=Path(__file__).resolve().parent/'analysis_v1';R=json.loads((P/'results.json').read_text('utf-8'))
rows=list(csv.DictReader(open(P/'per_GT_stages_and_transitions.csv',encoding='utf-8-sig')))
assert len(rows)==77518
idx={(int(r['seed']),int(r['annotation_id'])):r for r in rows};assert len(idx)==77518
details={};fixed4={50:[],75:[]};all4G=[]
for s in (1,2):
 rr=[r for r in rows if int(r['seed'])==s];assert len(rr)==38759
 for branch in ('raw','EMA'):
  ref=R['summaries'][f'{s}_{branch}']['allGT'];assert ref['GT']==len(rr)
  for th in (50,75):
   t=th/100;states=Counter(r[f'{branch}_state{th}'] for r in rr);bits=Counter(r[f'{branch}_geometry_bits{th}'] for r in rr)
   assert sum(states.values())==sum(bits.values())==len(rr)
   assert all(states[k]==ref.get(f'state{th}_'+k,0) for k in 'GSKCMT')
   assert all(v==ref[f'geometry_bits{th}_'+k] for k,v in bits.items())
   for r in rr:
    mx=[float(r[f'{branch}_{name}_maxIoU']) for name in ('initial','pre','final')]
    assert ''.join(str(int(x>=t)) for x in mx)==r[f'{branch}_geometry_bits{th}']
    assert (mx[-1]<t)==(r[f'{branch}_state{th}']=='G')
    assert int(r[f'{branch}_hit{th}'])==int(r[f'{branch}_state{th}']=='T')
    assert float(r[f'{branch}_pre_best_query_final_IoU'])<=mx[-1]
    assert float(r[f'{branch}_initial_best_query_final_IoU'])<=mx[-1]
   pre=[r for r in rr if float(r[f'{branch}_pre_maxIoU'])>=t]
   loss=[r for r in pre if float(r[f'{branch}_final_maxIoU'])<t]
   gain=[r for r in rr if float(r[f'{branch}_pre_maxIoU'])<t and float(r[f'{branch}_final_maxIoU'])>=t]
   details[f'{s}_{branch}_{th}']={'pre_covered':len(pre),'pre_uncovered':len(rr)-len(pre),'pre_to_final_loss':len(loss),'pre_to_final_gain':len(gain),
    'loss_with_pre_at_least_threshold_plus005':sum(float(r[f'{branch}_pre_maxIoU'])>=t+.05 for r in loss),
    'loss_with_pre_margin005_and_final_gap005':sum(float(r[f'{branch}_pre_maxIoU'])>=t+.05 and float(r[f'{branch}_final_maxIoU'])<t-.05 for r in loss),
    'loss_either_endpoint_within_1e4_of_threshold':sum(min(abs(float(r[f'{branch}_pre_maxIoU'])-t),abs(float(r[f'{branch}_final_maxIoU'])-t))<1e-4 for r in loss)}
 for th in (50,75):
  c=Counter((int(r[f'raw_hit{th}']),int(r[f'EMA_hit{th}'])) for r in rr);z=R['same_backend_transitions'][f'{s}_{th}']
  assert [c[0,1],c[0,0],c[1,0],c[1,1]]==[z['recovered'],z['persistent_miss'],z['reverse_loss'],z['both_hit']]
for aid in range(38759):
 a,b=idx[1,aid],idx[2,aid]
 assert [a[k] for k in ('image_id','category_id','size_group')]==[b[k] for k in ('image_id','category_id','size_group')]
 for th in (50,75):
  if all(int(r[f'{br}_hit{th}'])==0 for r in (a,b) for br in ('raw','EMA')):fixed4[th].append(aid)
 if all(r[f'{br}_state50']=='G' for r in (a,b) for br in ('raw','EMA')):all4G.append(aid)
old=list(csv.DictReader(open(P.parent.parent/'persistent_four_snapshots_GT.csv',encoding='utf-8-sig')));oldset={int(r['annotation_id']) for r in old}
out={'status':'PASS','independent_perGT_rows':len(rows),'all_state_geometry_transition_counts_recomputed':True,'pre_to_final':details,
 'same_CPU_four_snapshot_persistent':{str(k):len(v) for k,v in fixed4.items()},'same_CPU_all_four_G50':len(all4G),
 'old_to_CPU_persistent50_ID_set':{'old':len(oldset),'CPU':len(fixed4[50]),'old_only':sorted(oldset-set(fixed4[50])),'CPU_only':sorted(set(fixed4[50])-oldset)},
 'source_sha256':hashlib.sha256((P/'per_GT_stages_and_transitions.csv').read_bytes()).hexdigest(),'forwards':0,'evaluator_calls':0}
with open(P/'independent_counts.json','x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
