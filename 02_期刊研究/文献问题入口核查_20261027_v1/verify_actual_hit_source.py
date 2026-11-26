"""Resolve GT-hit source identities; correct the earlier 'other query' shorthand."""
import pathlib,json,hashlib,collections
import numpy as np
B=pathlib.Path(__file__).resolve().parent;R=B.parent
C=R/'data/00_实验计划与证据索引/正式132轮基线整图双问题诊断_20261008_v1/native_transform_v2'
P=R/'具体问题重新筛选_20261027_v1'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out={'purpose':'Correct provenance language, no new hypothesis or model/evaluation','correction':'Owner argmax wrong AND GT actually hit does not imply a different query produced the TP. Native flatten permits a non-argmax trueclass pair from the same query.','seeds':{},'inputs':{},'new_forwards':0,'training':0}
for seed in [1,2]:
 fp=B/f'frequency_classification_attempt02/seed{seed}_rows.json';sp=C/f'seed{seed}/prediction_sources.json';mp=P/f'06_seed{seed}_fixed_rank_arrays.npz'
 for f in [fp,sp,mp]:out['inputs'][str(f)]=sha(f)
 rows=read(fp);sources=read(sp);hit={}
 with np.load(mp) as z:
  for c in range(10):
   row=z[f'a0_c{c}_row'];m=z[f'a0_c{c}_matches'][0];ig=z[f'a0_c{c}_ignored'][0]
   for rr,g in zip(row[(m>0)&~ig],m[(m>0)&~ig]):
    assert int(g)-1 not in hit;hit[int(g)-1]=sources[int(rr)]['query']
 result={}
 for name,ids in [('low3',[7,6,8]),('high3',[3,0,9]),('all10',list(range(10)))]:
  rr=[r for r in rows if r['category_id'] in ids and r['owner_iou']>=.5 and r['owner_class']!=r['category_id'] and r['actual_TP50']]
  same=[r for r in rr if hit[r['annotation_id']]==r['owner_query']]
  result[name]={'owner_argmax_wrong_but_actual_TP':len(rr),'TP_same_owner_nonargmax_trueclass':len(same),'TP_different_query':len(rr)-len(same),'GT_ids_same_query':[r['annotation_id'] for r in same],'GT_ids_different_query':[r['annotation_id'] for r in rr if hit[r['annotation_id']]!=r['owner_query']]}
 out['seeds'][str(seed)]=result
for p,h in out['inputs'].items():assert sha(pathlib.Path(p))==h
(B/'05a_实际命中query归属_口径更正.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({s:{k:{f:v for f,v in d.items() if not f.startswith('GT_ids')} for k,d in r.items()} for s,r in out['seeds'].items()}))
