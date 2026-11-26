"""Conditional evaluator-penalty sensitivity, NOT corrected AP or new GT."""
import json, hashlib, collections
from pathlib import Path
import numpy as np
from cross_image_probe import accumulate, overlap
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
BASE=ROOT/'data/00_实验计划与证据索引/正式132轮基线整图双问题诊断_20261008_v1/native_transform_v2'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
proto=HERE/'09_固定案例评价敏感性_事前协议.json';ph=sha(proto)
(HERE/'09_固定案例评价敏感性_事前协议.sha256').write_text(ph,encoding='utf-8')
sample=read(HERE/'03_漏标审计_样本与总体.json');ref={v['case_id']:v for v in sample['cases']}
arms={'strong3':['R015','R026','R086'],'strong3_plus_boundary1':['R015','R026','R086','R061']}
result={'protocol_sha256':ph,'meaning':'Conditional FP-penalty sensitivity only; not corrected GT/AP or model improvement','seeds':{},'source_hashes':{str(proto):ph,str(Path(__file__).resolve()):sha(Path(__file__).resolve()),str(HERE/'cross_image_probe.py'):sha(HERE/'cross_image_probe.py')}}
for s in [1,2]:
    p=BASE/f'seed{s}/predictions.json';pred=read(p);zp=HERE/f'06_seed{s}_fixed_rank_arrays.npz';z=np.load(zp)
    result['source_hashes'][str(p)]=sha(p);result['source_hashes'][str(zp)]=sha(zp)
    data={}
    for a in range(4):
        for c in range(10):
            data[(a,c)]={k:z[f'a{a}_c{c}_{k}'] for k in ['row','matches','ignored','npig']}
    sr={'native':accumulate(data,z['scores'])[0],'arms':{}}
    for arm,cids in arms.items():
        affected=[];records=[]
        for cid in cids:
            v=ref[cid];c=v['category_id'];x=data[(0,c)];cand=[int(k) for k in x['row'] if pred[int(k)]['image_id']==v['image_id']]
            I=overlap([v['bbox']],[pred[k]['bbox'] for k in cand])[0]
            ids=[k for k,u in zip(cand,I) if u>=.5]
            for k in ids:
                pos=int(np.flatnonzero(x['row']==k)[0]);assert not(x['matches'][:,pos]>0).any();assert not x['ignored'][:,pos].any()
            affected.extend(ids);records.append({'case':cid,'image_id':v['image_id'],'class':c,'affected_rows':ids})
        mask=np.ones(len(pred),bool);mask[np.unique(affected)]=False
        metric,_=accumulate(data,z['scores'],mask)
        sr['arms'][arm]={'metrics':metric,'delta_pp':{k:(metric[k]-sr['native'][k])*100 for k in metric},
            'removed_actual_FP_rows':len(set(affected)),'original_cap100_rows':sum(len(data[(0,c)]['row']) for c in range(10)),
            'unchanged_TP50':sum(int(np.count_nonzero(data[(0,c)]['matches'][0]>0)) for c in range(10)),
            'cases':records}
    result['seeds'][str(s)]=sr
result['seed2_minus_seed1_AP_pp']={'native':100*(result['seeds']['2']['native']['AP']-result['seeds']['1']['native']['AP'])}
for arm in arms:
    result['seed2_minus_seed1_AP_pp'][arm]=100*(result['seeds']['2']['arms'][arm]['metrics']['AP']-result['seeds']['1']['arms'][arm]['metrics']['AP'])
result['AP_order_reversed']=any(v<=0 for v in result['seed2_minus_seed1_AP_pp'].values())
assert sha(proto)==ph
assert all(sha(Path(p))==h for p,h in result['source_hashes'].items())
with (HERE/'09_固定案例评价敏感性_结果.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({'seed_results':{s:{a:{k:v[k] for k in ['delta_pp','removed_actual_FP_rows','original_cap100_rows','unchanged_TP50']} for a,v in sr['arms'].items()} for s,sr in result['seeds'].items()},'seed2_minus_seed1_AP_pp':result['seed2_minus_seed1_AP_pp'],'AP_order_reversed':result['AP_order_reversed']},ensure_ascii=False,indent=2))
