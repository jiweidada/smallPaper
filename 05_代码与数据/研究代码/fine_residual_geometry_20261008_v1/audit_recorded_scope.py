"""Join existing fixed-cohort GO identities to recorded topk token indices only."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
INDEX=HERE.parents[1]/'data/00_实验计划与证据索引'
LOOKUP=INDEX/'固定目标回归覆盖归档核对_20261008_v1'
ASSIGN=INDEX/'固定普通样本双目标分配核查_20261008_v1'
BASE=INDEX/'正式132轮基线整图双问题诊断_20261008_v1/native_transform_v2'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=LOOKUP/'02_原固定目标的已存监督身份.json'
    prior=read(LOOKUP/'03_来源绑定与执行回执.json')
    assert sha(p)==prior['result_sha256']
    cases=read(p)['cases']
    topk={};source_hashes={};rows=[]
    manifests={s:{v['file']:v['sha256'] for v in read(BASE/f'seed{s}/array_manifest.json')} for s in (1,2)}
    for row in cases:
        s,i=row['seed'],row['image_id']
        key=(s,i)
        if key not in topk:
            a=BASE/f'seed{s}/arrays/{i:06d}.npz'
            h=sha(a);assert h==manifests[s][a.name]
            b=ASSIGN/f'seed{s}/image_{i:06d}_assignments.npz'
            assert sha(b)==prior['verified_archived_inputs'][f'seed{s}/{b.name}']
            with np.load(a,allow_pickle=False) as z, np.load(b,allow_pickle=False) as recorded:
                ids=z['topk_ind']
                assert ids.shape==(300,) and len(np.unique(ids))==300
                assert np.all((ids>=0)&(ids<8400))
                assert np.array_equal(z['enc_box'],recorded['stage_boxes'][0])
                topk[key]=ids.copy()
            source_hashes[str(a)]=h
        ids=topk[key]
        recorded_queries=row['recorded_GO_queries']
        levels=[3 if ids[q]<6400 else 4 if ids[q]<8000 else 5 for q in recorded_queries]
        rows.append(dict(seed=s,image_id=i,annotation_id=row['annotation_id'],
                         group=row['group'],role=row['role'],
                         recorded_GO_queries=recorded_queries,recorded_GO_token_ids=[int(ids[q]) for q in recorded_queries],
                         recorded_GO_levels=levels))
    summary={}
    for s in (1,2):
        summary[str(s)]={}
        for group,role in (('N','failure'),('N','control'),('C','failure'),('C','control')):
            part=[r for r in rows if r['seed']==s and r['group']==group and r['role']==role]
            summary[str(s)][group+'_'+role]=dict(fixed_targets=len(part),
                has_recorded_P3_GO=sum(3 in r['recorded_GO_levels'] for r in part),
                only_coarse_GO=[r['annotation_id'] for r in part if r['recorded_GO_levels'] and 3 not in r['recorded_GO_levels']],
                no_GO=[r['annotation_id'] for r in part if not r['recorded_GO_levels']])
    result=dict(status='RECORDED_IDENTITY_JOIN_COMPLETED',summary=summary,cases=rows,
                fixed_targets_per_seed=157,fixed_images_per_seed=73,
                source_hashes=source_hashes,prior_identity_result_sha256=sha(p),
                detector_calls=0,matcher_calls=0,parameter_access=0,
                limitation='Validation endpoint scope only, not training histories, feature sufficiency, or recovered detections.')
    with (HERE/'04_归档作用对象的层级核对.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(status=result['status'],summary=summary,verified_array_count=len(source_hashes)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
