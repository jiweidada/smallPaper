"""Independent scalar geometry/count/order-bound checks and input-hash verification."""
import json, csv, hashlib, collections, time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def scalar(a,b):
    xx=max(0.,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]));yy=max(0.,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
    inter=xx*yy
    return inter/max(a[2]*a[3]+b[2]*b[3]-inter,1e-300)
sample=read(HERE/'03_漏标审计_样本与总体.json');cross=read(HERE/'06_跨图排序_完整结果.json')
g=read(Path('C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val/annotations/val.json'))
by=collections.defaultdict(list)
for a in g['annotations']:by[a['image_id']].append(a)
view={r['case_id']:r for r in csv.DictReader((HERE/'07_原像素首轮观察.csv').open(encoding='utf-8'))}
assert set(view)=={v['case_id'] for v in sample['cases']}
check={'sample':{},'scalar_case_max_IoU':{},'rank_order_bounds':{},'legacy_FP_pool_parity':{},'source_hashes':{}}
geometry=read(HERE/'07_首轮观察与原始GT几何对照.json')
mxerr=0.;n=0
for r in geometry['all_cases']:
    v=next(v for v in sample['cases'] if v['case_id']==r['case_id'])
    vals=[scalar(v['bbox'],a['bbox']) for a in by[v['image_id']]]
    err=abs(max(vals,default=0.)-r['max_GT_IoU']);mxerr=max(mxerr,err);n+=len(vals)
assert mxerr<=1e-12
check['scalar_case_max_IoU']={'cases':len(sample['cases']),'scalar_box_pairs':n,'max_abs_difference':mxerr,'pass':True}
pool=read(HERE/'03_漏标审计_完整合格池.json')
for s in [1,2]:
    vals=[v for v in sample['cases'] if v['review_role']=='candidate' and v['seed']==s]
    counts=collections.Counter(view[v['case_id']]['pixel_reading'] for v in vals)
    assert len(vals)==40 and all(v['image_id'] not in [51,263,428,512] for v in vals)
    for c in range(10):
        for b in ['025_050','050_100']:
            assert len([v for v in vals if v['category_id']==c and v['score_band']==b])==2
    check['sample'][str(s)]={'selected_occurrences':len(vals),'images':len({v['image_id'] for v in vals}),'visible_object_V':counts['V'],'uncertain_U':counts['U'],'no_clear_target_N':counts['N']}
    old=ROOT/f'data/00_实验计划与证据索引/逐类别AP漏检误检与标注复核_20261008_v1/seed{s}_误检逐条.json'
    oldrows=read(old)
    ids={v['prediction_row'] for v in oldrows if v['same_class_max_IoU']<.1 and v['other_class_max_IoU']<.1 and v['raw_0_max_detection_IoA']<.5 and v['raw_11_max_detection_IoA']<.5}
    new={v['prediction_row'] for v in pool['available_pool']+pool['old_anchor_pool'] if v['seed']==s}
    assert ids==new,(s,len(ids),len(new),sorted(ids^new)[:20])
    check['legacy_FP_pool_parity'][str(s)]={'source':str(old),'sha256':sha(old),'old_n':len(ids),'new_n':len(new),'symmetric_difference':len(ids^new)}
    z=np.load(HERE/f'06_seed{s}_fixed_rank_arrays.npz')
    scores=z['scores'];logits=np.log(scores)-np.log1p(-scores);means=z['image_mean_logits'];ii=z['image_ids']
    minpositive=float('inf')
    for iid in np.unique(ii):
        gaps=np.diff(np.sort(logits[ii==iid]));positive=gaps[gaps>0]
        if len(positive):minpositive=min(minpositive,float(positive.min()))
    bound=8*np.finfo(float).eps*(float(np.abs(logits).max())+float(np.abs(means).max())+1)
    assert minpositive>2*bound
    # All saved image offsets are scalar per image, so this gap bound protects every null order; exact ties subtract the same scalar.
    for arm in ['unconditional','conditional']:
        maps=z[f'{arm}_permutations'];assert maps.shape==(199,548)
        assert all(np.array_equal(np.sort(p),np.arange(548)) for p in maps)
        if arm=='conditional':assert np.all(z['strata'][maps]==z['strata'][None,:])
    tp={};fp={}
    for ti in [0,5]:
        tp[str(50+ti*5)]=sum(int(np.count_nonzero((z[f'a0_c{c}_matches'][ti]>0)&~z[f'a0_c{c}_ignored'][ti])) for c in range(10))
        fp[str(50+ti*5)]=sum(int(np.count_nonzero((z[f'a0_c{c}_matches'][ti]==0)&~z[f'a0_c{c}_ignored'][ti])) for c in range(10))
    for th in ['50','75']:
        assert tp[th]==cross['seed_results'][str(s)]['counts_unchanged_all_arms'][th]['TP']
        assert fp[th]==cross['seed_results'][str(s)]['counts_unchanged_all_arms'][th]['FP']
    high_tp=high_fp=0
    for c in range(10):
        ids=z[f'a0_c{c}_row'];m=z[f'a0_c{c}_matches'][0];ig=z[f'a0_c{c}_ignored'][0];high=scores[ids]>=.25
        high_tp+=int(np.count_nonzero(high&(m>0)&~ig));high_fp+=int(np.count_nonzero(high&(m==0)&~ig))
    check['rank_order_bounds'][str(s)]={'minimum_positive_logit_gap':minpositive,'conservative_numeric_bound':bound,
        'all199x2_permutation_orders_protected':True,'TP':tp,'FP':fp,'score025_TP50':high_tp,'score025_FP50':high_fp}
all_sources=sample['sources']|cross['sources']
for p,h in all_sources.items():check['source_hashes'][p]={'expected':h,'actual':sha(Path(p))}
assert all(v['expected']==v['actual'] for v in check['source_hashes'].values())
assert sha(HERE/'02_漏标审计_执行前协议.json')==(HERE/'02_漏标审计_执行前协议.sha256').read_text().strip()
assert sha(HERE/'05_跨图排序_执行前协议.json')==(HERE/'05_跨图排序_执行前协议.sha256').read_text().strip()
check['status']='PASS';check['boundary']='Not a full independent rerun of COCO/permutations or gold-standard manual annotation. Legacy source equality is cross-checking earlier stored evaluator evidence.'
(HERE/'10_独立算术与来源核验.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','sample':check['sample'],'scalar':check['scalar_case_max_IoU'],'rank_checks':check['rank_order_bounds'],'FP_pool_parity':{s:{k:v[k] for k in ['old_n','new_n','symmetric_difference']} for s,v in check['legacy_FP_pool_parity'].items()},'source_files':len(all_sources)},ensure_ascii=False,indent=2))
