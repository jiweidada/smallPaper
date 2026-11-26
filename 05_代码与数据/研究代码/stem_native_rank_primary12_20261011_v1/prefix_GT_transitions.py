"""Describe same-GT native TP-prefix transitions; no cross-run detection alignment."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
SOURCE=ROOT/'outputs/stem_native_rank_primary12_20261011_v1'
OUT=ROOT/'outputs/stem_native_rank_GT_prefix12_20261011_v1'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def analyze():
    assert not OUT.exists()
    report=read(SOURCE/'report.json')
    assert report['status']=='ALL_FOUR_PRIMARY_NATIVE_SCORE_ORDERS_AND10100_PR_CELLS_REPRODUCED_EXACTLY'
    for n,h in read(SOURCE/'manifest.json')['files'].items():assert sha(SOURCE/n)==h
    assert report['maximum_abs_PR_difference']==0 and report['independent_saved_EMA_verification_pending']
    assert sha(HERE/'analyze.py')==report['source_sha256']
    seeds=[]
    for seed in (1,2):
        f=load(SOURCE/f'seed{seed}_full_GT_order.npz');r=load(SOURCE/f'seed{seed}_RGB_GT_order.npz')
        fd=read(SOURCE/f'seed{seed}_full_order.json');rd=read(SOURCE/f'seed{seed}_RGB_order.json')
        for key in ('annotation_id','category_id'):assert np.array_equal(f[key],r[key])
        classes=[]
        for cid in range(10):
            mask=f['category_id']==cid;thresholds=[]
            for ti in range(10):
                fr=f['native_true_match_score_rank'][mask,ti];rr=r['native_true_match_score_rank'][mask,ti]
                common=(fr>0)&(rr>0);prefixes=[]
                fp={x['rank']:x for x in fd['classes'][cid]['by_IoU'][ti]['prefixes']}
                rp={x['rank']:x for x in rd['classes'][cid]['by_IoU'][ti]['prefixes']}
                for n in (100,500,1000,2000,5000,10000,20000):
                    if n not in fp or n not in rp:continue
                    ef=(fr>0)&(fr<=n);er=(rr>0)&(rr<=n)
                    row=dict(rank=n,new_native_TP_in_full_prefix=int((ef&(rr<0)).sum()),
                        lost_native_TP_in_RGB_prefix=int((er&(fr<0)).sum()),
                        common_GT_entering_full_prefix=int((common&ef&~er).sum()),
                        common_GT_leaving_full_prefix=int((common&er&~ef).sum()),
                        TP_prefix_delta=int(ef.sum()-er.sum()),full_TP=int(ef.sum()),RGB_TP=int(er.sum()))
                    assert row['full_TP']==fp[n]['TP'] and row['RGB_TP']==rp[n]['TP']
                    assert row['new_native_TP_in_full_prefix']-row['lost_native_TP_in_RGB_prefix']+row['common_GT_entering_full_prefix']-row['common_GT_leaving_full_prefix']==row['TP_prefix_delta']
                    prefixes.append(row)
                thresholds.append(dict(IoU=fd['classes'][cid]['by_IoU'][ti]['IoU'],prefixes=prefixes))
            classes.append(dict(class_id=cid,name=fd['classes'][cid]['name'],by_IoU=thresholds))
        seeds.append(dict(seed=seed,classes=classes))
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    OUT.mkdir();result=dict(status='SAME_GT_NATIVE_SCORE_PREFIX_TRANSITIONS_ACCOUNTED_EXACTLY',seeds=seeds,
        source_sha256=sha(__file__),source_rank_report_sha256=sha(SOURCE/'report.json'),source_rank_manifest_sha256=sha(SOURCE/'manifest.json'),
        all10_categories_all10_IoUs_both_seeds_retained=True,independent_saved_EMA_verification_pending=True,
        causal_root_proven=False,new_model_forwards=0,new_evaluator_calls=0,new_matching_calls=0,new_optimizer_updates=0,new_training_epochs=0,
        scope='An exact count identity for saved native TP-prefix changes, conditioned on same-GT native match outcomes. '
        'Matched boxes/queries are not aligned across runs. Common matched-GT rank changes can reflect geometry, '
        'classification, candidate retention and native greedy matching together; they do not prove a classification-head '
        'or Stem cause. This is exploratory output accounting, not an AP causal decomposition or efficacy test.')
    write(OUT/'report.json',result)
    lines=['相同GT的原生分数前段变化（主记录，独立EMA复核待完成）','',
        '两种子、10类、10个IoU均保留。前段TP差额 = 新增TP进入前段 − 丢失TP原在前段 + 两组均正确检出的GT进入前段 − 两组均正确检出的GT离开前段。',
        '以下展示car / IoU0.75；全部类别和阈值在report.json。','']
    for seed in seeds:
        lines.append(f"seed{seed['seed']}：前N | 新增TP在增强前段 | 丢失TP原在RGB前段 | 两组都检出的GT进入/离开前段 | TP净变化")
        for row in seed['classes'][3]['by_IoU'][5]['prefixes']:
            lines.append(f"{row['rank']} | {row['new_native_TP_in_full_prefix']} | {row['lost_native_TP_in_RGB_prefix']} | {row['common_GT_entering_full_prefix']}/{row['common_GT_leaving_full_prefix']} | {row['TP_prefix_delta']:+d}")
        lines.append('')
    lines+=['seed1前1000个预测的TP净减29：新增/丢失检出贡献+1；两组均检出的GT在前1000位置的进入/离开净差为−30。',
        '这比“新增正确框都排得晚”更具体，但只是按GT对齐的输出账目。两次运行对应的检测框、查询、匹配框可能不同，不能把它当成同一查询被某层移动的证据。',
        'seed2前1000个预测净增22：新增/丢失检出贡献+5，均检出GT前段变动贡献+17。没有稳定的同向负效果。']
    (OUT/'GT前段变化证据.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    write(OUT/'manifest.json',dict(files={x.name:sha(x) for x in OUT.iterdir() if x.is_file()}))
    print(json.dumps(dict(status=result['status'],all_categories=10,seeds=[1,2],IoUs=10)))

if __name__=='__main__':analyze()
