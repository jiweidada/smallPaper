"""One descriptive many-to-many GT coverage pass on saved predictions only."""
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import hashlib
import json
import numpy as np

P=Path(__file__).resolve().parent
OUT=P/'saved_B4_geometry_coverage_v1'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def summarize(mask,any_iou,same_iou):
    n=int(mask.sum());a=any_iou[mask];s=same_iou[mask]
    counts=dict(same_class_IoU_ge_05=int((s>=.5).sum()),
        any_class_IoU_ge_05_but_same_class_below_05=int(((a>=.5)&(s<.5)).sum()),
        any_class_IoU_01_to_05=int(((a>=.1)&(a<.5)).sum()),any_class_IoU_below_01=int((a<.1).sum()))
    assert sum(counts.values())==n
    return dict(GT=n,partition_counts=counts,partition_percent={k:100*v/n if n else None for k,v in counts.items()},
        same_class_IoU_ge_075=int((s>=.75).sum()),any_class_IoU_ge_05=int((a>=.5).sum()),
        any_class_IoU_ge_075=int((a>=.75).sum()),median_best_same_class_IoU=float(np.median(s)) if n else None,
        median_best_any_class_IoU=float(np.median(a)) if n else None)

def main():
    scope=json.loads((P/'saved_B4_geometry_coverage_scope_FROZEN_v1.json').read_text())
    assert sha(Path(__file__))==scope['script_sha256'] and not OUT.exists()
    for name,h in scope['input_sha256'].items():assert sha(P/name)==h,name
    gt=json.loads((P/scope['GT_file']).read_text());pred=json.loads((P/scope['predictions_file']).read_text())
    assert len(gt['annotations'])==38602 and len(pred)==78414
    by_gt=defaultdict(list);by_pred=defaultdict(list)
    for v in gt['annotations']:by_gt[v['image_id']].append(v)
    for v in pred:by_pred[v['image_id']].append(v)
    rows=[];pairs=0
    for im in gt['images']:
        targets=by_gt[im['id']];proposals=by_pred[im['id']]
        gb=np.array([v['bbox'] for v in targets],dtype=np.float64).reshape(-1,4)
        pb=np.array([v['bbox'] for v in proposals],dtype=np.float64).reshape(-1,4)
        gl=np.array([v['category_id'] for v in targets]);pl=np.array([v['category_id'] for v in proposals])
        pairs+=len(gb)*len(pb)
        if len(gb) and len(pb):
            lower=np.maximum(gb[:,None,:2],pb[None,:,:2])
            upper=np.minimum(gb[:,None,:2]+gb[:,None,2:],pb[None,:,:2]+pb[None,:,2:])
            sides=np.maximum(upper-lower,0);intersection=sides[...,0]*sides[...,1]
            union=(gb[:,2]*gb[:,3])[:,None]+(pb[:,2]*pb[:,3])[None,:]-intersection
            iou=intersection/union
            best_any=iou.max(axis=1);best_same=np.where(gl[:,None]==pl[None,:],iou,0).max(axis=1)
        else:best_any=np.zeros(len(gb));best_same=np.zeros(len(gb))
        assert np.isfinite(best_any).all() and np.isfinite(best_same).all()
        assert (best_same<=best_any+1e-12).all()
        for i,target in enumerate(targets):
            x,y,w,h=target['bbox'];short=min(w*640/im['width'],h*640/im['height'])
            rows.append((target['id'],im['id'],target['category_id'],target['area'],short,best_any[i],best_same[i]))
    values=np.asarray(rows,dtype=np.float64);assert len(values)==38602
    areas=values[:,3];short=values[:,4];any_iou=values[:,5];same_iou=values[:,6]
    small=areas<=1024;very_tiny=small&(short<=4);assert int(small.sum())==26444
    groups=dict(all=summarize(np.ones(len(values),dtype=bool),any_iou,same_iou),
        small=summarize(small,any_iou,same_iou),very_tiny=summarize(very_tiny,any_iou,same_iou))
    classes={str(c):summarize(small&(values[:,2]==c),any_iou,same_iou) for c in range(1,11)}
    OUT.mkdir();save(OUT/'run_once.json',dict(scope_sha256=sha(P/'saved_B4_geometry_coverage_scope_FROZEN_v1.json'),new_model_forwards=0,new_optimizer_updates=0))
    np.savez_compressed(OUT/'per_GT_coverage.npz',GT_id=values[:,0].astype(np.int64),image_id=values[:,1].astype(np.int64),
        category_id=values[:,2].astype(np.int64),original_area=areas,input_short_side=short,best_any_class_IoU=any_iou,best_same_class_IoU=same_iou)
    result=dict(status='COMPLETED_SAVED_B4_OPTIMISTIC_GEOMETRY_COVERAGE',recorded_client_utc=datetime.now(timezone.utc).isoformat(),
        scope_sha256=sha(P/'saved_B4_geometry_coverage_scope_FROZEN_v1.json'),input_sha256=scope['input_sha256'],
        images=548,predictions=78414,GT=38602,GT_prediction_IoU_pairs=pairs,groups=groups,small_by_class=classes,
        new_model_forwards=0,new_optimizer_updates=0,new_COCO_evaluations=0,new_image_reads=0,
        causal_error_decomposition=False,formal_TIDE_analysis=False,AP_or_AR_recomputation=False,
        limitations=scope['limitations'],training_protocol_or_gates_changed=False)
    save(OUT/'result.json',result)
    description='B4保存预测的几何覆盖画像\n仅使用已保存GT与预测，0新增模型推理/更新/COCO重评。各GT独立选择最佳IoU，可共享同一预测，因此是乐观候选覆盖，不是TP/召回/AP或正式TIDE分析。\n'
    for name,row in groups.items():
        description+=f"\n{name}：{row['GT']}个GT\n"
        for key,count in row['partition_counts'].items():description+=f"{key}: {count} ({row['partition_percent'][key]:.3f}%)\n"
        description+=f"同类框IoU≥0.75: {row['same_class_IoU_ge_075']}\n"
    description+='\n解释边界：低覆盖不区分编码器信息、训练成熟度、定位路径、置信度筛选等原因；类别缺口也不等于模型仅有分类错误。不能直接把实例加权覆盖率与跨类别/IoU平均的AP相减作因果分解。当前统计不能证明几何时刻是漏检原因，也不能替代Q对M/B的固定12轮验证。实验与门槛不变。\n'
    with (OUT/'01_覆盖画像与解释边界.txt').open('x',encoding='utf-8') as f:f.write(description)
    print(json.dumps(dict(status=result['status'],groups=groups,pairs=pairs,new_forwards=0,new_updates=0,new_COCO_evaluations=0),ensure_ascii=False))

if __name__=='__main__':main()
