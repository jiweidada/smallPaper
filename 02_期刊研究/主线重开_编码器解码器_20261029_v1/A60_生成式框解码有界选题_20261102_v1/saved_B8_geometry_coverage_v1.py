"""One predeclared paired-GT descriptive pass on B8 saved predictions."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
import numpy as np
from saved_B4_geometry_coverage_v1 import summarize

P=Path(__file__).resolve().parent
OUT=P/'saved_B8_geometry_coverage_v1'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def transitions(mask,old_iou,new_iou):
    old=old_iou[mask]>=.5;new=new_iou[mask]>=.5;n=len(old)
    counts=dict(covered_both=int((old&new).sum()),gained_coverage=int((~old&new).sum()),
        lost_coverage=int((old&~new).sum()),uncovered_both=int((~old&~new).sum()))
    assert sum(counts.values())==n
    return dict(GT=n,counts=counts,percent={k:100*v/n if n else None for k,v in counts.items()},
        B4_no_any_coverage_percent=100*int((~old).sum())/n if n else None,
        B8_no_any_coverage_percent=100*int((~new).sum())/n if n else None,
        B8_minus_B4_no_any_coverage_pp=100*(int((~new).sum())-int((~old).sum()))/n if n else None)

def main():
    scope=json.loads((P/'saved_B8_geometry_coverage_scope_FROZEN_v1.json').read_text())
    assert sha(Path(__file__))==scope['script_sha256'] and not OUT.exists()
    for name,h in scope['input_sha256'].items():assert sha(P/name)==h,name
    gt=json.loads((P/scope['GT_file']).read_text());pred=json.loads((P/scope['predictions_file']).read_text())
    assert len(gt['annotations'])==38602 and len(pred)==79938
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
        assert np.isfinite(best_any).all() and np.isfinite(best_same).all() and (best_same<=best_any+1e-12).all()
        for i,target in enumerate(targets):
            x,y,w,h=target['bbox'];short=min(w*640/im['width'],h*640/im['height'])
            rows.append((target['id'],im['id'],target['category_id'],target['area'],short,best_any[i],best_same[i]))
    values=np.asarray(rows,dtype=np.float64);assert len(values)==38602
    areas=values[:,3];short=values[:,4];any_iou=values[:,5];same_iou=values[:,6]
    small=areas<=1024;very_tiny=small&(short<=4);assert int(small.sum())==26444 and int(very_tiny.sum())==5361
    with np.load(P/scope['B4_per_GT_file']) as old:
        assert np.array_equal(old['GT_id'],values[:,0].astype(np.int64))
        assert np.array_equal(old['image_id'],values[:,1].astype(np.int64))
        assert np.array_equal(old['category_id'],values[:,2].astype(np.int64))
        assert np.array_equal(old['original_area'],areas) and np.array_equal(old['input_short_side'],short)
        b4_any=old['best_any_class_IoU'].copy();b4_same=old['best_same_class_IoU'].copy()
    masks=dict(all=np.ones(len(values),dtype=bool),small=small,very_tiny=very_tiny)
    groups={name:summarize(mask,any_iou,same_iou) for name,mask in masks.items()}
    paired={name:transitions(mask,b4_any,any_iou) for name,mask in masks.items()}
    small_classes={str(c):summarize(small&(values[:,2]==c),any_iou,same_iou) for c in range(1,11)}
    OUT.mkdir();save(OUT/'run_once.json',dict(scope_sha256=sha(P/'saved_B8_geometry_coverage_scope_FROZEN_v1.json'),new_model_forwards=0,new_optimizer_updates=0))
    np.savez_compressed(OUT/'per_GT_coverage.npz',GT_id=values[:,0].astype(np.int64),image_id=values[:,1].astype(np.int64),
        category_id=values[:,2].astype(np.int64),original_area=areas,input_short_side=short,
        best_any_class_IoU=any_iou,best_same_class_IoU=same_iou)
    result=dict(status='COMPLETED_SAVED_B8_PAIRED_GT_GEOMETRY_COVERAGE',recorded_client_utc=datetime.now(timezone.utc).isoformat(),
        scope_sha256=sha(P/'saved_B8_geometry_coverage_scope_FROZEN_v1.json'),input_sha256=scope['input_sha256'],
        images=548,predictions=79938,GT=38602,GT_prediction_IoU_pairs=pairs,groups=groups,B4_B8_paired_GT_transitions=paired,
        small_by_class=small_classes,new_model_forwards=0,new_optimizer_updates=0,new_COCO_evaluations=0,new_image_reads=0,
        causal_error_decomposition=False,formal_TIDE_analysis=False,AP_or_AR_recomputation=False,
        limitations=scope['limitations'],training_protocol_or_gates_changed=False)
    save(OUT/'result.json',result)
    text='B8保存预测的几何覆盖画像与B4配对比较\n两次GT逐项一致；每个GT独立取最佳预测，同一预测可复用。该统计是乐观候选覆盖，不是标准TP/召回/AP或正式TIDE分析。\n'
    for name,row in paired.items():
        text+=f"\n{name}: {row['GT']}个GT；任意类别均无IoU≥0.5保留预测：B4 {row['B4_no_any_coverage_percent']:.3f}%，B8 {row['B8_no_any_coverage_percent']:.3f}%，B8−B4 {row['B8_minus_B4_no_any_coverage_pp']:+.3f}个百分点。\n"
        text+='配对转移：'+json.dumps(row['counts'],ensure_ascii=False)+'\n'
    text+='\n覆盖变化与权重训练阶段一起发生，不能归因于几何时刻；B8的缺口也可能与编码器、候选框、分类置信度和后处理有关。M/Q尚未训练，B12门槛不变。0新增模型推理、优化器更新、COCO重评或原始图像读取。\n'
    with (OUT/'01_配对覆盖画像与边界.txt').open('x',encoding='utf-8') as f:f.write(text)
    print(json.dumps(dict(status=result['status'],groups=groups,paired=paired,pairs=pairs,new_forwards=0,new_updates=0,new_COCO_evaluations=0),ensure_ascii=False))

if __name__=='__main__':main()
