"""Independent scalar-rectangle reconstruction of every primary target/draw event."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import statistics
import numpy as np

root=Path(__file__).resolve().parent
out=root/'A10_跨实例负去噪参考框几何审计_v1'
manifest=json.loads((out/'reference_manifest.json').read_text(encoding='utf-8'))
per_image=json.loads((out/'per_image.json').read_text(encoding='utf-8'))
image_index={r['file']:r for r in per_image}
gt_rows=[json.loads(r) for r in (out/'per_gt.jsonl').read_text(encoding='utf-8').splitlines()]
gt_index={r['uid']:r for r in gt_rows}
checks=0
pooled_tiny_events=0
pooled_tiny_draws=0
image_means=[]
for item in manifest['files']:
    path=out/item['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    with np.load(path,allow_pickle=False) as z:
        gt=z['gt_boxes'].astype(float)
        n=len(gt)
        labels=z['gt_labels']
        sizes=z['gt_sizes800']
        refs=z['negative_refs'][:,:n,:]
        classes=z['negative_input_labels'][:,:n]
        expected=z['one_group_same_input_class_iou0.5']
        uids=z['gt_uids']
        actual=np.zeros((16,n),np.uint8)
        for draw in range(16):
            raw=refs[draw]
            # Match the stored float32 coordinate conversion,then use scalar float64 rectangles.
            lower=(raw[:,:2]-raw[:,2:]*np.float32(.5)).astype(float)
            upper=(raw[:,:2]+raw[:,2:]*np.float32(.5)).astype(float)
            for j,(tx1,ty1,tx2,ty2) in enumerate(gt):
                ta=(tx2-tx1)*(ty2-ty1)
                for i in range(n):
                    if i==j or int(classes[draw,i])!=int(labels[j]):
                        continue
                    x1,y1=lower[i]
                    x2,y2=upper[i]
                    iw=min(x2,tx2)-max(x1,tx1)
                    ih=min(y2,ty2)-max(y1,ty1)
                    if iw<=0 or ih<=0:
                        continue
                    inter=iw*ih
                    union=(x2-x1)*(y2-y1)+ta-inter
                    if union>0 and inter/union>=.5:
                        actual[draw,j]=1
                        break
        assert np.array_equal(actual,expected), path.name
        checks+=actual.size
        for j,uid in enumerate(uids):
            assert int(actual[:,j].sum())==gt_index[str(uid)]['exposed_draws']
        tiny=sizes<8
        pooled_tiny_events+=int(actual[:,tiny].sum())
        pooled_tiny_draws+=16*int(tiny.sum())
        if tiny.any():
            val=statistics.fmean(int(v) for v in actual[:,tiny].ravel())
            image_means.append(val)
            assert abs(val-image_index[path.stem+'.jpg']['lt8_one_group_same_input_class_iou0.5'])<1e-12
summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
assert len(manifest['files'])==100 and checks==4491*16
assert abs(statistics.fmean(image_means)-summary['primary']['mean'])<1e-12
receipt={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_ALL_PRIMARY_TARGET_DRAWS_INDEPENDENT_SCALAR_RECTANGLE_CHECK','files_verified':len(manifest['files']),'target_draw_events_verified':checks,'tiny_target_draws':pooled_tiny_draws,'tiny_exposed_target_draws':pooled_tiny_events,'tiny_pooled_rate':pooled_tiny_events/pooled_tiny_draws,'image_mean_reconstructed':statistics.fmean(image_means),'scope':'All primary first-group same-input-class IoU.5 event flags,per-GT counts and image-mean aggregation; secondary thresholds/all-group events not independently re-evaluated. No causal,novelty or training claim.','new_full_model_forwards':0,'new_fits':0}
with (out/'independent_verification.json').open('x',encoding='utf-8') as f: json.dump(receipt,f,ensure_ascii=False,indent=2)
print(json.dumps(receipt,ensure_ascii=False))
