"""Frozen A2 local geometry counterfactuals on existing A0 predictions; no ML fit."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import io
import json
import os
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
ARCHIVE = BASE/'A0_冻结模型诊断结果_v1/backup/A0_A0b_bounded_diagnostics.tar.gz'
SHA = '4a265de93e5b4af230b0020c07c066d02602ea8bcd2d10216ea2493e79a8887d'
RATIOS = [.25,.5,1.,2.,4.]
TEMPLATES = np.array([[s*np.sqrt(r),s/np.sqrt(r)] for s in (4.,8.,16.) for r in RATIOS])

def ious(gt, boxes):
    lower = boxes[:,:2]-boxes[:,2:]*.5
    upper = boxes[:,:2]+boxes[:,2:]*.5
    inter = np.prod(np.maximum(np.minimum(gt[2:],upper)-np.maximum(gt[:2],lower),0),1)
    return inter/np.maximum(np.prod(gt[2:]-gt[:2])+np.prod(boxes[:,2:],1)-inter,1e-12)

def one(gt, boxes):
    if len(boxes)==0:
        return dict(base=0.,center=0.,size=0.,template=0.)
    base = float(ious(gt,boxes).max())
    center = boxes.copy(); center[:,:2] = (gt[:2]+gt[2:])*.5
    size = boxes.copy(); size[:,2:] = gt[2:]-gt[:2]
    template = np.concatenate([boxes[:,:2].repeat(len(TEMPLATES),0),np.tile(TEMPLATES,(len(boxes),1))],1)
    return dict(base=base,center=max(base,float(ious(gt,center).max())),
                size=max(base,float(ious(gt,size).max())),template=max(base,float(ious(gt,template).max())))

def bootstrap(rows,key):
    rows = [r for r in rows if r.get(key) is not None]
    groups = sorted({r['group'] for r in rows})
    if not rows:
        return dict(mean=None,low=None,high=None,images=0,groups=0)
    rng=np.random.default_rng(20261030)
    values=[]
    for _ in range(2000):
        selected=rng.integers(0,len(groups),len(groups))
        values.append(np.mean([r[key] for j in selected for r in rows if r['group']==groups[j]]))
    return dict(mean=float(np.mean([r[key] for r in rows])),low=float(np.quantile(values,.025)),
                high=float(np.quantile(values,.975)),images=len(rows),groups=len(groups))

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    freeze=json.loads((ROOT/'freeze_manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==freeze['script_sha256']
    assert hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()==freeze['protocol_sha256']
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()==SHA
    with (ROOT/'started.json').open('x') as f:
        json.dump(dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat()),f)
    # Known counterfactuals: offset-only and size-only errors must reverse oracle preference.
    gt=np.array([8.,8.,12.,12.])
    offset=one(gt,np.array([[12.,10.,4.,4.]]))
    scale=one(gt,np.array([[10.,10.,8.,8.]]))
    assert offset['center']==1. and offset['size']<.5
    assert scale['size']==1. and scale['center']==.25
    records=[]; verified=[]
    counts=dict(all_gt=0,lt16_gt=0,persistent_lt16_miss50=0,persistent_lt16_with_local=0,persistent_lt16_without_local=0)
    with tarfile.open(ARCHIVE,'r:gz') as archive:
        manifest={r['path']:r for r in json.load(archive.extractfile('backup/file_manifest.json'))}
        paths=sorted(p for p in manifest if p.startswith('traces/') and p.endswith('.npz'))
        assert len(paths)==100
        for index,path in enumerate(paths):
            raw=archive.extractfile(path).read()
            assert hashlib.sha256(raw).hexdigest()==manifest[path]['sha256']
            with np.load(io.BytesIO(raw),allow_pickle=False) as z:
                a={k:z[k].copy() for k in z.files}
            verified.append(dict(path=path,sha256=manifest[path]['sha256']))
            token=a['topk_token_ids']
            anchor_row=(token//80); anchor_col=token%80
            # Only native selected stride8 tokens in the GT-local3x3 anchor neighborhood.
            gt_boxes=a['gt_boxes']
            final=a['decoder_3_boxes'].astype(np.float64)*640.
            initial=a['encoder_topk300_boxes'].astype(np.float64)*640.
            for g,gt in enumerate(gt_boxes):
                c=(gt[:2]+gt[2:])*.5
                col,row=np.floor(c/8).astype(int)
                neighborhood=(token<6400)&(np.abs(anchor_row-row)<=1)&(np.abs(anchor_col-col)<=1)
                init_iou=float(ious(gt,initial).max())
                final_iou=float(ious(gt,final).max())
                primary=(a['gt_sqrt_area'][g]<16 and init_iou<.5 and final_iou<.5 and neighborhood.any())
                counts['all_gt']+=1
                counts['lt16_gt']+=int(a['gt_sqrt_area'][g]<16)
                if a['gt_sqrt_area'][g]<16 and init_iou<.5 and final_iou<.5:
                    counts['persistent_lt16_miss50']+=1
                    counts['persistent_lt16_with_local' if neighborhood.any() else 'persistent_lt16_without_local']+=1
                for stage in ('encoder_topk300','decoder_0','decoder_3'):
                    pred=a[stage+'_boxes'][neighborhood].astype(np.float64)*640.
                    classes=a[stage+'_logits'][neighborhood].argmax(-1)
                    for mode,mask in [('geometry',np.ones(len(pred),bool)),('class_argmax',classes==a['gt_labels'][g])]:
                        result=one(gt,pred[mask])
                        rec=dict(uid=str(a['gt_uids'][g]),file_name=Path(path).stem+'.jpg',group=Path(path).stem.split('_')[0],
                                 sqrt_area=float(a['gt_sqrt_area'][g]),stage=stage,mode=mode,
                                 primary=bool(primary),local_selected_queries=int(neighborhood.sum()),
                                 class_eligible_queries=int(mask.sum()),init_any_iou=init_iou,final_any_iou=final_iou,**result)
                        for name,val in result.items():
                            rec[name+'_rec50']=int(val>=.5); rec[name+'_rec75']=int(val>=.75)
                        records.append(rec)
            if (index+1)%25==0:
                print(json.dumps(dict(status='A2_RUNNING',images=index+1)),flush=True)
    image_rows=[]
    for stage in ('encoder_topk300','decoder_0','decoder_3'):
        for mode in ('geometry','class_argmax'):
            for file in sorted({r['file_name'] for r in records}):
                selected=[r for r in records if r['primary'] and r['stage']==stage and r['mode']==mode and r['file_name']==file]
                if not selected:
                    continue
                row=dict(file_name=file,group=selected[0]['group'],stage=stage,mode=mode,n=len(selected))
                for name in ('base','center','size','template'):
                    row[name+'_rec50']=float(np.mean([r[name+'_rec50'] for r in selected]))
                    row[name+'_rec75']=float(np.mean([r[name+'_rec75'] for r in selected]))
                row['size_minus_center50']=row['size_rec50']-row['center_rec50']
                row['size_minus_template50']=row['size_rec50']-row['template_rec50']
                for name,lo,hi in [('lt8',0,8),('8to16',8,16)]:
                    subset=[r for r in selected if lo<=r['sqrt_area']<hi]
                    row['n_'+name]=len(subset)
                    row['size_minus_center50_'+name]=float(np.mean([r['size_rec50']-r['center_rec50'] for r in subset])) if subset else None
                image_rows.append(row)
    primary_rows=[r for r in image_rows if r['stage']=='encoder_topk300' and r['mode']=='geometry']
    difference=bootstrap(primary_rows,'size_minus_center50')
    bank_difference=bootstrap(primary_rows,'size_minus_template50')
    n=sum(r['n'] for r in primary_rows)
    class_rows=[r for r in image_rows if r['stage']=='encoder_topk300' and r['mode']=='class_argmax']
    class_difference=bootstrap(class_rows,'size_minus_center50')
    size_contrasts={name:bootstrap(primary_rows,'size_minus_center50_'+name) for name in ('lt8','8to16')}
    gates=dict(at_least100gt=n>=100,at_least15groups=difference['groups']>=15,
               size_advantage_at_least010=difference['mean'] is not None and difference['mean']>=.10,
               positive_primary_interval=difference['low'] is not None and difference['low']>0,
               instance_shape_beats_template_bank005=bank_difference['mean'] is not None and bank_difference['mean']>=.05,
               class_aware_contrast_positive=class_difference['mean'] is not None and class_difference['mean']>0,
               both_size_bins_positive=all(v['mean'] is not None and v['mean']>0 for v in size_contrasts.values()))
    summary=dict(status='A2_CACHED_GEOMETRY_DECOMPOSITION_COMPLETE',images=100,source_sha256=SHA,
                 original_gt=len({r['uid'] for r in records}),cohort_counts=counts,primary_gt=n,primary_size_minus_center=difference,
                 primary_size_minus_template_bank=bank_difference,class_aware_size_minus_center=class_difference,
                 size_contrasts=size_contrasts,
                 gates=gates,gate_passed=all(gates.values()),
                 stage_table=[dict(stage=s,mode=m,**{name:float(np.mean([r[name] for r in image_rows if r['stage']==s and r['mode']==m])) for name in
                            ['base_rec50','center_rec50','size_rec50','template_rec50','base_rec75','center_rec75','size_rec75','template_rec75']})
                            for s in ('encoder_topk300','decoder_0','decoder_3') for m in ('geometry','class_argmax')],
                 new_full_model_forwards=0,new_readout_fits=0,main_model_optimizer_updates=0,
                 scope='GT-oracle component corrections on fixed locally associated native queries; not deployable detection,AP,causal mechanism proof or template-bank algorithm evaluation',
                 limitations=['Local anchor association is not true object identity.','Native selected-only head avoids unselected-token supervision confound but excludes GT with no nearby selected query.',
                              'Center correction and size correction optionally retain the original box; max coverage is many-to-one and ignores scores.',
                              'Template bank is an oracle expansion of15 shapes per query and does not keep candidate compute equal; it is a strong explanatory control only.',
                              'Primary cohort uses final failure and initial failure; conditional exploration,not independent generalization.'])
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (ROOT/'per_image.json').write_text(json.dumps(image_rows,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (ROOT/'per_gt.jsonl').open('w',encoding='utf-8') as f:
        for row in records:
            f.write(json.dumps(row,allow_nan=False)+'\n')
    (ROOT/'input_trace_verification.json').write_text(json.dumps(verified,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,allow_nan=False),flush=True)

if __name__=='__main__':
    main()
