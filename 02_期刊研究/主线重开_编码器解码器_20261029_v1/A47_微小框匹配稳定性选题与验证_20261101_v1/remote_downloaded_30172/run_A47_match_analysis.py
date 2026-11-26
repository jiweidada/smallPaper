"""A47 prespecified matching diagnostic; executed on frozen exported final outputs."""
from pathlib import Path
import json, hashlib, csv, time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

def xyxy(boxes):
    return torch.cat((boxes[:, :2]-boxes[:, 2:]/2, boxes[:, :2]+boxes[:, 2:]/2), dim=1)

def paired_iou(pred, target):
    a,b=xyxy(pred),xyxy(target)
    inter=(torch.minimum(a[:,2:],b[:,2:])-torch.maximum(a[:,:2],b[:,:2])).clamp(min=0).prod(-1)
    area_a=(a[:,2:]-a[:,:2]).prod(-1)
    area_b=(b[:,2:]-b[:,:2]).prod(-1)
    return inter/(area_a+area_b-inter)

def nwd_assignment(logits, boxes, target, matcher):
    prob=logits.sigmoid()[:, target["labels"]]
    neg=(1-matcher.alpha)*prob.pow(matcher.gamma)*(-(1-prob+1e-8).log())
    pos=matcher.alpha*(1-prob).pow(matcher.gamma)*(-(prob+1e-8).log())
    cls=pos-neg
    l1=torch.cdist(boxes, target["boxes"], p=1)
    a=boxes[:,None,:]*640
    b=target["boxes"][None,:,:]*640
    delta=(a-b)*a.new_tensor([1,1,.5,.5])
    nwd=torch.exp(-torch.linalg.vector_norm(delta,dim=-1)/12.8)
    cost=matcher.cost_class*cls+matcher.cost_bbox*l1-matcher.cost_giou*nwd
    assert torch.isfinite(cost).all()
    q,g=linear_sum_assignment(cost.numpy())
    owner=np.empty(len(g),dtype=np.int64);owner[g]=q
    return owner

def native_assignment(logits, boxes, target, matcher):
    got=matcher({"pred_logits":logits[None], "pred_boxes":boxes[None]},[target])["indices"][0]
    q,g=[x.numpy() for x in got]
    owner=np.empty(len(g),dtype=np.int64);owner[g]=q
    return owner

def query_labels(owner, labels, q=300):
    lab=np.full(q,-1,dtype=np.int64)
    lab[owner]=labels
    return lab

def soft_targets(owner, labels, pred, target):
    out=np.zeros((300,10),dtype=np.float32)
    quality=paired_iou(pred[owner],target).clamp(0,1).pow(1.5).numpy()
    out[owner,labels]=quality
    return out

def perturb(boxes, width, height, noise, amplitude):
    edges=xyxy(boxes).numpy().astype(np.float64)
    edges*=np.array([width,height,width,height],dtype=np.float64)
    edges+=noise*amplitude
    edges=np.minimum(np.maximum(edges,0),[width,height,width,height])
    assert np.all(edges[:,2:]>edges[:,:2]),"Nonpositive perturbed GT"
    edges/=np.array([width,height,width,height],dtype=np.float64)
    out=np.concatenate(((edges[:,:2]+edges[:,2:])/2,edges[:,2:]-edges[:,:2]),axis=1)
    return torch.from_numpy(out.astype(np.float32))

def analyze(root, matcher):
    start=time.monotonic()
    out=Path(root)/"D0_results_v1"
    cache=np.load(out/"raw_final_outputs.npz",allow_pickle=False)
    selected=json.loads((out/"selected_inputs_FROZEN.json").read_text(encoding="utf-8"))["rows"]
    image_count=len(selected)
    assert 0<image_count<=128 and cache["logits"].shape==(image_count,300,10)
    torch.set_num_threads(1)
    matcher=matcher.cpu()
    assert matcher.use_focal_loss
    rows=[];assignment_dump={};native_equal=0
    summaries={}
    for i,meta in enumerate(selected):
        begin,end=cache["offsets"][i:i+2]
        target={"boxes":torch.from_numpy(cache["target_boxes"][begin:end].copy()),
                "labels":torch.from_numpy(cache["target_labels"][begin:end].copy())}
        logits=torch.from_numpy(cache["logits"][i].copy())
        boxes=torch.from_numpy(cache["boxes"][i].copy())
        labels=target["labels"].numpy()
        area=cache["original_area"][begin:end]
        clean_b=native_assignment(logits,boxes,target,matcher)
        assert np.array_equal(clean_b,cache["native_owners"][begin:end]),("Native CPU identity",i)
        native_equal+=1
        clean_m=nwd_assignment(logits,boxes,target,matcher)
        zero=perturb(target["boxes"],meta["width"],meta["height"],np.zeros((len(labels),4)),0)
        assert torch.allclose(zero,target["boxes"],atol=1e-7,rtol=0)
        # Numerical roundtrip is checked in coordinates; zero-amplitude matching uses original tensors.
        assert np.array_equal(native_assignment(logits,boxes,target,matcher),clean_b)
        assert np.array_equal(nwd_assignment(logits,boxes,target,matcher),clean_m)
        seed=int.from_bytes(hashlib.sha256(("A47-jitter-v1:"+meta["file"]).encode()).digest()[:8],"little")
        rng=np.random.default_rng(seed)
        half=rng.uniform(-1,1,size=(8,len(labels),4))
        noises=np.concatenate((half,-half),axis=0)
        img_arrays={}
        for method,clean in [("B",clean_b),("M",clean_m)]:
            clean_lab=query_labels(clean,labels)
            clean_soft=soft_targets(clean,labels,boxes,target["boxes"])
            clean_iou=paired_iou(boxes[clean],target["boxes"]).numpy()
            for amp in (.5,1.0):
                owners=[];flips=[];id_changes=[];soft_changes=[];events=[]
                for noise in noises:
                    pert=perturb(target["boxes"],meta["width"],meta["height"],noise,amp)
                    pt={"boxes":pert,"labels":target["labels"]}
                    match=(native_assignment(logits,boxes,pt,matcher) if method=="B"
                           else nwd_assignment(logits,boxes,pt,matcher))
                    assert len(np.unique(match))==len(labels) and match.min()>=0 and match.max()<300
                    lab=query_labels(match,labels)
                    soft=soft_targets(match,labels,boxes,pert)
                    owners.append(match)
                    flips.append(match!=clean)
                    id_changes.append(lab[clean]!=labels)
                    soft_changes.append(np.abs(soft[clean]-clean_soft[clean]).sum(-1))
                    changed=lab!=clean_lab
                    events.append({
                        "foreground_to_unmatched":int(((clean_lab>=0)&(lab<0)).sum()),
                        "unmatched_to_foreground":int(((clean_lab<0)&(lab>=0)).sum()),
                        "cross_class_foreground":int(((clean_lab>=0)&(lab>=0)&changed).sum()),
                        "same_class_changed_gt_owner":int(((match!=clean)&(lab[clean]==labels)).sum()),
                        "changed_query_class_or_positive_identity":int(changed.sum())})
                key=method+"_"+str(amp)
                img_arrays[key]=np.stack(owners).tolist()
                f=np.stack(flips).mean(0);c=np.stack(id_changes).mean(0)
                s=np.stack(soft_changes).mean(0)
                for g in range(len(labels)):
                    rows.append({"image":meta["file"],"gt_index":g,"class_id":int(labels[g])+1,
                        "original_area":float(area[g]),"cohort":"tiny" if area[g]<=256 else "small" if area[g]<=1024 else "larger",
                        "method":method,"amplitude":amp,"owner_flip_fraction":float(f[g]),
                        "positive_class_identity_change_fraction":float(c[g]),
                        "mean_absolute_MAL_soft_target_change":float(s[g]),"clean_matched_iou":float(clean_iou[g])})
                for k in events[0]:
                    summaries.setdefault(key,{}).setdefault(k,[]).extend([x[k] for x in events])
        assignment_dump[meta["file"]]={"clean_B":clean_b.tolist(),"clean_M":clean_m.tolist(),"draws":img_arrays}
        if (i+1)%16==0:dump(out/"analysis_progress.json",{"images":i+1,"total":image_count})
    result={"status":"COMPLETE","native_clean_identity_checks":native_equal,"images":image_count,
            "matching":{"native_weights":{"class":matcher.cost_class,"L1":matcher.cost_bbox,"GIoU":matcher.cost_giou},
                        "focal_alpha":matcher.alpha,"focal_gamma":matcher.gamma},
            "cohorts":{},"common_support":{},"event_counts":{}}
    for method in ["B","M"]:
        for amp in (.5,1.0):
            key=method+"_"+str(amp);group=[r for r in rows if r["method"]==method and r["amplitude"]==amp]
            result["cohorts"][key]={}
            for cohort in ["tiny","small","larger","area_le_1024","all"]:
                rr=[r for r in group if cohort=="all" or r["cohort"]==cohort or (cohort=="area_le_1024" and r["original_area"]<=1024)]
                result["cohorts"][key][cohort]={"GT":len(rr),"images":len({r["image"] for r in rr}),
                    **{field:float(np.mean([r[field] for r in rr])) if rr else None
                       for field in ["owner_flip_fraction","positive_class_identity_change_fraction","mean_absolute_MAL_soft_target_change","clean_matched_iou"]}}
            paired={}
            for r in group:
                if r["cohort"] in ["tiny","larger"]:
                    paired.setdefault((r["image"],r["class_id"]),{}).setdefault(r["cohort"],[]).append(r["owner_flip_fraction"])
            byimage={}
            for (image,cls),v in paired.items():
                if "tiny" in v and "larger" in v:
                    byimage.setdefault(image,[]).append(float(np.mean(v["tiny"])-np.mean(v["larger"])))
            result["common_support"][key]={"image_class_cells":sum("tiny" in v and "larger" in v for v in paired.values()),
                "images":len(byimage),"image_class_macro_tiny_minus_larger":float(np.mean([np.mean(v) for v in byimage.values()])) if byimage else None}
            result["event_counts"][key]={k:{"sum_over_all_image_draws":int(np.sum(v)),"mean_per_image_draw":float(np.mean(v))}for k,v in summaries[key].items()}
    b=result["cohorts"]["B_0.5"]["tiny"];m=result["cohorts"]["M_0.5"]["tiny"]
    large=result["cohorts"]["B_0.5"]["larger"];support=result["common_support"]["B_0.5"]
    adequate=b["GT"]>=200 and b["images"]>=20 and large["GT"]>=200 and support["images"]>=20
    prominent=bool(adequate and b["owner_flip_fraction"]>=.10 and
                  support["image_class_macro_tiny_minus_larger"]>=.05 and b["positive_class_identity_change_fraction"]>=.05)
    reduction=(b["owner_flip_fraction"]-m["owner_flip_fraction"])/b["owner_flip_fraction"] if b["owner_flip_fraction"] else None
    drop=b["clean_matched_iou"]-m["clean_matched_iou"]
    removed=bool(reduction is not None and reduction>=.5 and drop<=.01)
    result["prespecified_screen"]={"adequate_support":adequate,"prominent_native_primary_sensitivity":prominent,
        "NWD_relative_tiny_flip_reduction":reduction,"NWD_clean_mean_tiny_IoU_drop":drop,
        "simple_cost_removes_sensitivity_without_large_clean_iou_drop":removed,
        "novel_training_approved":False,"interpretation":"Continuation screen only; query changes and soft targets are not causal harm or AP.",
        "decision":"NO_TRAINING_FROM_THIS_SCREEN" if not prominent or removed else "METHOD_DIFFERENCE_REVIEW_REQUIRED_NO_AUTOMATIC_TRAINING"}
    result["analysis_seconds"]=time.monotonic()-start
    dump(out/"assignment_owners.json",assignment_dump)
    with (out/"per_gt_results.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump(out/"analysis_summary.json",result)
    return result

