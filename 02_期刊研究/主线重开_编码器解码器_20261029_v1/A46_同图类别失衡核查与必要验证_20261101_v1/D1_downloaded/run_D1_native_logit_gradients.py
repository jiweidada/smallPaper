from pathlib import Path
import sys
sys.dont_write_bytecode=True
import os,json,time,hashlib,signal,traceback
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"D1_results_v1"
OLD=Path("/root/c10_region_restore_20261029_v1/architecture_probe_v1/a43_ecdet_distill_three_arm_20261031_v1")
COUNTERS={"model_forwards":0,"logit_gradient_calls":0,"model_backward_calls":0,"optimizer_updates":0}
TAIL=[3,5,6,7,8,9,10]
HEAD={1,2,4}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def save(name,value):
    p=OUT/name
    temp=p.with_suffix(p.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    os.replace(temp,p)

def check_assets(expected):
    got={}
    for rel,key in [("runtime_common.py","runtime_common_sha256"),("arm_B/epoch12.pth","checkpoint_sha256"),
                    ("common_init_10class.pth","common_init_sha256"),("train_data_manifest.json","train_data_manifest_sha256")]:
        actual=sha(OLD/rel);assert actual==expected[key],("asset hash mismatch",rel);got[str(OLD/rel)]=actual
    source=OLD/"source/EdgeCrafter-d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecdetseg"
    for rel,expected_sha in expected["native_source_sha256"].items():
        actual=sha(source/rel);assert actual==expected_sha,("source hash mismatch",rel);got[str(source/rel)]=actual
    return got

def check_hashes(values):
    for path,expected in values.items():assert sha(path)==expected,("input changed",path)

def hash_tensors(items):
    h=hashlib.sha256()
    for name,tensor in sorted(items,key=lambda x:x[0]):
        v=tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()

def entry():
    t0=time.monotonic()
    protocol=json.loads((ROOT/"02_D1梯度来源诊断协议_冻结.json").read_text(encoding="utf-8"))
    expected=json.loads((ROOT/"D1_asset_expectations.json").read_text(encoding="utf-8"))
    source_hashes=check_assets(expected)
    import torch
    import numpy as np
    sys.path.insert(0,str(OLD))
    import runtime_common as native
    assert native.R==OLD
    assert len(expected["native_source_sha256"])==71
    dataset=native.VisDroneDataset("train")
    dataset.epoch=0
    old_manifest=json.loads((OLD/"train_data_manifest.json").read_text())
    byfile={r["file"]:r for r in old_manifest["rows"]}
    assert len(dataset.rows)==len(byfile)==6471
    eligible=[]
    for idx,r in enumerate(dataset.rows):
        assert native.digest(r["annotation"])==byfile[r["file"]]["annotation_sha256"]
        if not (1<=len(r["valid"])<=300):continue
        small_classes={int(v[4])+1 for v in r["valid"] if int(v[4])+1 in TAIL and (v[2]-v[0])*(v[3]-v[1])<=1024}
        if not small_classes:continue
        key=hashlib.sha256(("A46-D1-v1:"+r["file"]).encode()).hexdigest()
        eligible.append((key,idx,small_classes))
    eligible.sort(key=lambda x:x[0])
    chosen=[];chosen_set=set();selection_counts={}
    for c in TAIL:
        count=0
        for _,idx,classes in eligible:
            if c in classes and idx not in chosen_set:
                chosen.append(idx);chosen_set.add(idx);count+=1
                if count==8:break
        selection_counts[str(c)]=count
    for _,idx,_ in eligible:
        if len(chosen)==64:break
        if idx not in chosen_set:chosen.append(idx);chosen_set.add(idx)
    assert len(chosen)==64
    selected=[]
    for idx in chosen:
        r=dataset.rows[idx]
        ish=native.digest(dataset.images/r["file"])
        ash=native.digest(r["annotation"])
        assert ish==byfile[r["file"]]["image_sha256"]
        selected.append({"index":idx,"image_id":r["id"],"file":r["file"],"n_GT":len(r["valid"]),
                         "image_sha256":ish,"annotation_sha256":ash})
    save("selected_inputs_FROZEN_before_forward.json",{"indices":chosen,"images":selected,
         "class_priority_additions":selection_counts,"eligible_images":len(eligible),"forward_count_at_freeze":0})
    model,criterion,_=native.create_components("B",role="student")
    ckpt=torch.load(OLD/"arm_B/epoch12.pth",map_location="cpu",weights_only=True)
    assert ckpt["epoch"]==12 and ckpt["arm"]=="B"
    model.load_state_dict(ckpt["ema"]["module"],strict=True)
    del ckpt
    model.requires_grad_(False);model.eval();model=model.to("cuda:0")
    criterion=criterion.to("cuda:0");criterion.eval()
    assert "mal" in criterion.losses
    assert criterion.weight_dict["loss_mal"]==1
    assert native.MODEL_COUNTER["forwards"]==0
    p0=hash_tensors(model.named_parameters())
    s0=hash_tensors(model.state_dict().items())
    torch.cuda.reset_peak_memory_stats()
    class_totals={str(c):{"positive_grad":0.,"small_positive_grad":0.,"large_positive_grad":0.,
                         "negative_head_grad":0.,"negative_other_tail_grad":0.,"negative_unmatched_grad":0.,
                         "positive_count":0,"small_positive_count":0,"large_positive_count":0,
                         "negative_head_count":0,"negative_other_tail_count":0,"negative_unmatched_count":0}
                  for c in TAIL}
    details=[];max_conservation_error=0.
    for start in range(0,64,4):
        if time.monotonic()-t0>850:raise TimeoutError("diagnostic hard-timeout guard")
        batch_indices=chosen[start:start+4]
        images,targets=native.collate([dataset[idx] for idx in batch_indices])
        targets=native.targets_to_device(targets,"cuda:0")
        images=images.to("cuda:0")
        assert COUNTERS["model_forwards"]<16
        COUNTERS["model_forwards"]+=1
        with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            raw=model(images)
        logits=raw["pred_logits"].float().detach().clone().requires_grad_(True)
        boxes=raw["pred_boxes"].float().detach().clone()
        assert torch.isfinite(logits).all() and torch.isfinite(boxes).all()
        output={"pred_logits":logits,"pred_boxes":boxes}
        with torch.no_grad():
            matched=criterion.matcher({"pred_logits":logits.detach(),"pred_boxes":boxes},targets)
        indices=matched["indices"] if isinstance(matched,dict) else matched
        normalizer=max(1,sum(len(t["labels"]) for t in targets))
        for i,(q,gtidx) in enumerate(indices):
            assert len(q)==len(targets[i]["labels"]),(len(q),len(targets[i]["labels"]))
        loss=criterion.loss_labels_mal(output,targets,indices,normalizer)["loss_mal"]
        assert torch.isfinite(loss)
        grad=torch.autograd.grad(loss,logits,create_graph=False,retain_graph=False)[0].detach().abs().double()
        COUNTERS["logit_gradient_calls"]+=1
        assert torch.isfinite(grad).all()
        assigned=torch.full(logits.shape[:2],-1,dtype=torch.long,device="cuda:0")
        area=torch.zeros(logits.shape[:2],dtype=torch.float64,device="cuda:0")
        for i,(q,gtidx) in enumerate(indices):
            assigned[i,q]=targets[i]["labels"][gtidx]+1
            r=dataset.rows[batch_indices[i]]
            original_areas=torch.tensor([(v[2]-v[0])*(v[3]-v[1]) for v in r["valid"]],
                                        dtype=torch.float64,device="cuda:0")
            area[i,q]=original_areas[gtidx]
        for i,idx in enumerate(batch_indices):
            r=dataset.rows[idx]
            for c in TAIL:
                gc=grad[i,:,c-1]
                pos=assigned[i]==c
                sm=pos&(area[i]<=1024)
                large=pos&~sm
                nh=(assigned[i]==1)|(assigned[i]==2)|(assigned[i]==4)
                nt=(assigned[i]>=0)&~pos&~nh
                un=assigned[i]<0
                assert ((pos.long()+nh.long()+nt.long()+un.long())==1).all()
                masks={"positive":pos,"small_positive":sm,"large_positive":large,
                       "negative_head":nh,"negative_other_tail":nt,"negative_unmatched":un}
                record={"image_id":r["id"],"file":r["file"],"category_id":c,
                        "n_GT":len(r["valid"]),"head_GT":sum(int(v[4])+1 in HEAD for v in r["valid"])}
                for key,mask in masks.items():
                    value=float(gc[mask].sum());count=int(mask.sum())
                    record[key+"_grad"]=value;record[key+"_count"]=count
                    class_totals[str(c)][key+"_grad"]+=value
                    class_totals[str(c)][key+"_count"]+=count
                partition=record["positive_grad"]+record["negative_head_grad"]+record["negative_other_tail_grad"]+record["negative_unmatched_grad"]
                err=abs(partition-float(gc.sum()))
                max_conservation_error=max(max_conservation_error,err)
                assert err<1e-9
                details.append(record)
        save("progress.json",{"completed_images":start+4,"loss_mal":float(loss.detach()),"counters":COUNTERS,
             "elapsed_seconds":time.monotonic()-t0,"peak_allocated_bytes":torch.cuda.max_memory_allocated()})
        del raw,output,grad,loss,logits,boxes,images,targets
    assert COUNTERS["model_forwards"]==native.MODEL_COUNTER["forwards"]==16
    assert native.MODEL_COUNTER["teacher_forwards"]==0
    p1=hash_tensors(model.named_parameters());s1=hash_tensors(model.state_dict().items())
    assert p0==p1 and s0==s1,"Model weights/buffers changed"
    check_hashes(source_hashes)
    for r in selected:
        row=dataset.rows[r["index"]]
        assert native.digest(dataset.images/row["file"])==r["image_sha256"]
        assert native.digest(row["annotation"])==r["annotation_sha256"]
    supported=[];dominant=[]
    for c in TAIL:
        t=class_totals[str(c)]
        neg=t["negative_head_grad"]+t["negative_other_tail_grad"]+t["negative_unmatched_grad"]
        t["negative_total_grad"]=neg
        t["negative_head_fraction"]=t["negative_head_grad"]/neg if neg>0 else None
        t["negative_unmatched_fraction"]=t["negative_unmatched_grad"]/neg if neg>0 else None
        hm=t["negative_head_grad"]/max(1,t["negative_head_count"])
        um=t["negative_unmatched_grad"]/max(1,t["negative_unmatched_count"])
        t["mean_head_negative_grad_per_query"]=hm;t["mean_unmatched_negative_grad_per_query"]=um
        t["head_to_unmatched_mean_per_query_ratio"]=hm/um if um>0 else None
        t["supported"]=t["small_positive_count"]>=8
        t["head_specific_dominance"]=bool(t["supported"] and t["negative_head_fraction"] is not None and
                                        t["negative_head_fraction"]>=.5 and um>0 and hm/um>=2)
        if t["supported"]:supported.append(c)
        if t["head_specific_dominance"]:dominant.append(c)
    save("per_image_class_gradients.json",details)
    result={"status":"COMPLETE","recorded_utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":time.monotonic()-t0,
      "protocol_sha256":sha(ROOT/"02_D1梯度来源诊断协议_冻结.json"),"script_sha256":sha(__file__),
      "source_input_sha256":source_hashes,"dataset_images":len(dataset),"selected_images":64,
      "model_name":"native ECDet-S A43 original B fixed12 EMA","torch":torch.__version__,"cuda_device":torch.cuda.get_device_name(0),
      "classification_component":"Native loss_labels_mal final normal queries only","native_gamma":criterion.gamma,
      "native_mal_alpha":criterion.mal_alpha,"counters":COUNTERS,"runtime_counter":native.MODEL_COUNTER,
      "parameters_sha256_before":p0,"parameters_sha256_after":p1,"state_sha256_before":s0,"state_sha256_after":s1,
      "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"max_partition_conservation_error":max_conservation_error,
      "classes":class_totals,"supported_tail_classes":supported,"head_dominant_tail_classes":dominant,
      "prespecified_head_specific_screen_pass":len(dominant)>=4,
      "new_training_jobs":0,"new_optimizer_updates":0,"mainline_validated":False,
      "scope":"Local final-logit gradient sources on a balanced selected training-image set at frozen eval weights; NOT full train-mode/aux/DN/backbone gradient, intervention, AP gain or population estimate."}
    save("D1_complete.json",result)
    print(json.dumps({"status":"COMPLETE","elapsed_seconds":result["elapsed_seconds"],"counters":COUNTERS,
         "head_dominant_tail_classes":dominant,"classes":class_totals}),flush=True)

def main():
    OUT.mkdir(exist_ok=False)
    with (OUT/"run_once.lock").open("x") as f:f.write(str(os.getpid()))
    save("started.json",{"pid":os.getpid(),"started_utc":datetime.now(timezone.utc).isoformat(),
          "script_sha256":sha(__file__),"protocol_sha256":sha(ROOT/"02_D1梯度来源诊断协议_冻结.json"),"counters":COUNTERS})
    def alarm(sig,frame):raise TimeoutError("900second diagnostic timeout")
    signal.signal(signal.SIGALRM,alarm);signal.alarm(900)
    try:
        entry()
        save("exit_receipt.json",{"exit_code":0,"counters":COUNTERS})
    except BaseException as e:
        save("failure.json",{"type":type(e).__name__,"error":str(e),"traceback":traceback.format_exc(),"counters":COUNTERS})
        save("exit_receipt.json",{"exit_code":1,"counters":COUNTERS})
        raise
    finally:signal.alarm(0)

if __name__=="__main__":main()

