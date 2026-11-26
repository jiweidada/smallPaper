from pathlib import Path
import sys
sys.dont_write_bytecode=True
import os,json,time,hashlib,signal,traceback
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"D0_results_v1"
OLD=Path("/root/c10_region_restore_20261029_v1/architecture_probe_v1/a43_ecdet_distill_three_arm_20261031_v1")
COUNTERS={"model_forwards":0,"model_backward_calls":0,"optimizer_updates":0}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def save(name,value):
    p=OUT/name;temp=p.with_suffix(p.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    os.replace(temp,p)

def assets(expected):
    got={}
    for rel,key in [("runtime_common.py","runtime_common_sha256"),("arm_B/epoch12.pth","checkpoint_sha256"),
                    ("common_init_10class.pth","common_init_sha256"),("train_data_manifest.json","train_data_manifest_sha256")]:
        actual=sha(OLD/rel);assert actual==expected[key],("asset hash mismatch",rel);got[str(OLD/rel)]=actual
    source=OLD/"source/EdgeCrafter-d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecdetseg"
    for rel,v in expected["native_source_sha256"].items():
        assert sha(source/rel)==v,rel;got[str(source/rel)]=v
    return got

def hash_state(items):
    h=hashlib.sha256()
    for name,tensor in sorted(items,key=lambda x:x[0]):
        v=tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()

def main():
    import torch,numpy as np
    t0=time.monotonic()
    protocol=json.loads((ROOT/"protocol_FROZEN.json").read_text(encoding="utf-8"))
    expected=json.loads((ROOT/"asset_expectations.json").read_text(encoding="utf-8"))
    input_hashes=assets(expected)
    sys.path.insert(0,str(OLD))
    import runtime_common as native
    assert native.R==OLD and len(expected["native_source_sha256"])==71
    native.seed_all();torch.set_num_threads(1)
    dataset=native.VisDroneDataset("train");dataset.epoch=0
    manifest=json.loads((OLD/"train_data_manifest.json").read_text())
    byfile={r["file"]:r for r in manifest["rows"]}
    assert len(dataset.rows)==len(byfile)==6471
    eligible=[]
    for idx,r in enumerate(dataset.rows):
        assert native.digest(r["annotation"])==byfile[r["file"]]["annotation_sha256"]
        if 1<=len(r["valid"])<=300 and any((v[2]-v[0])*(v[3]-v[1])<=1024 for v in r["valid"]):
            eligible.append((hashlib.sha256(("A47-D0-v1::"+r["file"]).encode()).hexdigest(),idx))
    chosen=[idx for _,idx in sorted(eligible)[:128]]
    assert len(chosen)==128
    rows=[]
    for idx in chosen:
        r=dataset.rows[idx];m=byfile[r["file"]]
        for p,h in [(dataset.images/r["file"],m["image_sha256"]),(Path(r["annotation"]),m["annotation_sha256"])]:
            assert sha(p)==h;input_hashes[str(p)]=h
        rows.append({k:r[k] for k in ["id","file","width","height"]}|{"index":idx,"GT":len(r["valid"]),
             "image_sha256":m["image_sha256"],"annotation_sha256":m["annotation_sha256"]})
    save("selected_inputs_FROZEN.json",{"eligible":len(eligible),"selection":"first128 sorted SHA256(A47-D0-v1::filename)","rows":rows})
    model,criterion,_=native.create_components("B",role="student")
    ckpt=torch.load(OLD/"arm_B/epoch12.pth",map_location="cpu",weights_only=True)
    assert ckpt["epoch"]==12 and ckpt["arm"]=="B"
    model.load_state_dict(ckpt["ema"]["module"],strict=True)
    model.requires_grad_(False);model.eval();model=model.to("cuda:0")
    state_before=hash_state(model.state_dict().items())
    params_before=hash_state(model.named_parameters())
    criterion.eval();criterion=criterion.cpu()
    assert criterion.gamma==1.5 and criterion.mal_alpha is None
    logits_all=[];boxes_all=[];targets_all=[];labels_all=[];owners_all=[];original_areas=[];offsets=[0]
    torch.cuda.reset_peak_memory_stats()
    for start in range(0,128,4):
        batch=[dataset[idx] for idx in chosen[start:start+4]]
        images,targets=native.collate(batch)
        assert COUNTERS["model_forwards"]<32
        COUNTERS["model_forwards"]+=1
        with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            raw=model(images.to("cuda:0"))
        logits=raw["pred_logits"].detach().float().cpu().clone()
        boxes=raw["pred_boxes"].detach().float().cpu().clone()
        assert logits.shape==(4,300,10) and boxes.shape==(4,300,4)
        assert torch.isfinite(logits).all() and torch.isfinite(boxes).all()
        with torch.no_grad():matches=criterion.matcher({"pred_logits":logits,"pred_boxes":boxes},targets)["indices"]
        for k,t in enumerate(targets):
            q,g=[x.cpu().numpy() for x in matches[k]]
            owner=np.empty(len(g),dtype=np.int64);owner[g]=q
            assert len(g)==len(t["labels"])
            logits_all.append(logits[k].numpy());boxes_all.append(boxes[k].numpy())
            targets_all.append(t["boxes"].numpy());labels_all.append(t["labels"].numpy())
            owners_all.append(owner);offsets.append(offsets[-1]+len(g))
            original_areas.append(np.asarray([(v[2]-v[0])*(v[3]-v[1]) for v in dataset.rows[chosen[start+k]]["valid"]],dtype=np.float64))
        save("capture_progress.json",{"images":start+4,"counters":COUNTERS})
    state_after=hash_state(model.state_dict().items());params_after=hash_state(model.named_parameters())
    assert state_before==state_after and params_before==params_after
    assert native.MODEL_COUNTER=={"forwards":32,"teacher_forwards":0}
    peak=int(torch.cuda.max_memory_allocated())
    np.savez_compressed(OUT/"raw_final_outputs.npz",logits=np.stack(logits_all),boxes=np.stack(boxes_all),
        target_boxes=np.concatenate(targets_all),target_labels=np.concatenate(labels_all),
        native_owners=np.concatenate(owners_all),original_area=np.concatenate(original_areas),offsets=np.asarray(offsets,dtype=np.int64))
    del model,raw
    torch.cuda.empty_cache()
    from run_A47_match_analysis import analyze
    with torch.no_grad():analysis=analyze(ROOT,criterion.matcher)
    for path,h in input_hashes.items():assert sha(path)==h,("input changed",path)
    save("complete.json",{"status":"COMPLETE","elapsed_seconds":time.monotonic()-t0,"counters":COUNTERS,
        "runtime_counters":native.MODEL_COUNTER,"selected_images":128,"parameters_unchanged":True,
        "state_unchanged":True,"parameter_hash_before":params_before,"parameter_hash_after":params_after,
        "state_hash_before":state_before,"state_hash_after":state_after,"max_memory_allocated_bytes":peak,
        "torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),
        "protocol_sha256":sha(ROOT/"protocol_FROZEN.json"),"capture_script_sha256":sha(__file__),
        "analysis_script_sha256":sha(ROOT/"run_A47_match_analysis.py"),"asset_expectations_sha256":sha(ROOT/"asset_expectations.json"),
        "source_input_sha256":input_hashes,"raw_outputs_sha256":sha(OUT/"raw_final_outputs.npz"),
        "screen":analysis["prespecified_screen"],
        "scope":"Frozen eval final ordinary queries and synthetic annotation-coordinate perturbations only; no training/AP/causal claim."})
    print(json.dumps({"status":"COMPLETE","counters":COUNTERS,"screen":analysis["prespecified_screen"]}),flush=True)

if __name__=="__main__":
    lock=os.open(ROOT/"RUN_ONCE.lock",os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.write(lock,str(os.getpid()).encode());os.close(lock)
    OUT.mkdir(exist_ok=False)
    save("started.json",{"pid":os.getpid(),"utc":datetime.now(timezone.utc).isoformat(),"counters":COUNTERS})
    def timeout(signum,frame):raise TimeoutError("A47 hard timeout900s")
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    code=0
    try:main()
    except BaseException as e:
        code=1;save("failure.json",{"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc(),"counters":COUNTERS})
        traceback.print_exc()
    finally:
        signal.alarm(0)
        save("exit_receipt.json",{"exit_code":code,"counters":COUNTERS,"utc":datetime.now(timezone.utc).isoformat()})
    sys.exit(code)

