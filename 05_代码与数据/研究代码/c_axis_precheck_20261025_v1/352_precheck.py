"""352 frozen seed0 baseline/C CPU precheck. All files written exclusively in this directory."""
import os, sys
sys.dont_write_bytecode = True
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                  MKL_THREADING_LAYER="SEQUENTIAL")
import json, hashlib, time, platform, importlib.util, contextlib, io, traceback, gc
from pathlib import Path
from collections import Counter
import numpy as np
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
OUT = Path(__file__).resolve().parent
EN = OUT.parent.parent
NATIVE = EN / "code/dn_pre_frozen_diagnostic_20261020_v1/native"
ER = EN / "data/00_实验计划与证据索引/单一候选评价依赖补齐与构建核验_20261008_v1/isolated_runtime"
GOLD = EN / "data/04_冻结权重诊断_20261001"
PAIR = EN / "data/01_C主线_已完成"
VAL = Path(r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val")
LIGHT = PAIR / "C实际代码_远端_hgnetv2_light.py"
WEIGHTS = {"baseline": PAIR/"weights/baseline_main_seed0_best_stg2_remote.pth",
           "c": PAIR/"weights/p2_skip_main_seed0_best_stg2_remote.pth"}
sys.path[:0] = [str(NATIVE), str(ER)]
NAMES = ["AP","AP50","AP75","APs","APm","APl","AR1","AR10","AR100","ARs","ARm","ARl"]
BINS = np.array([0,2,3,4,8,16,32,np.inf])
LABELS = ["[0,2)","[2,3)","[3,4)","[4,8)","[8,16)","[16,32)","[32,inf)"]
EVENTS = []
PROTOCOL = None

def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda:f.read(2**20),b""): h.update(chunk)
    return h.hexdigest()

def put(name, data):
    with (OUT/name).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")

def textfile(name, text):
    with (OUT/name).open("x",encoding="utf-8",newline="\n") as f: f.write(text)

def event(kind, **kwargs):
    row={"time_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"event":kind,**kwargs}
    EVENTS.append(row)
    print(json.dumps(row,ensure_ascii=False),flush=True)

def state_sha(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        v=v.detach().cpu().contiguous()
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()

def geometry(dataset):
    ims={int(i["id"]):i for i in dataset["images"]}
    rows=[]
    for a in dataset["annotations"]:
        im=ims[int(a["image_id"])]
        xywh=torch.tensor(a["bbox"],dtype=torch.float32)
        xyxy=torch.cat((xywh[:2],xywh[:2]+xywh[2:]))
        # Same float32 multiply and subtraction as the 342/349 cached GT boxes.
        xyxy=xyxy*torch.tensor([640/im["width"],640/im["height"]]*2,dtype=torch.float32)
        w,h=(xyxy[2:]-xyxy[:2]).tolist()
        rows.append({"gt_id":int(a["id"]),"image_id":int(a["image_id"]),
                     "category_id":int(a["category_id"]),"w640":w,"h640":h,
                     "original_bbox":a["bbox"],"iscrowd":int(a.get("iscrowd",0))})
    return sorted(rows,key=lambda r:r["gt_id"])

def fixed_cells(rows):
    cells=[]
    w_all=[r for r in rows if r["w640"]<4 and r["h640"]>=4]
    for cat in range(10):
        for b,(lo,hi) in enumerate(zip([4,8,16,32],[8,16,32,np.inf])):
            w=[r for r in w_all if r["category_id"]==cat and lo<=r["h640"]<hi]
            m=[r for r in rows if r["category_id"]==cat and 4<=r["w640"]<8 and lo<=r["h640"]<hi]
            wi=len(set(r["image_id"] for r in w));mi=len(set(r["image_id"] for r in m))
            eligible=len(w)>=20 and len(m)>=20 and wi>=10 and mi>=10
            cells.append({"category_id":cat,"height_bin":b,"height_lo":lo,
                          "height_hi":None if np.isinf(hi) else hi,"W_n":len(w),"M_n":len(m),
                          "W_images":wi,"M_images":mi,"eligible":eligible,
                          "W_ids":[r["gt_id"] for r in w],"M_ids":[r["gt_id"] for r in m]})
    nw=sum(c["W_n"] for c in cells if c["eligible"])
    for c in cells: c["weight"]=c["W_n"]/nw if c["eligible"] and nw else 0.0
    return {"W_n":len(w_all),"eligible_W_n":nw,"coverage":nw/len(w_all),
            "categories":sorted(set(c["category_id"] for c in cells if c["eligible"])),"cells":cells}

def prepare():
    global PROTOCOL
    PROTOCOL=read(OUT/"352_protocol_frozen.json")
    assert not list(OUT.glob("*predictions*")), "Prediction output already present before prepare"
    dataset=read(VAL/"annotations/val.json")
    assert len(dataset["images"])==548 and len(dataset["annotations"])==38759
    assert sha(VAL/"annotations/val.json")=="0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd"
    audit=read(PAIR/"C正式配对审计_远端.json")
    assert audit["status"]=="passed" and audit["same_resolved_recipe_except_backbone_and_output"]
    assert audit["first_batch_hashes_match_all_96_epochs"]
    maps=list(audit["engine_source_hashes"].values())
    core=["engine/backbone/hgnetv2.py","engine/deim/hybrid_encoder.py","engine/deim/dfine_decoder.py",
          "engine/deim/dfine_utils.py","engine/deim/utils.py","engine/deim/deim.py",
          "engine/deim/postprocessor.py","engine/deim/box_ops.py","engine/deim/denoising.py","engine/deim/o2m_c.py"]
    checks={}
    for rel in core:
        actual=sha(NATIVE/rel)
        assert any(m.get(rel)==actual for m in maps), ("functional source mismatch",rel)
        checks[rel]=actual
    assert any(m.get("engine/backbone/hgnetv2_light.py")==sha(LIGHT) for m in maps)
    files=[Path(__file__),OUT/"352_protocol_frozen.json",OUT/"352_执行计划_冻结.txt",
           VAL/"annotations/val.json",PAIR/"C正式配对审计_远端.json",LIGHT]
    for label,wp in WEIGHTS.items():
        golden=read(GOLD/f"{label}_640_evaluation.json")
        assert sha(wp)==golden["checkpoint_sha256"]
        assert sha(GOLD/f"{label}_640_predictions.json")==golden["predictions_sha256"]
        assert golden["weights"]=="ema" and golden["precision"]=="fp32" and not golden["deployed"]
        files += [wp,GOLD/f"{label}_640_evaluation.json",GOLD/f"{label}_640_predictions.json"]
    for p in NATIVE.rglob("*.py"): files.append(p)
    for p in (ER/"pycocoeval").glob("*"):
        if p.is_file(): files.append(p)
    for name in ["T","S"]:
        files.append(EN/f"code/aspect_precheck_20261023_v1/349_cohort_{name}_ids.json")
    manifest={"frozen_before_new_forward":True,"created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
              "files":[{"path":str(p),"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(set(files),key=str)],
              "functional_source_verified":checks,"C_backbone_sha256":sha(LIGHT),
              "native_init_stubs":"isolated import-only __init__.py stubs; direct audited class construction",
              "common_initialization_sha256":audit["common_initialization_sha256"],
              "recipe_audit_scope":"same resolved recipe except backbone/output; first batch hashes match all96, not all batches"}
    rows=geometry(dataset)
    t=sorted(r["gt_id"] for r in rows if min(r["w640"],r["h640"])<4)
    s=sorted(r["gt_id"] for r in rows if 4<=min(r["w640"],r["h640"])<16)
    assert t==read(EN/"code/aspect_precheck_20261023_v1/349_cohort_T_ids.json"),("T mismatch",len(t))
    assert s==read(EN/"code/aspect_precheck_20261023_v1/349_cohort_S_ids.json"),("S mismatch",len(s))
    assert len(t)==4952 and len(s)==22197
    images=[]
    for im in sorted(dataset["images"],key=lambda i:i["id"]):
        p=VAL/"images"/im["file_name"]
        images.append({"image_id":int(im["id"]),"path":str(p),"sha256":sha(p),"width":im["width"],"height":im["height"]})
    put("352_input_manifest.json",manifest)
    put("352_image_manifest.json",images)
    put("352_gt_geometry.json",rows)
    put("352_cohorts.json",{"T_ids":t,"S_ids":s,
        "W_ids":[r["gt_id"] for r in rows if r["w640"]<4 and r["h640"]>=4],
        "M_ids":[r["gt_id"] for r in rows if 4<=r["w640"]<8 and r["h640"]>=4]})
    cells=fixed_cells(rows);put("352_fixed_cells.json",cells)
    put("352_prepare_receipt.json",{"status":"ready","images":548,"GT":len(rows),"T":len(t),"S":len(s),
        "W":cells["W_n"],"eligible_W":cells["eligible_W_n"],"coverage":cells["coverage"],
        "classes":cells["categories"],"eligible_cells":sum(c["eligible"] for c in cells["cells"]),
        "protocol_sha256":sha(OUT/"352_protocol_frozen.json"),"script_sha256":sha(__file__),
        "optimizer_steps":0,"forward_calls":0})
    event("prepared",W=cells["W_n"],coverage=cells["coverage"],classes=cells["categories"])

def verify_inputs():
    for record in read(OUT/"352_input_manifest.json")["files"]:
        assert sha(record["path"])==record["sha256"],("changed input",record["path"])
    for record in read(OUT/"352_image_manifest.json"):
        assert sha(record["path"])==record["sha256"],("changed image",record["image_id"])

def load_model(label):
    from engine.backbone.hgnetv2 import HGNetv2
    from engine.deim.hybrid_encoder import HybridEncoder
    from engine.deim.dfine_decoder import DFINETransformer
    from engine.deim.deim import DEIM
    from engine.deim.postprocessor import PostProcessor
    cfg=read(GOLD/f"{label}_640_evaluation.json")["resolved_config"]
    if label=="c":
        name="engine.backbone.c_axis_352"
        spec=importlib.util.spec_from_file_location(name,LIGHT)
        mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
        backbone=mod.HGNetv2_Light(**cfg["HGNetv2_Light"])
    else: backbone=HGNetv2(**cfg["HGNetv2"])
    enc=HybridEncoder(**cfg["HybridEncoder"],eval_spatial_size=[640,640])
    dec=DFINETransformer(**cfg["DFINETransformer"],num_classes=10,eval_spatial_size=[640,640])
    model=DEIM(backbone,enc,dec,num_classes=10)
    checkpoint=torch.load(WEIGHTS[label],map_location="cpu",weights_only=False)
    state=dict(checkpoint["ema"]["module"])
    initial=model.state_dict(); regenerated=[]
    for key in ["decoder.anchors","decoder.valid_mask"]:
        if key in initial: state[key]=initial[key];regenerated.append(key)
    result=model.load_state_dict(state,strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    model.eval().requires_grad_(False)
    assert model.encoder.eval_spatial_size==[640,640] and model.decoder.eval_spatial_size==[640,640]
    assert not any(m.training for m in model.modules())
    assert all(p.device.type=="cpu" and not p.requires_grad for p in model.parameters())
    post=PostProcessor(num_classes=10,num_top_queries=300,score_threshold=0.0).eval()
    audit=read(PAIR/"C正式配对审计_远端.json")
    maps=list(audit["engine_source_hashes"].values());consumed=[]
    for name,mod in list(sys.modules.items()):
        file=getattr(mod,"__file__",None)
        if name.startswith("engine.") and file and Path(file).is_relative_to(NATIVE):
            p=Path(file);rel=p.relative_to(NATIVE).as_posix()
            if p.suffix==".py" and p.name!="__init__.py":
                assert any(m.get(rel)==sha(p) for m in maps), ("consumed source mismatch",rel)
            consumed.append({"module":name,"path":str(p),"sha256":sha(p)})
    receipt={"checkpoint_last_epoch":checkpoint["last_epoch"],"checkpoint_best_stat":checkpoint["best_stat"],
        "source":"ema.module","strict_load":True,"regenerated_buffers":regenerated,"deployed":False,
        "eval_spatial_size":[640,640],"parameter_count":sum(p.numel() for p in model.parameters()),
        "state_sha256_before":state_sha(model),"consumed_modules":consumed}
    del checkpoint,state
    return model,post,receipt

def validate_predictions(rows, image_ids):
    counts=Counter()
    for r in rows:
        assert r["image_id"] in image_ids and r["category_id"] in range(10)
        assert np.isfinite(r["bbox"]).all() and np.isfinite(r["score"])
        assert r["bbox"][2]>0 and r["bbox"][3]>0,("nonpositive exported box",r)
        assert 0<r["score"]<=1
        counts[r["image_id"]]+=1
    assert set(counts)==set(image_ids) and set(counts.values())=={300}
    return len(rows)

def infer(label, dataset):
    from PIL import Image
    from torchvision.transforms import functional as TF
    model,post,receipt=load_model(label)
    event("loaded",label=label,parameters=receipt["parameter_count"])
    image_records=read(OUT/"352_image_manifest.json")
    cache=OUT/f"352_{label}_batch_cache";cache.mkdir()
    predictions=[];forward_seconds=0;post_seconds=0
    for start in range(0,len(image_records),4):
        batch=image_records[start:start+4];tensors=[]
        for rec in batch:
            with Image.open(rec["path"]) as image:
                assert image.size==(rec["width"],rec["height"])
                tensors.append(TF.to_tensor(TF.resize(image.convert("RGB"),[640,640])))
        x=torch.stack(tensors);sizes=torch.tensor([[r["width"],r["height"]] for r in batch])
        with torch.inference_mode():
            tick=time.perf_counter();outputs=model(x);elapsed=time.perf_counter()-tick
            forward_seconds+=elapsed
            event("forward",label=label,batch=start//4,image_count=len(batch),seconds=elapsed) if start%80==0 else None
            assert all(torch.isfinite(outputs[k]).all() for k in ["pred_logits","pred_boxes"])
            record={"image_ids":[r["image_id"] for r in batch],
                    "pred_logits":outputs["pred_logits"].cpu(),"pred_boxes":outputs["pred_boxes"].cpu(),
                    "forward_seconds":elapsed}
            with (cache/f"batch_{start//4:03d}.pt").open("xb") as f:torch.save(record,f)
            # Persist actual calls before any output assertion.
            EVENTS.append({"event":"forward_count","label":label,"image_count":len(batch),"seconds":elapsed})
            tick=time.perf_counter();results=post(outputs,sizes,for_eval=True,resize_pad=None)
            post_seconds+=time.perf_counter()-tick
            for rec,r in zip(batch,results):
                boxes=r["boxes"].clone();boxes[:,2:]-=boxes[:,:2]
                for box,score,category in zip(boxes.tolist(),r["scores"].tolist(),r["labels"].tolist()):
                    predictions.append({"image_id":rec["image_id"],"category_id":int(category),
                                        "bbox":box,"score":score})
        del outputs,x
    receipt.update(state_sha256_after=state_sha(model),forward_calls=137,image_forwards=548,
                   forward_seconds=forward_seconds,post_seconds=post_seconds,optimizer_steps=0,
                   gradients_all_none=all(p.grad is None for p in model.parameters()))
    assert receipt["state_sha256_before"]==receipt["state_sha256_after"]
    assert receipt["gradients_all_none"]
    # Save every raw exported row before positive-size assertion, preserving any failure evidence.
    put(f"352_{label}_predictions.json",predictions);put(f"352_{label}_inference_receipt.json",receipt)
    validate_predictions(predictions,[r["image_id"] for r in image_records])
    del model,post;gc.collect()
    return predictions

def evaluate(label, rows, dataset):
    from pycocoeval.coco import COCO
    from pycocoeval.cocoeval import COCOeval
    log=io.StringIO();tick=time.perf_counter()
    with contextlib.redirect_stdout(log):
        gt=COCO();gt.dataset=dataset;gt.createIndex()
        dt=gt.loadRes(rows)
        ev=COCOeval(gt,dt,"bbox",using_tiny_metrice=False,using_vtiny_metrice=False,max_dets=100)
        ev.params.imgIds=sorted(gt.getImgIds())
        ev.evaluate();ev.accumulate();ev.summarize()
    stats={k:float(v) for k,v in zip(NAMES,ev.stats)}
    assert list(ev.params.maxDets)==[1,10,100]
    textfile(f"352_{label}_coco.log",log.getvalue())
    put(f"352_{label}_metrics.json",{"metrics":stats,"maxDets":list(ev.params.maxDets),
        "eval_seconds":time.perf_counter()-tick,"GT_id_zero_preserved":True})
    matches={.5:{},.75:{}}
    for row in ev.evalImgs:
        if row is None or list(row["aRng"])!=list(ev.params.areaRng[0]):continue
        for threshold in matches:
            tidx=int(np.flatnonzero(np.isclose(ev.params.iouThrs,threshold))[0])
            for col,gid in enumerate(row["gtIds"]):
                did=int(row["gtMatches"][tidx,col])
                ignored=bool(row["gtIgnore"][col])
                matches[threshold][int(gid)]={"matched":did>0 and not ignored,"ignored":ignored,
                                             "detection_id":did if did>0 else None}
    assert len(matches[.5])==38759 and len(matches[.75])==38759
    assert not any(v["ignored"] for v in matches[.5].values())
    put(f"352_{label}_gt_matches.json",{str(t):m for t,m in matches.items()})
    # Keep detection evidence accessible by GT match id without serializing a huge evaluator object.
    matched_dt={}
    for t in matches:
        for v in matches[t].values():
            if v["detection_id"] is not None:
                did=v["detection_id"];matched_dt[did]=ev.cocoDt.anns[did]
    put(f"352_{label}_matched_detections.json",matched_dt)
    event("evaluated",label=label,AP=stats["AP"],APs=stats["APs"])
    del ev;gc.collect()
    return stats,matches

def group_summary(ids, evidence):
    rr=[evidence[i] for i in ids];n=len(rr)
    b=sum(r["baseline_tp"] for r in rr);c=sum(r["c_tp"] for r in rr)
    rescued=[r["gt_id"] for r in rr if r["c_tp"] and not r["baseline_tp"]]
    lost=[r["gt_id"] for r in rr if r["baseline_tp"] and not r["c_tp"]]
    net=len(rescued)-len(lost)
    assert c-b==net
    return {"n":n,"baseline_tp":b,"c_tp":c,"rescued":len(rescued),"lost":len(lost),"net_tp":net,
            "delta_recall":net/n if n else None,
            "rescue_rate_of_baseline_misses":len(rescued)/(n-b) if n>b else None,
            "loss_rate_of_baseline_tp":len(lost)/b if b else None,
            "rescued_gt_ids":rescued,"lost_gt_ids":lost}

def analyze(tag, stats, matches, gt):
    fixed=read(OUT/"352_fixed_cells.json");cohorts=read(OUT/"352_cohorts.json")
    imids=sorted({r["image_id"] for r in gt});imindex={v:i for i,v in enumerate(imids)}
    results={}
    for threshold in [.5,.75]:
        ev={}
        for r in gt:
            gid=r["gt_id"]
            b=matches["baseline"][threshold][gid];c=matches["c"][threshold][gid]
            ev[gid]={**r,"baseline_tp":b["matched"],"c_tp":c["matched"],
                     "baseline_detection_id":b["detection_id"],"c_detection_id":c["detection_id"]}
        put(f"352_{tag}_iou{int(threshold*100)}_paired_GT.json",list(ev.values()))
        groups={name:group_summary(ids,ev) for name,ids in cohorts.items()}
        groups["all"]=group_summary(list(ev),ev)
        grid=[]
        for w in range(7):
            for h in range(7):
                ids=[r["gt_id"] for r in gt if BINS[w]<=r["w640"]<BINS[w+1] and BINS[h]<=r["h640"]<BINS[h+1]]
                grid.append({"width":LABELS[w],"height":LABELS[h],**group_summary(ids,ev)})
        assert sum(g["n"] for g in grid)==len(gt)
        classes=[{"category_id":k,**group_summary([r["gt_id"] for r in gt if r["category_id"]==k],ev)}
                 for k in range(10)]
        width=[{"width":LABELS[k],**group_summary([r["gt_id"] for r in gt if BINS[k]<=r["w640"]<BINS[k+1]],ev)}
               for k in range(7)]
        height=[{"height":LABELS[k],**group_summary([r["gt_id"] for r in gt if BINS[k]<=r["h640"]<BINS[k+1]],ev)}
                for k in range(7)]
        cells=[]
        for cell in fixed["cells"]:
            a=group_summary(cell["W_ids"],ev);b=group_summary(cell["M_ids"],ev)
            gap=a["delta_recall"]-b["delta_recall"] if a["n"] and b["n"] else None
            cells.append({"category_id":cell["category_id"],"height_bin":cell["height_bin"],
                          "eligible":cell["eligible"],"weight":cell["weight"],"W":a,"M":b,"gap":gap})
        d=sum(c["weight"]*c["gap"] for c in cells if c["eligible"])
        results[str(threshold)]={"groups":groups,"width":width,"height":height,"width_height":grid,
                                 "class":classes,"cells":cells,"D":d}
        if threshold==.5:
            selected=[c for c in fixed["cells"] if c["eligible"]]
            ng=len(imids);arr_w=np.zeros((ng,2),np.int64)
            ngw=np.zeros((ng,len(selected)),np.int64);ngm=np.zeros_like(ngw)
            dw=np.zeros_like(ngw);dm=np.zeros_like(ngw)
            for gid in cohorts["W_ids"]:
                r=ev[gid];i=imindex[r["image_id"]]
                arr_w[i,0]+=1;arr_w[i,1]+=int(r["c_tp"])-int(r["baseline_tp"])
            for j,cell in enumerate(selected):
                for cohort,ns,ds in [("W_ids",ngw,dw),("M_ids",ngm,dm)]:
                    for gid in cell[cohort]:
                        r=ev[gid];i=imindex[r["image_id"]]
                        ns[i,j]+=1;ds[i,j]+=int(r["c_tp"])-int(r["baseline_tp"])
            rng=np.random.default_rng(20261025)
            multiplicities=rng.multinomial(ng,np.full(ng,1/ng),size=5000)
            totals=multiplicities@arr_w
            bootw=totals[:,1]/totals[:,0]
            nw=multiplicities@ngw;nm=multiplicities@ngm
            valid=np.all((nw>0)&(nm>0),axis=1) if selected else np.zeros(5000,dtype=bool)
            bootd=np.full(5000,np.nan)
            if selected:
                weights=np.array([c["weight"] for c in selected])
                bootd[valid]=(((multiplicities@dw)[valid]/nw[valid]-(multiplicities@dm)[valid]/nm[valid])*weights).sum(1)
            bwci=np.percentile(bootw,[2.5,97.5]).tolist()
            bdci=np.percentile(bootd[valid],[2.5,97.5]).tolist() if valid.any() else None
            with (OUT/f"352_{tag}_bootstrap_samples.npz").open("xb") as f:
                np.savez_compressed(f,W_delta_recall=bootw,D=bootd,valid=valid,
                                    seed=np.array(20261025),image_ids=np.array(imids))
            bootstrap={"replicates":5000,"seed":20261025,"unit":"paired images",
                       "W_delta_recall_CI95":bwci,"D_CI95":bdci,"valid_D_replicates":int(valid.sum()),
                       "valid_fraction":float(valid.mean()),
                       "meaning":"conditional on these two trained seed0 checkpoints; not across-seed uncertainty"}
    main=results["0.5"];w=main["groups"]["W_ids"];s=main["groups"]["S_ids"]
    small_drop=(s["baseline_tp"]-s["c_tp"])/s["baseline_tp"] if s["baseline_tp"] else None
    gates={
        "eligible_support":fixed["coverage"]>=.8 and len(fixed["categories"])>=2 and bootstrap["valid_fraction"]>=.99,
        "W_positive_with_CI":w["delta_recall"]>0 and bootstrap["W_delta_recall_CI95"][0]>0,
        "D_positive_with_CI":main["D"]>0 and bootstrap["D_CI95"] is not None and bootstrap["D_CI95"][0]>0,
        "AP_APs_non_decreasing":stats["c"]["AP"]>=stats["baseline"]["AP"] and stats["c"]["APs"]>=stats["baseline"]["APs"],
        "S_TP_drop_le_1pct":small_drop is not None and small_drop<=.01}
    report={"source":tag,"IoU":results,"bootstrap":bootstrap,"support":{k:v for k,v in fixed.items() if k!="cells"},
            "S_TP_drop_fraction":small_drop,"gates":gates,"primary_pass":all(gates.values())}
    put(f"352_{tag}_analysis.json",report)
    event("analyzed",source=tag,primary_pass=report["primary_pass"],gates=gates,
          W_delta=w["delta_recall"],D=main["D"],W_CI=bootstrap["W_delta_recall_CI95"],D_CI=bootstrap["D_CI95"])
    return report

def seal(result):
    put("352_result.json",result)
    lines=["# 352 C 横向关联零训练预检结果","",
           "结论："+result["decision"]+"。","",
           "仅使用正式 seed0 配对、EMA、val548、CPU FP32；本次优化步数为0。"]
    if "reproduction" in result:
        lines+=["","历史独立评估复现（差值为本次减历史；AP/APs容差1e-4，全部12项容差5e-4）：","",
                "| 模型 | 指标 | 历史 | CPU | 差值 |","|---|---|---:|---:|---:|"]
        for model,rep in result["reproduction"].items():
            for metric,row in rep["metrics"].items():
                lines.append(f'|{model}|{metric}|{row["reference"]:.10f}|{row["actual"]:.10f}|{row["delta"]:+.10g}|')
    if "cpu" in result:
        a=result["cpu"];p=a["IoU"]["0.5"]
        lines+=["","IoU0.5配对匹配：","",
                "| 固定GT组 | GT数 | baseline TP | C TP | 新增 | 丢失 | 净召回变化（百分点） |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        for name,row in p["groups"].items():
            lines.append(f'|{name}|{row["n"]}|{row["baseline_tp"]}|{row["c_tp"]}|{row["rescued"]}|{row["lost"]}|{100*row["delta_recall"]:+.4f}|')
        lines+=["",f'W 覆盖率 {100*a["support"]["coverage"]:.2f}%，纳入类别 {a["support"]["categories"]}。',
                f'W净召回变化95%区间 {a["bootstrap"]["W_delta_recall_CI95"]}（比例单位）。',
                f'D={p["D"]:.8f}；95%区间 {a["bootstrap"]["D_CI95"]}。',
                f'S TP下降比例={a["S_TP_drop_fraction"]:.8f}。',"","逐条判据：",""]
        for k,v in a["gates"].items():lines.append(f'- {k}：'+("满足" if v else "不满足"))
        lines+=['- 历史GPU预测敏感性结论一致：'+str(result["historical_primary_decision_consistent"]),
                "","宽度单轴分段（已控制比较另见D；此表仅描述）：","",
                "| 宽度 | GT | baseline TP | C TP | 新增 | 丢失 | 净召回百分点 |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        for row in p["width"]:
            lines.append(f'|{row["width"]}|{row["n"]}|{row["baseline_tp"]}|{row["c_tp"]}|{row["rescued"]}|{row["lost"]}|{100*row["delta_recall"]:+.4f}|')
        lines+=["","高度、宽×高49格、类别、各层、新增/丢失ID与IoU0.75均在352_cpu_analysis.json中。"]
    if "error" in result:lines+=["","硬停止原因："+result["error"]]
    lines+=["","训练状态："+result["training_status"],
            "","解释限制：单seed，按图bootstrap不代表跨seed不确定性；C是P2→P3残差桥。分层关联不能因果证明只补横向，不能证明不等步长具有新颖性。保留原GT id=0和已发布COCO口径及其哨兵局限。未访问test-dev。"]
    textfile("352_中文结果.txt","\n".join(lines)+"\n")
    put("352_execution_events.json",EVENTS)
    events=[e for e in EVENTS if e["event"]=="forward_count"]
    receipts=[]
    for p in OUT.glob("352_*_inference_receipt.json"):receipts.append(read(p))
    put("352_execution_counts.json",{"device":"cpu","optimizer_steps":0,
        "physical_forward_calls_this_invocation":len(events),"image_forwards_this_invocation":sum(e["image_count"] for e in events),
        "forward_seconds":sum(e["seconds"] for e in events),
        "model_loads":len([e for e in EVENTS if e["event"]=="loaded"]),
        "COCO_evaluations":len([e for e in EVENTS if e["event"]=="evaluated"]),
        "weight_files_unchanged":all(sha(p)==read(GOLD/f"{k}_640_evaluation.json")["checkpoint_sha256"] for k,p in WEIGHTS.items()),
        "inference_receipts":receipts,"training_started":False})
    put("352_delivery_SHA256.json",{"files":[{"path":str(p.relative_to(OUT)),"bytes":p.stat().st_size,"sha256":sha(p)}
        for p in sorted(OUT.rglob("*")) if p.is_file()]})
    event("sealed",decision=result["decision"])

def run():
    global PROTOCOL
    PROTOCOL=read(OUT/"352_protocol_frozen.json");verify_inputs()
    event("started",python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform())
    data=read(VAL/"annotations/val.json");stats={};matches={};repro={}
    for label in ["baseline","c"]:
        rows=infer(label,data)
        metric,match=evaluate(label,rows,data)
        stats[label]=metric;matches[label]=match
        ref=read(GOLD/f"{label}_640_evaluation.json")["metrics"]
        details={k:{"reference":ref[k],"actual":metric[k],"delta":metric[k]-ref[k],
                    "tolerance":.0001 if k in ["AP","APs"] else .0005} for k in NAMES}
        ok=all(abs(v["delta"])<=v["tolerance"] for v in details.values())
        repro[label]={"passed":ok,"metrics":details}
        put(f"352_{label}_reproduction.json",repro[label])
        if not ok:
            seal({"decision":"reproduction_failed","reproduction":repro,
                  "training_status":"未启动：复现硬门槛未通过","error":label+" historical reproduction failed"})
            return
        del rows
    gt=read(OUT/"352_gt_geometry.json")
    cpu=analyze("cpu",stats,matches,gt)
    historical_stats={};historical_matches={}
    for label in ["baseline","c"]:
        rows=read(GOLD/f"{label}_640_predictions.json")
        validate_predictions(rows,[i["id"] for i in data["images"]])
        historical_stats[label],historical_matches[label]=evaluate("historical_"+label,rows,data)
        ref=read(GOLD/f"{label}_640_evaluation.json")["metrics"]
        assert max(abs(historical_stats[label][k]-ref[k]) for k in NAMES)<1e-12,"bound evaluator mismatch on golden rows"
        del rows
    history=analyze("historical",historical_stats,historical_matches,gt)
    consistent=bool(cpu["primary_pass"]==history["primary_pass"])
    passed=bool(cpu["primary_pass"] and consistent)
    decision="pass" if passed else ("insufficient" if not cpu["gates"]["eligible_support"] else "mixed")
    seal({"decision":decision,"reproduction":repro,"cpu":cpu,
          "historical":history,"historical_primary_decision_consistent":consistent,
          "all_frozen_gates_passed":passed,"training_authorized_if_passed":True,
          "training_status":"待执行独立训练冻结与资源核验；本CPU预检进程未启动训练" if passed else "未启动：冻结预检判据未全部满足"})

if __name__=="__main__":
    if "--prepare" in sys.argv:prepare()
    elif "--run" in sys.argv:
        try:run()
        except Exception as exc:
            traceback.print_exc()
            if not (OUT/"352_result.json").exists():
                seal({"decision":"hard_check_failed","error":repr(exc),"traceback":traceback.format_exc(),
                      "training_status":"未启动：硬检查失败，未改变判据"})
            raise
    else:raise SystemExit("use --prepare then --run")
