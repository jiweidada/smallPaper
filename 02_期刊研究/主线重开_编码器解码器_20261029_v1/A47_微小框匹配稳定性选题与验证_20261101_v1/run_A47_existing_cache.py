"""Existing-cache-only A47 diagnostic. No detector, checkpoint or optimizer is loaded."""
from pathlib import Path
import sys,os,json,hashlib,time,ast,logging,traceback
from typing import Dict
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
ENG=ROOT.parents[1]
RUNTIME=ENG/"code/dn_pre_frozen_diagnostic_20261020_v1/runtime"
sys.path.insert(0,str(RUNTIME))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import importlib.util
from run_A47_match_analysis import analyze,dump
torch.set_num_threads(1)
T0=time.monotonic()
CACHE=ENG/"code/unit_instance_completion_20261021_v1/342_codex_stage3_20261022_v1"
AUDIT=ENG/"code/problem_definition_audit_20261026_v1/cache_contract_audit.json"
ANN=Path("C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val/annotations/val.json")
NATIVE=ENG/"code/dn_pre_frozen_diagnostic_20261020_v1/native/engine/deim"
BUNDLES=ENG/"data/00_实验计划与证据索引/本次24轮双基线模型下载_20261009_v1"
SOURCES={}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def protect(p):
    SOURCES[str(p)]=sha(p)
    return SOURCES[str(p)]

def build_matcher(cfg):
    # Execute the original class AST, including method bodies and decorators unchanged.
    spec=importlib.util.spec_from_file_location("a47_original_box_ops",NATIVE/"box_ops.py")
    boxops=importlib.util.module_from_spec(spec);spec.loader.exec_module(boxops)
    tree=ast.parse((NATIVE/"matcher.py").read_text(encoding="utf-8"))
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="HungarianMatcher")
    ns=dict(torch=torch,nn=nn,F=F,np=np,linear_sum_assignment=linear_sum_assignment,Dict=Dict,
        register=lambda:(lambda x:x),logger=logging.getLogger("A47"),VIS_MATCH=False,
        VIS_MATCH_EVERY_N_EPOCHS=1,VIS_MATCH_DIR="unused_A47",os=os,_vis_match_call_idx=0)
    ns.update({n:getattr(boxops,n) for n in ["box_cxcywh_to_xyxy","generalized_box_iou","box_iou",
                   "batch_dice_loss","batch_sigmoid_ce_loss","generalized_box_inner_iou"]})
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(NATIVE/"matcher.py"),"exec"),ns)
    options=dict(cfg["DEIMCriterion"]["matcher"]);options.pop("type")
    assert not options.get("change_matcher",False)
    return ns["HungarianMatcher"](**options,use_focal_loss=cfg["use_focal_loss"]),boxops

def numpy_assignment(logits,boxes,gt,labels):
    p=1/(1+np.exp(-logits.astype(np.float64)))
    p=p[:,labels]
    cl=.25*(1-p)**2*(-np.log(p+1e-8))-.75*p**2*(-np.log(1-p+1e-8))
    a=np.concatenate((boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2),axis=1).astype(np.float64)
    b=np.concatenate((gt[:,:2]-gt[:,2:]/2,gt[:,:2]+gt[:,2:]/2),axis=1).astype(np.float64)
    inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(-1)
    union=(a[:,2:]-a[:,:2]).prod(-1)[:,None]+(b[:,2:]-b[:,:2]).prod(-1)[None,:]-inter
    enclosing=(np.maximum(a[:,None,2:],b[None,:,2:])-np.minimum(a[:,None,:2],b[None,:,:2])).prod(-1)
    giou=inter/union-(enclosing-union)/enclosing
    cost=2*cl+5*np.abs(boxes[:,None,:].astype(np.float64)-gt[None,:,:]).sum(-1)-2*giou
    q,g=linear_sum_assignment(cost)
    owner=np.empty(len(g),dtype=np.int64);owner[g]=q
    return owner,cost

def main():
    protocol=json.loads((ROOT/"offline_protocol_FROZEN.json").read_text(encoding="utf-8"))
    for p in [Path(__file__),ROOT/"offline_protocol_FROZEN.json",ROOT/"run_A47_match_analysis.py",AUDIT,ANN,NATIVE/"matcher.py",NATIVE/"box_ops.py"]:
        protect(p)
    audit=json.loads(AUDIT.read_text(encoding="utf-8"))
    manifest={x["path"]:x["sha256"] for x in audit["input_manifest"]}
    assert sha(ANN)=="0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd"
    ann=json.loads(ANN.read_text(encoding="utf-8"));assert len(ann["images"])==548
    by={}
    for a in ann["annotations"]:by.setdefault(a["image_id"],[]).append(a)
    annmap={a["id"]:a for a in ann["annotations"]}
    eligible=[]
    for im in ann["images"]:
        items=by.get(im["id"],[])
        if not 1<=len(items)<=300:continue
        if not all(a["bbox"][2]>0 and a["bbox"][3]>0 and a["bbox"][0]>=0 and a["bbox"][1]>=0 and
                   a["bbox"][0]+a["bbox"][2]<=im["width"] and a["bbox"][1]+a["bbox"][3]<=im["height"] and
                   0<=a["category_id"]<10 for a in items):continue
        if not any(a["bbox"][2]*a["bbox"][3]<=1024 for a in items):continue
        key=hashlib.sha256(("A47-OFFLINE-v1:"+im["file_name"]).encode()).hexdigest()
        eligible.append((key,im))
    selected=[im for key,im in sorted(eligible,key=lambda x:x[0])[:128]]
    assert len(selected)==128
    # This receipt is written before reading any selected prediction tensors.
    dump(ROOT/"offline_selected_inputs_FROZEN.json",{"eligible":len(eligible),"selected":selected,
         "selection":"label-only first128 SHA256(A47-OFFLINE-v1:filename)","protocol_sha256":sha(ROOT/"offline_protocol_FROZEN.json")})
    loaded={1:[],2:[]};excluded=[];keep=[];identity_checks=[]
    for im in selected:
        pair=[]
        for seed in (1,2):
            p=CACHE/f"342_cache_val_v2_seed{seed}"/f"342_{im['id']:06d}.pt"
            h=protect(p);assert h==manifest[str(p)],("cache audit mismatch",str(p))
            d=torch.load(p,map_location="cpu",weights_only=True)
            assert d["image_id"]==im["id"] and d["seed"]==seed and d["file_name"]==im["file_name"]
            assert d["orig_wh"]==[im["width"],im["height"]] and d["annotation_sha256"]==sha(ANN)
            assert d["final300_boxes"].shape==(300,4) and d["final300_logits"].shape==(300,10)
            items=[annmap[aid] for aid in d["gt_annotation_ids"]]
            assert [a["category_id"] for a in items]==d["gt_labels"].tolist()
            xywh=torch.tensor([a["bbox"] for a in items],dtype=torch.float32)
            corners=torch.cat((xywh[:,:2],xywh[:,:2]+xywh[:,2:]),dim=1)
            expected=corners/torch.tensor([im["width"],im["height"]]*2)*640
            assert torch.allclose(expected,d["gt_boxes_640"],atol=1e-4,rtol=0)
            assert len(items)==len(by[im["id"]])
            pair.append((d,items,p))
        valid=all(torch.isfinite(d["final300_boxes"]).all() and torch.isfinite(d["final300_logits"]).all() and
                  (d["final300_boxes"][:,2:]>0).all() for d,items,p in pair)
        if not valid:
            excluded.append({"image_id":im["id"],"file":im["file_name"],"reason":"entire paired image excluded: at least one raw query nonfinite or nonpositive; no replacement"})
            continue
        keep.append(im)
        for seed,item in zip((1,2),pair):loaded[seed].append(item)
    assert len(keep)>0
    summaries={}
    for seed in (1,2):
        config_path=BUNDLES/f"seed{seed}/bundle/resolved.yml"
        protect(config_path)
        assert sha(config_path)==next(x["sha256"] for x in audit["decoder_configs"] if x["seed"]==seed)
        cfg=json.loads(config_path.read_text(encoding="utf-8"))
        assert cfg["ResidualGeometryDEIM"]["arm"]=="baseline"
        assert cfg["DEIMCriterion"]["gamma"]==1.5 and cfg["DEIMCriterion"].get("mal_alpha") is None
        matcher,ops=build_matcher(cfg);assert matcher.use_focal_loss
        subroot=ROOT/f"offline_seed{seed}";out=subroot/"D0_results_v1";out.mkdir(parents=True,exist_ok=False)
        arrs={k:[] for k in ["logits","boxes","target_boxes","target_labels","native_owners","original_area"]}
        offsets=[0];metas=[]
        for im,(d,items,path) in zip(keep,loaded[seed]):
            edges=d["gt_boxes_640"]/640
            gt=torch.cat(((edges[:,:2]+edges[:,2:])/2,edges[:,2:]-edges[:,:2]),dim=1)
            target={"boxes":gt,"labels":d["gt_labels"]}
            logits=d["final300_logits"];boxes=d["final300_boxes"]
            with torch.no_grad():q,g=matcher({"pred_logits":logits[None],"pred_boxes":boxes[None]},[target])["indices"][0]
            owner=np.empty(len(g),dtype=np.int64);owner[g.numpy()]=q.numpy()
            check,cost=numpy_assignment(logits.numpy(),boxes.numpy(),gt.numpy(),d["gt_labels"].numpy())
            mismatch=int(np.sum(check!=owner))
            # Different FP32/FP64 tie choices are a visible source precision result.
            native_cost=float(cost[owner,np.arange(len(owner))].sum())
            check_cost=float(cost[check,np.arange(len(owner))].sum())
            identity_checks.append({"seed":seed,"image":im["file_name"],"different_owner_GT":mismatch,
                                    "native_cost_on_float64_formula_minus_optimum":native_cost-check_cost})
            assert mismatch==0,("FP32 source vs FP64 reference assignment differs; no silent substitution",seed,im["file_name"],mismatch)
            for key,v in [("logits",logits.numpy()),("boxes",boxes.numpy()),("target_boxes",gt.numpy()),
                          ("target_labels",d["gt_labels"].numpy()),("native_owners",owner),
                          ("original_area",np.asarray([a["bbox"][2]*a["bbox"][3] for a in items],dtype=np.float64))]:
                arrs[key].append(v)
            offsets.append(offsets[-1]+len(items))
            metas.append({"file":im["file_name"],"id":im["id"],"width":im["width"],"height":im["height"],
                          "GT":len(items),"source_cache":str(path),"checkpoint_sha256_from_cache":d["checkpoint_sha256"]})
        np.savez_compressed(out/"raw_final_outputs.npz",logits=np.stack(arrs["logits"]),boxes=np.stack(arrs["boxes"]),
              **{k:np.concatenate(arrs[k]) for k in ["target_boxes","target_labels","native_owners","original_area"]},
              offsets=np.asarray(offsets,dtype=np.int64))
        dump(out/"selected_inputs_FROZEN.json",{"source":"old val548 BASE24 cache","seed":seed,"rows":metas,"excluded_pairs":excluded})
        with torch.no_grad():summaries[str(seed)]=analyze(subroot,matcher)
        print(json.dumps({"seed_complete":seed,"images":len(keep),"screen":summaries[str(seed)]["prespecified_screen"]}),flush=True)
        assert time.monotonic()-T0<600,"A47 offline timeout"
    for p,h in SOURCES.items():assert sha(p)==h,("protected input changed",p)
    dump(ROOT/"offline_complete.json",{"status":"COMPLETE","elapsed_seconds":time.monotonic()-T0,
        "selected_images":128,"analyzed_paired_images":len(keep),"excluded_images":excluded,
        "source_sha256":SOURCES,"native_vs_numpy_identity":identity_checks,
        "environment":{"torch":torch.__version__,"numpy":np.__version__,"python":sys.version},
        "new_model_forwards":0,"new_GPU_calls":0,"new_optimizer_updates":0,"new_training_epochs":0,
        "summaries":summaries,"scope":"Existing-cache synthetic bounded annotation jitter, two old model seeds; not ECDet or training/AP evidence."})

if __name__=="__main__":
    fd=os.open(ROOT/"OFFLINE_RUN_ONCE.lock",os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(fd,str(os.getpid()).encode());os.close(fd)
    code=0
    try:main()
    except BaseException as e:
        code=1;dump(ROOT/"offline_failure.json",{"status":"FAILED","type":type(e).__name__,"message":str(e),
             "traceback":traceback.format_exc(),"new_model_forwards":0,"new_optimizer_updates":0})
        traceback.print_exc()
    finally:dump(ROOT/"offline_exit_receipt.json",{"exit_code":code,"elapsed_seconds":time.monotonic()-T0})
    sys.exit(code)

