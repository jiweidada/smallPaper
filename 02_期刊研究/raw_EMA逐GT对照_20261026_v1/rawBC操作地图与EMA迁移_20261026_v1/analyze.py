"""Frozen cache-only raw B/C operation map, then same-CPU EMA A/B/C transfer."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import sys, json, csv, gzip, hashlib, time, copy, gc, threading, argparse, importlib.util
from pathlib import Path
from collections import Counter, defaultdict
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
RUN = Path(__file__).resolve().parent
PAIR = RUN.parent
ROOT = PAIR.parent
PREV = ROOT / "问题优先诊断_20261026_v1"
BC = PAIR / "后续零训练预检_20261026_v1"
CPU = PAIR / "同后端CPU重放_20261026_v1"
VAL = Path(r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val\annotations\val.json")
spec = importlib.util.spec_from_file_location("frozen_previous_audit", PREV/"audit.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
from pycocoeval import mask as mask_utils
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
torch.set_num_threads(2)
THRESHOLDS = (50, 75)
parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=("raw", "ema"))
parser.add_argument("--attempt", default="v1")
args = parser.parse_args()
if not args.attempt.replace("_", "").isalnum():
    raise ValueError("Unsafe attempt name")
OUT = RUN / (("raw_operations_" if args.phase=="raw" else "EMA_transfer_") + args.attempt)
OUT.mkdir(exist_ok=False)
A.OUT = OUT
START = time.monotonic()
STOP = threading.Event()
OPERATIONS = {"evaluator_calls": 0, "new_model_forwards": 0, "head_calls": 0,
              "checkpoint_loads": 0, "model_builds": 0, "training": 0,
              "GPU": False, "remote_compute": False, "test_access": False}
def dump(name, value):
    with (OUT/name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1<<20), b""):
            h.update(block)
    return h.hexdigest()
def guard():
    while not STOP.wait(2):
        reason = None
        if time.monotonic()-START > 900:
            reason = "900 second phase budget"
        elif sum(p.stat().st_size for p in RUN.rglob("*") if p.is_file()) > 250*(1<<20):
            reason = "250MiB output budget"
        if reason:
            try:
                dump("hard_budget_stop.json", {"reason": reason, "operations": OPERATIONS,
                                               "elapsed_seconds": time.monotonic()-START})
                print("HARD STOP:", reason, flush=True)
            finally:
                os._exit(124)
threading.Thread(target=guard, daemon=True).start()
INPUTS = {}
def track(path, expected=None):
    path = Path(path)
    key = str(path.resolve())
    if key not in INPUTS:
        INPUTS[key] = sha(path)
    if expected is not None and INPUTS[key] != expected:
        raise ValueError("Input hash differs: "+key)
    return INPUTS[key]
def csv_open(name, fields):
    stream = gzip.open(OUT/name, "xt", encoding="utf-8", newline="") if name.endswith(".gz") else (OUT/name).open("x", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    return stream, writer
def write_csv(name, rows):
    assert rows
    stream, writer = csv_open(name, list(rows[0]))
    with stream:
        writer.writerows(rows)
def read(path):
    track(path)
    return json.loads(Path(path).read_text("utf-8"))
def setup():
    global META
    protocol_hash = track(RUN/"protocol.json")
    assert protocol_hash == (RUN/"protocol.sha256").read_text("utf-8").strip()
    track(RUN/"protocol.sha256")
    track(Path(__file__))
    track(PREV/"audit.py")
    complete, preflight = read(CPU/"complete.json"), read(CPU/"preflight.json")
    assert complete["status"]=="COMPLETE_READONLY_CPU_REPLAY"
    assert complete["total_global_forwards"]==2192
    declared = {str(Path(e["path"]).resolve()):e["sha256"] for e in preflight["entries"]+complete["entries"]}
    assert len(declared)==2192
    ds = read(VAL)
    assert len(ds["images"])==548 and len(ds["annotations"])==38759
    assert len({int(g["id"]) for g in ds["annotations"]})==38759
    assert all(g.get("iscrowd",0)==0 and not g.get("ignore",False) for g in ds["annotations"])
    ds_pos = copy.deepcopy(ds)
    for g in ds_pos["annotations"]:
        g["id"] += 1
    images = sorted(ds["images"], key=lambda x:x["id"])
    gt_by_image = defaultdict(list)
    for g in ds["annotations"]:
        gt_by_image[g["image_id"]].append(g)
    meta_path=CPU/"analysis_v1/per_GT_stages_and_transitions.csv"
    track(meta_path)
    with meta_path.open(encoding="utf-8-sig",newline="") as stream:
        META={(int(r["seed"]),int(r["annotation_id"])):(r["size_group"],float(r["short_side_640"])) for r in csv.DictReader(stream)}
    assert len(META)==77518
    bindings={}
    for name,module in list(sys.modules.items()):
        if name=="pycocoeval" or name.startswith("pycocoeval."):
            filename=getattr(module,"__file__",None)
            if filename and Path(filename).is_file():
                bindings[name]={"path":filename,"sha256":track(filename)}
    dump("execution_receipt.json", {"protocol_sha256":protocol_hash,"script_sha256":sha(__file__),
        "python":sys.executable,"torch":torch.__version__,"numpy":np.__version__,
        "CPU_threads":torch.get_num_threads(),"evaluator_bindings":bindings,
        "stage":args.phase,"GT_ID_policy":"memory id+1; original annotation IDs restored in maps",
        "operations":OPERATIONS,"started_unix":time.time()})
    return ds, ds_pos, images, gt_by_image, declared
def load_seed(seed, branch, arms, images, gt_by_image, declared):
    records={arm:{} for arm in arms}
    predictions={arm:[] for arm in arms}
    base={}
    gates=[]
    a_capped=[]
    precision=Counter()
    for pos,im in enumerate(images):
        iid=int(im["id"])
        path=CPU/f"seed{seed}_{branch}"/f"image_{iid:06d}.npz"
        track(path,declared[str(path.resolve())])
        with np.load(path,allow_pickle=False) as z:
            assert int(z["seed"])==seed and str(z["branch"])==branch and int(z["image_id"])==iid
            boxes=z["final_boxes"].copy()
            logits=z["final_logits"].copy()
        assert boxes.shape==(300,4) and logits.shape==(300,10)
        assert np.isfinite(boxes).all() and np.isfinite(logits).all()
        boxes_t=torch.from_numpy(boxes)
        scores=torch.from_numpy(logits).sigmoid()
        edges=A.edges_norm(boxes_t)*torch.tensor([im["width"],im["height"]]*2)
        xywh=torch.cat((edges[:,:2],edges[:,2:]-edges[:,:2]),dim=-1).tolist()
        av,af=scores.flatten().topk(300)
        keep=av>.001
        av,af=av[keep],af[keep]
        aq,al=af//10,af%10
        U=torch.unique(aq,sorted=True)
        positive=(boxes_t[:,2:]>0).all(-1)&torch.isfinite(boxes_t).all(-1)
        offered=int(((scores[U]>.001)&torch.isfinite(scores[U])&positive[U,None]).sum())
        gate={"seed":seed,"branch":branch,"image_id":iid,"U_queries":len(U),"offered":offered,"pass600":offered>=600}
        gates.append(gate)
        aa=gt_by_image[iid]
        original_gt=np.asarray([g["bbox"] for g in aa],dtype=np.float64)
        M=np.asarray(mask_utils.iou(xywh,original_gt.tolist(),[0]*len(aa)),dtype=np.float64)
        assert M.shape==(300,len(aa)) and np.isfinite(M).all()
        assert M.min()>=0 and M.max()<=1+1e-12
        old_edges=original_gt.copy()
        old_edges[:,2:]+=old_edges[:,:2]
        old_M=A.iou_np(edges.numpy(),old_edges)
        for th in THRESHOLDS:
            precision[f"raw_edge_vs_native_bbox_pair_crossings_{th}"]+=int(((old_M>=th/100)!=(M>=th/100)).sum())
        base[iid]={"image":im,"GT":aa,"labels":np.asarray([g["category_id"] for g in aa],int),
                   "M":M,"scores":scores.numpy(),"U":set(map(int,U.tolist()))}
        for j in A.cap_indices(al.numpy()):
            q,c=int(aq[j]),int(al[j])
            a_capped.append({"image_id":iid,"category_id":c,"bbox":xywh[q],"score":float(av[j])})
        for arm in arms:
            if arm=="A":
                values,flat=av,af
            elif arm=="B":
                if offered<600:
                    continue
                values,local=scores[U].flatten().topk(600)
                flat=U[local//10]*10+local%10
                keep=values>.001
                values,flat=values[keep],flat[keep]
            else:
                values,flat=scores.flatten().topk(600)
                keep=values>.001
                values,flat=values[keep],flat[keep]
            q=(flat//10).numpy()
            labels=(flat%10).numpy()
            valid=bool(positive[torch.from_numpy(q)].all())
            gate["export_"+arm+"_valid"]=valid
            if not valid:
                raise ValueError(f"Invalid exported {branch}/{arm} seed{seed} image{iid}")
            if arm in ("B","C") and len(q)!=600:
                gate["pass600"]=False
            capped=np.zeros(len(q),bool)
            capped[A.cap_indices(labels)]=True
            offset=len(predictions[arm])
            records[arm][iid]={"flat":flat.numpy().astype(np.int32),"q":q.astype(np.int16),
                "label":labels.astype(np.int8),"score":values.numpy().copy(),"cap":capped,
                "offset":offset,"matched":np.full((2,len(q)),-1,dtype=np.int32)}
            predictions[arm].extend({"image_id":iid,"category_id":int(c),"bbox":xywh[int(qi)],"score":float(s)}
                                     for qi,c,s in zip(q,labels,values.tolist()))
        if (pos+1)%180==0:
            print(f"PREPARE {branch} seed{seed} {pos+1}/548",flush=True)
    return base,records,predictions,gates,a_capped,precision
def evaluate_arm(ds_pos, seed, branch, arm, pred, records, base):
    print(f"EVALUATE {branch} seed{seed} {arm}, rows={len(pred)}",flush=True)
    OPERATIONS["evaluator_calls"]+=1
    ev,metrics=A.evaluate(ds_pos,pred,f"seed{seed}_{branch}_{arm}_positiveIDs")
    all10_hits={t:set() for t in range(10)}
    evaluated=set()
    max_iou_error=0.
    for e in ev.evalImgs:
        if e is None or list(e["aRng"])!=list(ev.params.areaRng[0]):
            continue
        assert e["maxDet"]==100 and not np.asarray(e["gtIgnore"]).any()
        iid=int(e["image_id"])
        rec=records[iid]
        gtlist=base[iid]["GT"]
        gtpos={int(g["id"]):j for j,g in enumerate(gtlist)}
        dtpos={int(d):j for j,d in enumerate(e["dtIds"])}
        for did in e["dtIds"]:
            did=int(did)
            ix=did-1-rec["offset"]
            assert 0<=ix<len(rec["flat"]) and rec["cap"][ix]
            evaluated.add(did)
        for j,aid_pos in enumerate(e["gtIds"]):
            aid=int(aid_pos)-1
            assert aid in gtpos
            for t,did_f in enumerate(e["gtMatches"][:,j]):
                did=int(did_f)
                if did>0:
                    ej=dtpos[did]
                    assert int(e["dtMatches"][t,ej])==aid_pos and not e["dtIgnore"][t,ej]
                    all10_hits[t].add(aid)
                    if t in (0,5):
                        ix=did-1-rec["offset"]
                        rec["matched"][0 if t==0 else 1,ix]=aid
        # The same native bbox kernel must reproduce each evaluator pair.
        eval_M=ev.ious[(iid,int(e["category_id"]))]
        original_cat=[g for g in gtlist if int(g["category_id"])==int(e["category_id"])]
        assert len(original_cat)==len(e["gtIds"])
        assert [int(g["id"])+1 for g in original_cat]==list(e["gtIds"])
        for di,did in enumerate(e["dtIds"]):
            ix=int(did)-1-rec["offset"]
            for gj,g in enumerate(original_cat):
                max_iou_error=max(max_iou_error,abs(float(eval_M[di,gj])-float(base[iid]["M"][rec["q"][ix],gtpos[int(g["id"])]])))
    assert max_iou_error<=1e-12
    expected_cap=sum(int(r["cap"].sum()) for r in records.values())
    assert expected_cap==len(evaluated)
    class_AP={}
    for k,cid in enumerate(ev.params.catIds):
        pr=ev.eval["precision"][:,:,k,0,2]
        valid=pr[pr>-1]
        class_AP[str(int(cid))]=float(valid.mean()) if valid.size else None
    result={"metrics":metrics,"class_AP":class_AP,"hits":{50:all10_hits[0],75:all10_hits[5]},
            "all10_counts":[len(all10_hits[t]) for t in range(10)],
            "exported":len(pred),"evaluated":len(evaluated),
            "TP50":len(all10_hits[0]),"TP75":len(all10_hits[5]),
            "FP50":len(evaluated)-len(all10_hits[0]),"native_IoU_max_error":max_iou_error}
    print(f"DONE {branch} seed{seed} {arm}: AP={100*metrics['AP']:.6f} TP50={result['TP50']}",flush=True)
    del ev
    gc.collect()
    return result
def state_and_qual(base,rec,j,th,hit):
    t=th/100
    truth=int(base["labels"][j])
    M=base["M"][:,j]
    pool=rec["label"]==truth
    qe=np.where(pool & (M[rec["q"]]>=t))[0]
    qc=qe[rec["cap"][qe]]
    if hit:
        assert len(qc)>0
        state="T"
    elif M.max()<t:
        state="G"
    elif not np.any((M>=t)&(base["scores"][:,truth]>.001)):
        state="S"
    elif not len(qe):
        state="K"
    elif not len(qc):
        state="C"
    else:
        state="M"
    return state,qe,qc
def pair_fate(flat,gt,other,tindex):
    if flat<0:
        return "no_source_match"
    where=np.where(other["flat"]==flat)[0]
    if not len(where):
        return "not_exported"
    ix=int(where[0])
    if not other["cap"][ix]:
        return "cap_removed"
    recipient=int(other["matched"][tindex,ix])
    return "same_GT" if recipient==gt else "capped_unmatched" if recipient<0 else "other_GT"
def old_raw_expected():
    b=read(BC/"B_results.json")
    old=read(PREV/"results.json")
    path=BC/"B_per_GT.csv"
    track(path)
    rows=list(csv.DictReader(path.open(encoding="utf-8-sig",newline="")))
    flags=defaultdict(lambda:defaultdict(set))
    for r in rows:
        for arm in ("B","C"):
            for th in THRESHOLDS:
                if int(r[arm+str(th)]):
                    flags[int(r["seed"])][arm+str(th)].add(int(r["annotation_id"]))
    return b,old,flags
def operation_maps(seed,base,records,results,gt_writer,pair_writer,groups):
    summary={}
    membership=Counter()
    for iid,b in base.items():
        B,C=records["B"][iid],records["C"][iid]
        ib={int(f):j for j,f in enumerate(B["flat"])}
        ic={int(f):j for j,f in enumerate(C["flat"])}
        sb,sc=set(ib),set(ic)
        membership["B_exported"]+=len(sb)
        membership["C_exported"]+=len(sc)
        membership["exported_common"]+=len(sb&sc)
        membership["B_export_only"]+=len(sb-sc)
        membership["C_export_only"]+=len(sc-sb)
        capB={f for f,j in ib.items() if B["cap"][j]}
        capC={f for f,j in ic.items() if C["cap"][j]}
        for arm,rec in (("B",B),("C",C)):
            membership[arm+"_adjacent_export_score_ties"]+=int((np.diff(rec["score"])==0).sum())
            population=b["scores"][sorted(b["U"])] if arm=="B" else b["scores"]
            cutoff=rec["score"][-1]
            membership[arm+"_selection_boundary_tied_images"]+=int(int((population>cutoff).sum())<600<int((population>=cutoff).sum()))
            for label in range(10):
                ix=np.where(rec["label"]==label)[0]
                membership[arm+"_cap_boundary_tied_image_classes"]+=int(len(ix)>100 and rec["score"][ix[99]]==rec["score"][ix[100]])
        membership["capped_common"]+=len(capB&capC)
        membership["B_cap_only"]+=len(capB-capC)
        membership["C_cap_only"]+=len(capC-capB)
        for f in sorted(sb|sc):
            jb,jc=ib.get(f),ic.get(f)
            if jb is not None and jc is not None:
                assert float(B["score"][jb])==float(C["score"][jc])
                membership["export_common_cap_B_only"]+=int(B["cap"][jb] and not C["cap"][jc])
                membership["export_common_cap_C_only"]+=int(C["cap"][jc] and not B["cap"][jb])
            if f in sc-sb:
                membership["C_export_only_query_in_original_U"]+=int(f//10 in b["U"])
                membership["C_export_only_query_absent_original_U"]+=int(f//10 not in b["U"])
            row={"seed":seed,"image_id":iid,"query":f//10,"class":f%10,
                 "score":float(B["score"][jb] if jb is not None else C["score"][jc]),
                 "B_export":int(jb is not None),"C_export":int(jc is not None),
                 "B_cap":int(jb is not None and B["cap"][jb]),"C_cap":int(jc is not None and C["cap"][jc]),
                 "B_export_rank":jb+1 if jb is not None else -1,"C_export_rank":jc+1 if jc is not None else -1}
            for tindex,th in enumerate(THRESHOLDS):
                row["B_match_GT"+str(th)]=int(B["matched"][tindex,jb]) if jb is not None else -1
                row["C_match_GT"+str(th)]=int(C["matched"][tindex,jc]) if jc is not None else -1
                if f in capB&capC:
                    mb,mc=row["B_match_GT"+str(th)],row["C_match_GT"+str(th)]
                    membership["common_cap_recipient_changed_"+str(th)]+=int(mb!=mc)
            pair_writer.writerow(row)
        for c in range(10):
            bf=[int(f) for f in B["flat"] if int(f)%10==c and int(f) in capB&capC]
            cf=[int(f) for f in C["flat"] if int(f)%10==c and int(f) in capB&capC]
            membership["image_class_common_cap_order_changed"]+=int(bf!=cf)
        for tindex,th in enumerate(THRESHOLDS):
            count=summary.setdefault(str(th),{"state_cross":Counter(),"C_new_by_B_state":Counter(),
                  "C_lost_by_C_state":Counter(),"C_new_by_matched_pair_fate_in_B":Counter(),
                  "C_lost_by_matched_pair_fate_in_C":Counter(),"C_new":0,"C_lost":0})
            for j,g in enumerate(b["GT"]):
                aid=int(g["id"])
                hb,hc=aid in results["B"]["hits"][th],aid in results["C"]["hits"][th]
                bs,be,bc=state_and_qual(b,B,j,th,hb)
                cs,ce,cc=state_and_qual(b,C,j,th,hc)
                assert not ((hb or hc) and (bs in ("G","S") or cs in ("G","S")))
                bm=np.where(B["matched"][tindex]==aid)[0]
                cm=np.where(C["matched"][tindex]==aid)[0]
                assert len(bm)<=1 and len(cm)<=1
                bf=int(B["flat"][bm[0]]) if len(bm) else -1
                cf=int(C["flat"][cm[0]]) if len(cm) else -1
                b_fate=pair_fate(bf,aid,C,tindex)
                c_fate=pair_fate(cf,aid,B,tindex)
                count["state_cross"][bs+"|"+cs]+=1
                if hc and not hb:
                    count["C_new"]+=1;count["C_new_by_B_state"][bs]+=1
                    count["C_new_by_matched_pair_fate_in_B"][c_fate]+=1
                if hb and not hc:
                    count["C_lost"]+=1;count["C_lost_by_C_state"][cs]+=1
                    count["C_lost_by_matched_pair_fate_in_C"][b_fate]+=1
                ar=float(g["area"]);size,short=META[seed,aid]
                grow={"seed":seed,"image_id":iid,"annotation_id":aid,"category_id":int(g["category_id"]),
                      "IoU":th/100,"B_state":bs,"C_state":cs,"B_hit":int(hb),"C_hit":int(hc),
                      "B_qual_export":len(be),"B_qual_cap":len(bc),"C_qual_export":len(ce),"C_qual_cap":len(cc),
                      "B_match_flat":bf,"C_match_flat":cf,"B_matched_pair_fate_in_C":b_fate,"C_matched_pair_fate_in_B":c_fate}
                gt_writer.writerow(grow)
                for typ,label in (("all","all"),("class",str(g["category_id"])),("area","small" if ar<1024 else "medium" if ar<9216 else "large"),("size",size)):
                    z=groups[(seed,th,typ,label)]
                    z["GT"]+=1;z["B_TP"]+=int(hb);z["C_TP"]+=int(hc)
                    z["C_new"]+=int(hc and not hb);z["C_lost"]+=int(hb and not hc)
                    if hc and not hb:z["C_new_B_"+bs]+=1
                    if hb and not hc:z["C_lost_C_"+cs]+=1
    return {"membership":dict(membership),"thresholds":summary}
def finish(protocol_hash,extra=None):
    assert sha(RUN/"protocol.json")==protocol_hash
    changed=[p for p,h in INPUTS.items() if sha(p)!=h]
    dump("input_manifest.json",INPUTS)
    verdict={"status":"PASS" if not changed else "FAIL","changed_inputs":changed,
             "input_files_rehashed":len(INPUTS),"protocol_unchanged":True,
             "operations":OPERATIONS,"elapsed_seconds":time.monotonic()-START}
    if extra:verdict.update(extra)
    dump("verification.json",verdict)
    assert not changed
    return verdict
def run_raw(ds_pos,images,gt_by_image,declared,protocol_hash):
    b_old,prior,expected=old_raw_expected()
    gt_fields=["seed","image_id","annotation_id","category_id","IoU","B_state","C_state","B_hit","C_hit","B_qual_export","B_qual_cap","C_qual_export","C_qual_cap","B_match_flat","C_match_flat","B_matched_pair_fate_in_C","C_matched_pair_fate_in_B"]
    pair_fields=["seed","image_id","query","class","score","B_export","C_export","B_cap","C_cap","B_export_rank","C_export_rank","B_match_GT50","C_match_GT50","B_match_GT75","C_match_GT75"]
    gs,gw=csv_open("GT_operation_map.csv.gz",gt_fields)
    ps,pw=csv_open("pair_operation_map.csv.gz",pair_fields)
    summaries={};groups=defaultdict(Counter);allgates=[]
    try:
        for seed in (1,2):
            base,records,pred,gates,Acap,precision=load_seed(seed,"raw",("B","C"),images,gt_by_image,declared)
            assert all(g["pass600"] for g in gates)
            allgates.extend(gates)
            archived=ROOT/f"code/unit_instance_completion_20261021_v1/349_cpu_val_20261023_v1/349_seed{seed}_baseline_predictions.json"
            saved=read(archived)
            assert Acap==saved
            result={}
            for arm in ("B","C"):
                result[arm]=evaluate_arm(ds_pos,seed,"raw",arm,pred.pop(arm),records[arm],base)
                target=b_old["seeds"][str(seed)]["metrics_B"] if arm=="B" else prior["metrics"][str(seed)]["positive_id_top600"]
                assert max(abs(result[arm]["metrics"][k]-target[k]) for k in A.METRICS)<=1e-12
                for th in THRESHOLDS:assert result[arm]["hits"][th]==expected[seed][arm+str(th)]
            op=operation_maps(seed,base,records,result,gw,pw,groups)
            summaries[str(seed)]={"B":{k:v for k,v in result["B"].items() if k!="hits"},
                                  "C":{k:v for k,v in result["C"].items() if k!="hits"},
                                  "operation_map":op,"precision_boundaries":dict(precision)}
            del base,records,pred,result,Acap,saved
            gc.collect()
    finally:
        gs.close();ps.close()
    rows=[]
    fields=["GT","B_TP","C_TP","C_new","C_lost","C_new_B_G","C_new_B_S","C_new_B_K","C_new_B_C","C_new_B_M","C_lost_C_G","C_lost_C_S","C_lost_C_K","C_lost_C_C","C_lost_C_M"]
    for (seed,th,typ,label),z in groups.items():
        rows.append({"seed":seed,"IoU":th/100,"type":typ,"group":label,**{k:z[k] for k in fields}})
    write_csv("by_class_area_size.csv",rows)
    write_csv("preflight_by_image.csv",allgates)
    dump("results.json",{"status":"COMPLETE","protocol_sha256":protocol_hash,"seeds":summaries,
         "meaning":"Sequential operational states plus separate pair-survival/recipient witnesses; not causal shares.",
         "operations":OPERATIONS,"elapsed_seconds":time.monotonic()-START})
    finish(protocol_hash,{"raw_BC_metrics_and_GT_sets_replayed":True,"native_geometry_matches_evaluator":True})
def run_ema(ds_pos,images,gt_by_image,declared,protocol_hash):
    # Raw-map completion is a dependency, not a user-approval gate.
    raw_candidates=sorted(RUN.glob("raw_operations_*/verification.json"))
    assert raw_candidates and any(json.loads(p.read_text("utf-8"))["status"]=="PASS" for p in raw_candidates)
    source=read(CPU/"analysis_v1/results.json")
    reference_path=CPU/"analysis_v1/per_GT_stages_and_transitions.csv"
    track(reference_path)
    reference=list(csv.DictReader(reference_path.open(encoding="utf-8-sig",newline="")))
    prior_hits={(s,th):{int(r["annotation_id"]) for r in reference if int(r["seed"])==s and int(r["EMA_hit"+str(th)])}
                for s in (1,2) for th in THRESHOLDS}
    prepared={};allgates=[]
    for seed in (1,2):
        prepared[seed]=load_seed(seed,"EMA",("A","B","C"),images,gt_by_image,declared)
        allgates.extend(prepared[seed][3])
    write_csv("preflight_by_image.csv",allgates)
    bad=[g for g in allgates if not g["pass600"]]
    dump("preflight.json",{"status":"PASS" if not bad else "NA_EQUAL_BUDGET_GATE_FAILED",
          "min_offered":{str(s):min(g["offered"] for g in allgates if g["seed"]==s) for s in (1,2)},
          "affected_images":bad,"operations":OPERATIONS})
    if bad:
        dump("results.json",{"status":"NA_EQUAL_BUDGET_GATE_FAILED","affected_images":bad,"operations":OPERATIONS})
        finish(protocol_hash)
        return
    results={};rows=[];grouped=[];image_counts={}
    id_by_image={int(im["id"]):j for j,im in enumerate(images)}
    for seed in (1,2):
        base,records,pred,gates,Acap,precision=prepared.pop(seed)
        r={}
        for arm in ("A","B","C"):
            r[arm]=evaluate_arm(ds_pos,seed,"EMA",arm,pred.pop(arm),records[arm],base)
            if arm=="A":
                assert r[arm]["all10_counts"]==source["summaries"][str(seed)+"_EMA"]["GT_hits_all10"]
                for th in THRESHOLDS:assert r[arm]["hits"][th]==prior_hits[seed,th]
        dBA={k:100*(r["B"]["metrics"][k]-r["A"]["metrics"][k]) for k in A.METRICS}
        dCB={k:100*(r["C"]["metrics"][k]-r["B"]["metrics"][k]) for k in A.METRICS}
        image_counts[seed]=np.zeros((548,6),dtype=np.int64)
        localrows=[]
        for iid,b in base.items():
            for g in b["GT"]:
                aid=int(g["id"])
                size,short=META[seed,aid]
                row={"seed":seed,"image_id":iid,"annotation_id":aid,"category_id":int(g["category_id"]),"area_original":float(g["area"]),"size_group":size,"short_side_640":short}
                for th in THRESHOLDS:
                    for a in ("A","B","C"):
                        row[a+str(th)]=int(aid in r[a]["hits"][th])
                image_counts[seed][id_by_image[iid]]+=np.asarray([row[k] for k in ("A50","B50","C50","A75","B75","C75")])
                localrows.append(row)
        rows.extend(localrows)
        for typ,labels in (("class",list(range(10))),("area",["small","medium","large"]),("size",["[0,4)","[4,8)","[8,16)","[16,32)","[32,inf)"])):
            for label in labels:
                sub=[v for v in localrows if (v["category_id"]==label if typ=="class" else v["size_group"]==label if typ=="size" else ("small" if v["area_original"]<1024 else "medium" if v["area_original"]<9216 else "large")==label)]
                grouped.append({"seed":seed,"type":typ,"group":str(label),"GT":len(sub),
                                **{a+str(th):sum(v[a+str(th)] for v in sub) for a in ("A","B","C") for th in THRESHOLDS}})
        set_result={}
        for th in THRESHOLDS:
            aa,bb,cc=(r[a]["hits"][th] for a in ("A","B","C"))
            set_result[str(th)]={"B_new_vs_A":sorted(bb-aa),"B_lost_vs_A":sorted(aa-bb),
             "C_new_vs_B":sorted(cc-bb),"C_lost_vs_B":sorted(bb-cc),
             "B_new_intersect_C_new_vs_A":len((bb-aa)&(cc-aa))}
        results[str(seed)]={"arms":{a:{k:v for k,v in x.items() if k!="hits"} for a,x in r.items()},
            "B_minus_A_pp":dBA,"C_minus_B_pp":dCB,"sets":set_result,
            "class_hypothesis_gate":all(dBA[k]>1e-4 for k in ("AP","APs")) and r["B"]["TP50"]>r["A"]["TP50"],
            "absent_query_gate":all(dCB[k]>1e-4 for k in ("AP","APs")) and r["C"]["TP50"]>r["B"]["TP50"],
            "positive_ID_A_minus_original_ID_CPU_reference_pp":{k:100*(r["A"]["metrics"][k]-source["summaries"][str(seed)+"_EMA"]["metrics"][k]) for k in A.METRICS},
            "precision_boundaries":dict(precision)}
        del base,records,pred,r,Acap,localrows
        gc.collect()
    rng=np.random.default_rng(20261026)
    boot=rng.integers(0,548,size=(1000,548))
    ngt=np.asarray([len(gt_by_image[int(im["id"])]) for im in images])
    denom=ngt[boot].sum(1)
    for seed in (1,2):
        q=image_counts[seed];intervals={}
        for name,j,k in (("B_minus_A_50",1,0),("C_minus_B_50",2,1),("B_minus_A_75",4,3),("C_minus_B_75",5,4)):
            vals=(q[:,j]-q[:,k])[boot].sum(1)/denom
            intervals[name]={"point":float((q[:,j]-q[:,k]).sum()/38759),
                             "image_paired_percentile95":np.quantile(vals,[.025,.975]).tolist()}
        results[str(seed)]["TP_rate_bootstrap"]=intervals
    write_csv("per_GT.csv",rows)
    write_csv("by_class_area.csv",grouped)
    dump("results.json",{"status":"COMPLETE","protocol_sha256":protocol_hash,"seeds":results,
         "joint_class_hypothesis_signal":all(v["class_hypothesis_gate"] for v in results.values()),
         "joint_absent_query_signal":all(v["absent_query_gate"] for v in results.values()),
         "bootstrap_indices_sha256":hashlib.sha256(boot.tobytes()).hexdigest(),
         "operations":OPERATIONS,"elapsed_seconds":time.monotonic()-START})
    finish(protocol_hash,{"EMA_A_GT_sets_and_all10_counts_replayed":True,"native_geometry_matches_evaluator":True})
try:
    DS,DS_POS,IMAGES,GT_BY_IMAGE,DECLARED=setup()
    PH=sha(RUN/"protocol.json")
    if args.phase=="raw":
        run_raw(DS_POS,IMAGES,GT_BY_IMAGE,DECLARED,PH)
    else:
        run_ema(DS_POS,IMAGES,GT_BY_IMAGE,DECLARED,PH)
    print("COMPLETE",args.phase,flush=True)
except BaseException as exc:
    try:
        dump("failure.json",{"type":type(exc).__name__,"message":str(exc),"operations":OPERATIONS,
                             "elapsed_seconds":time.monotonic()-START})
    except Exception:
        pass
    raise
finally:
    STOP.set()
