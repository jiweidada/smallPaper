from pathlib import Path
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
import json, hashlib, csv, time
from collections import defaultdict, Counter
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
OUT=ROOT/"D0_results_v1"
HEAD={1,2,4}
TAIL={3,5,6,7,8,9,10}
CLASS_NAMES={1:"pedestrian",2:"people",3:"bicycle",4:"car",5:"van",6:"truck",7:"tricycle",8:"awning-tricycle",9:"bus",10:"motor"}
METRICS=["same50","head50","tail50","nogeom50","same75","head75","tail75","nogeom75"]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def save(p,value):
    Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def iou_xywh(a,b):
    a=np.asarray(a,dtype=np.float64).reshape(-1,4)
    b=np.asarray(b,dtype=np.float64).reshape(-1,4)
    lt=np.maximum(a[:,None,:2],b[None,:,:2])
    rb=np.minimum(a[:,None,:2]+a[:,None,2:],b[None,:,:2]+b[None,:,2:])
    wh=np.maximum(0,rb-lt)
    inter=wh[:,:,0]*wh[:,:,1]
    aa=a[:,2]*a[:,3];bb=b[:,2]*b[:,3]
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-15)

def outcome(ious,pred_classes,gt_class,threshold):
    if ious.size==0:return dict(same=0,head=0,tail=0,nogeom=1)
    same=bool(np.any((pred_classes==gt_class)&(ious>=threshold)))
    any_geo=bool(np.any(ious>=threshold))
    best=int(pred_classes[int(np.argmax(ious))])
    return {"same":int(same),"head":int(not same and any_geo and best in HEAD),
            "tail":int(not same and any_geo and best in TAIL),
            "nogeom":int(not any_geo)}

def sanity():
    x=iou_xywh([[0,0,10,10]],[[0,0,10,10],[10,0,10,10],[0,0,5,10]])
    assert np.allclose(x,[[1,0,.5]],rtol=0,atol=1e-15)
    # A higher-overlap head prediction does not imply confusion if another correct-label candidate covers the GT.
    assert outcome(np.array([1.,.8]),np.array([4,3]),3,.5)==dict(same=1,head=0,tail=0,nogeom=0)
    assert outcome(np.array([1.,.2]),np.array([4,3]),3,.5)==dict(same=0,head=1,tail=0,nogeom=0)
    assert outcome(np.array([.8]),np.array([6]),3,.5)==dict(same=0,head=0,tail=1,nogeom=0)
    assert outcome(np.array([]),np.array([],int),3,.5)==dict(same=0,head=0,tail=0,nogeom=1)
    return "PASS"

def mean_ap(z):
    z=z[z>=0]
    return float(z.mean()*100) if z.size else None

def class_metrics(armdir):
    with np.load(armdir/"epoch12_EMA_precision_recall.npz") as z:
        p=z["precision"].copy()
    assert p.ndim==5 and p.shape[2]==10
    agg={"AP":mean_ap(p[:,:,:,0,-1]),"APs":mean_ap(p[:,:,:,1,-1])}
    stored=read(armdir/"epoch12_EMA_metrics.json")
    for k,v in agg.items():assert abs(v-stored[k])<1e-8,(k,v,stored[k])
    return agg,[{"category_id":c,"name":CLASS_NAMES[c],"AP":mean_ap(p[:,:,c-1,0,-1]),
                 "APs":mean_ap(p[:,:,c-1,1,-1])} for c in range(1,11)]

def make_rows(gt,pred):
    gs=defaultdict(list);ps=defaultdict(list)
    for g in gt["annotations"]:gs[g["image_id"]].append(g)
    for p in pred:ps[p["image_id"]].append(p)
    rows=[]
    reuse_total=0
    for image in gt["images"]:
        iid=image["id"];gg=gs[iid];pp=ps[iid]
        if not gg:continue
        boxes=np.array([x["bbox"] for x in gg],dtype=np.float64)
        gc=np.array([x["category_id"] for x in gg],dtype=int)
        pb=np.array([x["bbox"] for x in pp],dtype=np.float64).reshape(-1,4)
        pc=np.array([x["category_id"] for x in pp],dtype=int)
        assert np.isfinite(boxes).all() and np.isfinite(pb).all()
        mat=iou_xywh(boxes,pb)
        n=len(gg);nh=int(np.isin(gc,list(HEAD)).sum())
        centers=boxes[:,:2]+boxes[:,2:]/2
        d2=((centers[:,None,:]-centers[None,:,:])**2).sum(-1)
        best_use=Counter()
        for j,g in enumerate(gg):
            c=int(gc[j]);area=float(g["area"])
            if c not in TAIL or area>1024:continue
            frac=nh/max(1,n-1)
            exposure="high" if frac>=.8 else ("low" if frac<=.5 else "middle")
            nearby=int((d2[j]<=4*area).sum()-1)
            ab=int(np.searchsorted([64,256,576,1024],area,side="left"))
            db=int(np.searchsorted([32,96,224],n,side="left"))
            cb=0 if nearby==0 else (1 if nearby<=3 else 2)
            mx=float(mat[j].max()) if len(pp) else 0.
            best=int(mat[j].argmax()) if len(pp) else -1
            if mx>=.5:best_use[best]+=1
            r={"gt_id":g["id"],"image_id":iid,"category_id":c,"area":area,
               "area_bin":ab,"density_bin":db,"crowd_bin":cb,"nearby_GT":nearby,
               "image_GT":n,"image_head_GT":nh,"head_fraction":frac,"exposure":exposure,
               "best_IoU":mx,"best_pred_class":int(pc[best]) if best>=0 else 0,
               "best_pred_index_in_image":best}
            for threshold,tag in [(.5,"50"),(.75,"75")]:
                values=outcome(mat[j],pc,c,threshold)
                for key,val in values.items():r[key+tag]=val
                assert sum(values.values())==1
            rows.append(r)
        reuse_total+=sum(v-1 for v in best_use.values() if v>1)
    return rows,{"extra_GT_sharing_the_same_best_geometric_candidate_at_50":reuse_total,
                 "scope":"Tail-small GT only; per-GT opportunities are not one-to-one TP or AP."}

def summarize(rows):
    out={}
    for c in sorted(TAIL):
        rr=[r for r in rows if r["category_id"]==c]
        out[str(c)]={"name":CLASS_NAMES[c],"N_GT":len(rr),"N_images":len({r["image_id"] for r in rr})}
        for metric in METRICS:
            out[str(c)][metric+"_N"]=sum(r[metric] for r in rr)
            out[str(c)][metric+"_pct"]=float(np.mean([r[metric] for r in rr])*100) if rr else None
        out[str(c)]["exposure_counts"]=dict(Counter(r["exposure"] for r in rr))
    out["all_tail_small"]={"N_GT":len(rows),"N_images":len({r["image_id"] for r in rows}),
                           "exposure_counts":dict(Counter(r["exposure"] for r in rows))}
    for metric in METRICS:
        out["all_tail_small"][metric+"_N"]=sum(r[metric] for r in rows)
        out["all_tail_small"][metric+"_pct"]=float(np.mean([r[metric] for r in rows])*100) if rows else None
    return out

def standardized(rows,image_ids,exclude_van=False):
    use=[r for r in rows if r["exposure"] in ("high","low") and not(exclude_van and r["category_id"]==5)]
    strata=defaultdict(lambda:{"high":[],"low":[]})
    for r in use:
        key=(r["category_id"],r["area_bin"],r["density_bin"],r["crowd_bin"])
        strata[key][r["exposure"]].append(r)
    retained={k:v for k,v in strata.items() if len(v["high"])>=5 and len(v["low"])>=5}
    class_mass=defaultdict(float)
    for k,v in retained.items():class_mass[k[0]]+=min(len(v["high"]),len(v["low"]))
    classes=sorted(class_mass)
    index={iid:j for j,iid in enumerate(image_ids)}
    den=np.zeros((2,len(image_ids)));nums=np.zeros((2,len(image_ids),len(METRICS)))
    support={"high_GT":0,"low_GT":0,"high_images":set(),"low_images":set(),
             "supported_classes":classes,"retained_strata":len(retained)}
    for key,groups in retained.items():
        mass=min(len(groups["high"]),len(groups["low"]))
        for a,exposure in enumerate(["high","low"]):
            coeff=mass/len(groups[exposure])/class_mass[key[0]]/len(classes)
            for r in groups[exposure]:
                j=index[r["image_id"]]
                den[a,j]+=coeff
                nums[a,j]+=coeff*np.array([r[m] for m in METRICS])
            support[exposure+"_GT"]+=len(groups[exposure])
            support[exposure+"_images"].update(r["image_id"] for r in groups[exposure])
    support["high_images"]=len(support["high_images"]);support["low_images"]=len(support["low_images"])
    adequate=(support["high_GT"]>=200 and support["low_GT"]>=200 and
              support["high_images"]>=20 and support["low_images"]>=20 and len(classes)>=3)
    support["adequate_for_prespecified_screen"]=adequate
    table={}
    if classes:
        rng=np.random.default_rng(20261101)
        boot=rng.multinomial(len(image_ids),np.full(len(image_ids),1/len(image_ids)),size=2000)
        bn=boot@den.T
        bnum=np.einsum("bi,aik->bak",boot,nums,optimize=True)
        for k,metric in enumerate(METRICS):
            rates=nums[:,:,k].sum(1)/den.sum(1)
            vals=np.divide(bnum[:,:,k],bn,out=np.full_like(bn,np.nan,dtype=float),where=bn>0)
            delta=(vals[:,0]-vals[:,1])*100
            finite=delta[np.isfinite(delta)]
            table[metric]={"high_pct":float(rates[0]*100),"low_pct":float(rates[1]*100),
                           "high_minus_low_pp":float((rates[0]-rates[1])*100),
                           "image_bootstrap_95pct_interval_pp":np.percentile(finite,[2.5,97.5]).tolist() if len(finite) else None,
                           "valid_bootstrap_replicates":len(finite)}
    selected=adequate and table["head50"]["high_minus_low_pp"]>=5 and table["same50"]["high_minus_low_pp"]<=-5 if table else False
    return {"exclude_van":exclude_van,"support":support,"standardized_metrics":table,
            "practical_signal_this_seed":bool(selected),
            "interpretation":"Observational composition association; no training causal inference, no method-novelty or deployable AP claim."}

def run_arm(label,folder):
    armdir=BASE/folder/"completed_results/B/arm_B"
    names=["epoch12_EMA_GT.json","epoch12_EMA_predictions.json","epoch12_EMA_precision_recall.npz","epoch12_EMA_metrics.json"]
    hashes={str(armdir/n):sha(armdir/n) for n in names}
    gt=read(armdir/names[0]);pred=read(armdir/names[1])
    aggregate,ap=class_metrics(armdir)
    assert len(gt["images"])==548 and len(gt["annotations"])==38602
    rows,reuse=make_rows(gt,pred)
    for p,h in hashes.items():assert sha(p)==h
    with (OUT/(label+"_tail_small_rows.csv")).open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    images=[r["id"] for r in gt["images"]]
    result={"label":label,"inputs_sha256":hashes,"image_count":len(images),
            "GT_count":len(gt["annotations"]),"predictions":len(pred),"aggregate_AP_reconstruction":aggregate,
            "class_AP_from_saved_precision":ap,"raw_tail_small":summarize(rows),"candidate_reuse":reuse,
            "primary_standardization":standardized(rows,images),
            "prespecified_leave_van_out":standardized(rows,images,True)}
    save(OUT/(label+"_summary.json"),result)
    return result

def main():
    t0=time.monotonic()
    protocol=read(ROOT/"01_D0诊断协议_冻结.json")
    assert protocol["frozen_before_class_stratified_analysis"]
    assert sanity()=="PASS"
    OUT.mkdir(exist_ok=False)
    save(OUT/"started.json",{"recorded_utc":datetime.now(timezone.utc).isoformat(),
          "protocol_sha256":sha(ROOT/"01_D0诊断协议_冻结.json"),"script_sha256":sha(__file__),
          "new_model_forwards":0,"new_optimizer_updates":0})
    a=run_arm("primary_A43_seed20261101","A43_ECDet三组蒸馏最小验证_20261031_v1")
    b=run_arm("consistency_A42_seed20261031","A42_ECDet三组最小验证_20261031_v1")
    pa=a["primary_standardization"];pb=b["primary_standardization"]
    dirs=False
    if pa["standardized_metrics"] and pb["standardized_metrics"]:
        dirs=pb["standardized_metrics"]["head50"]["high_minus_low_pp"]>0 and pb["standardized_metrics"]["same50"]["high_minus_low_pp"]<0
    passes=pa["practical_signal_this_seed"] and pb["support"]["adequate_for_prespecified_screen"] and dirs
    final={"status":"COMPLETE","recorded_utc":datetime.now(timezone.utc).isoformat(),
       "seconds":time.monotonic()-t0,"sanity_cases":"PASS",
       "protocol_sha256":sha(ROOT/"01_D0诊断协议_冻结.json"),"script_sha256":sha(__file__),
       "prespecified_practical_screen_pass":bool(passes),"consistency_seed_direction_match":bool(dirs),
       "training_justified_by_diagnostic_alone":False,"mainline_validated":False,
       "new_model_forwards":0,"new_optimizer_updates":0,"remote_access":False,
       "results":{"primary":a,"consistency":b},
       "scope":"Two saved fixed12 original detectors, no retraining or new model evaluation; same validation GT, not independent GT replicates."}
    save(OUT/"D0_complete.json",final)
    print(json.dumps({"status":"COMPLETE","seconds":final["seconds"],"screen_pass":bool(passes),
       "primary_raw":a["raw_tail_small"]["all_tail_small"],"primary":pa,
       "primary_leave_van_out":a["prespecified_leave_van_out"],
       "consistency":pb,"new_model_forwards":0,"new_optimizer_updates":0},ensure_ascii=False,allow_nan=False))

if __name__=="__main__":main()

