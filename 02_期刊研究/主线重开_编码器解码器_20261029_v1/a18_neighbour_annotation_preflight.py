"""One-shot, annotation-only resource preflight. No model imports or predictions."""
import json, hashlib, math, os, sys, time
from pathlib import Path
from fractions import Fraction as F
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8")
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"A18_邻近实例证据与编码机制审计_v1"
RULES=OUT/"cohort_preflight_rules.json"
DATA=Path(r"D:\数据集\visdrone2019\VisDrone2019-DET-val")
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def intersects(a,b):
    return min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1])
def outward(b): return [math.floor(b[0]),math.floor(b[1]),math.ceil(b[2]),math.ceil(b[3])]
def rounded_half_up(v): return math.floor(v+F(1,2))
def fracstr(v): return str(v)
def encode_box(b): return [fracstr(v) for v in b]
def main():
    rules=json.loads(RULES.read_text(encoding="utf-8"))
    assert rules["new_model_forwards"]==0
    lock=OUT/"annotation_started.json"
    with lock.open("x",encoding="utf-8") as f:
        json.dump({"pid":os.getpid(),"rules_sha256":digest(RULES),"started_unix":time.time()},f)
    start=time.monotonic()
    previous=ROOT/"A17_新编码器机制与多实例映射审计_v1"/"cached_cell_per_image.json"
    exclusion=sorted({r["prefix"] for r in json.loads(previous.read_text(encoding="utf-8"))})
    assert len(exclusion)==37
    images=sorted((DATA/"images").glob("*.jpg"))
    assert images and all((DATA/"annotations"/(p.stem+".txt")).exists() for p in images)
    inputs=[]; per_image=[]; pairs=[]
    counts={"images":len(images),"excluded_prefixes":len(exclusion),"excluded_images":0,
            "eligible_images":0,"candidate_oriented_pairs":0,"unseen_images":0,
            "unseen_tiny_GT":0}
    for im in images:
        ann=DATA/"annotations"/(im.stem+".txt")
        with Image.open(im) as image: width,height=image.size
        inputs.append({"image":im.name,"width":width,"height":height,
                       "annotation_sha256":digest(ann),"image_bytes":im.stat().st_size})
        prefix=im.stem.split("_")[0]
        if prefix in exclusion:
            counts["excluded_images"]+=1
            per_image.append({"image":im.name,"prefix":prefix,"excluded":True})
            continue
        counts["unseen_images"]+=1
        raw=[]
        for uid,line in enumerate(ann.read_text(encoding="utf-8-sig").splitlines()):
            if not line.strip(): continue
            cols=[int(x.strip()) for x in line.split(",") if x.strip()]
            assert len(cols)>=8
            x,y,w,h,score,cat,trunc,occ=cols[:8]
            l=max(0,min(width,x));r=max(0,min(width,x+w))
            t=max(0,min(height,y));b=max(0,min(height,y+h))
            if r<=l or b<=t: continue
            box=[F(l*800,width),F(t*800,height),F(r*800,width),F(b*800,height)]
            area=(box[2]-box[0])*(box[3]-box[1])
            raw.append({"uid":uid,"box":box,"area":area,"cat":cat,
                        "valid":score>0 and 1<=cat<=10,
                        "cx":(box[0]+box[2])/2,"cy":(box[1]+box[3])/2,
                        "occ":occ,"trunc":trunc})
        tiny=[r for r in raw if r["valid"] and r["area"]<64]
        counts["unseen_tiny_GT"]+=len(tiny)
        eligible=[]
        for a in tiny:
            guard=outward(a["box"])
            for b in tiny:
                if a["uid"]==b["uid"] or a["cat"]!=b["cat"]: continue
                if max(a["area"],b["area"])>4*min(a["area"],b["area"]): continue
                d2=(a["cx"]-b["cx"])**2+(a["cy"]-b["cy"])**2
                if d2*d2>16*a["area"]*b["area"]: continue
                nb=outward(b["box"])
                if intersects(guard,nb): continue
                if any(r["uid"]!=b["uid"] and intersects(nb,r["box"]) for r in raw): continue
                bw,bh=nb[2]-nb[0],nb[3]-nb[1]
                bx=rounded_half_up(2*a["cx"]-nb[2])
                by=rounded_half_up(2*a["cy"]-nb[3])
                bg=[bx,by,bx+bw,by+bh]
                if not (0<=bg[0]<bg[2]<=800 and 0<=bg[1]<bg[3]<=800): continue
                if any(intersects(bg,r["box"]) for r in raw): continue
                assert not intersects(guard,bg)
                dn=math.sqrt(float(d2))/math.pow(float(a["area"]*b["area"]),0.25)
                row={"image":im.name,"prefix":prefix,"target_uid":a["uid"],
                     "neighbour_uid":b["uid"],"category":a["cat"],
                     "target_box800_exact":encode_box(a["box"]),
                     "neighbour_box800_exact":encode_box(b["box"]),
                     "target_guard":guard,"neighbour_mask":nb,"background_mask":bg,
                     "normalized_distance":dn,"target_size":math.sqrt(float(a["area"])),
                     "neighbour_size":math.sqrt(float(b["area"])),
                     "target_occlusion":a["occ"],"target_truncation":a["trunc"]}
                eligible.append(row)
        eligible.sort(key=lambda r:(r["normalized_distance"],r["target_uid"],r["neighbour_uid"]))
        counts["candidate_oriented_pairs"]+=len(eligible)
        counts["eligible_images"]+=bool(eligible)
        per_image.append({"image":im.name,"prefix":prefix,"excluded":False,
                          "tiny_GT":len(tiny),"eligible_oriented_pairs":len(eligible)})
        if eligible: pairs.append(eligible[0])
    selected={}
    for row in sorted(pairs,key=lambda r:r["image"]):
        selected.setdefault(row["prefix"],row)
    counts["eligible_independent_prefixes"]=len(selected)
    write("annotation_inputs.json",inputs)
    write("per_image_feasibility.json",per_image)
    write("eligible_image_pairs.json",pairs)
    write("selected_prefix_candidates.json",list(selected.values()))
    write("cohort_preflight_summary.json",{
        "counts":counts,"rules_sha256":digest(RULES),
        "previous_prediction_exclusion_sha256":digest(previous),
        "excluded_prefixes":exclusion,"elapsed_seconds":time.monotonic()-start,
        "new_model_forwards":0,"new_fits":0,"main_model_updates":0,
        "prediction_effects_computed":False,"feasibility_only":True,
        "next":"Freeze exact independent question, counterfactual controls and feature-localization protocol before any new model call; counts are not failure evidence.",
        "exit_status":"COMPLETE"})
    print(json.dumps({"status":"COMPLETE","counts":counts,"new_model_forwards":0},ensure_ascii=False))
if __name__=="__main__": main()

