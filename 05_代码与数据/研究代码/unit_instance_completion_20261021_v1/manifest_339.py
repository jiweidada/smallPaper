import os, sys, json, time, hashlib
import torch
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import build_339 as B
OUT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
def sha_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda: f.read(1<<20), b""): h.update(c)
    return h.hexdigest()
imgs,by,order=B.load_meta()
man={"item":"339 manifest","created":time.strftime("%Y-%m-%dT%H:%M:%S"),
 "window_fix":"v3: cnt[row=y][col=x]; select_window returns (row=y,col=x); window_xyxy maps (row,col)->x=col*64,y=row*64",
 "definitions_source":{"stage1_cache_v3.py":sha_file(os.path.join(OUT,"stage1_cache_v3.py")),
   "stage1_cache_v2.py_SUPERSEDED":sha_file(os.path.join(OUT,"stage1_cache_v2.py")),
   "uic_m0.py":sha_file(os.path.join(OUT,"uic_m0.py"))},
 "sources":{}, "entries_per_seed":{}, "elapsed_secs":{"seed1":785.9,"seed2":None}}
for name,p in [("checkpoint_seed1",B.CKPT[1]),("checkpoint_seed2",B.CKPT[2]),
               ("annotation_train",B.ANN),("sample_plan",B.SP)]:
    man["sources"][name]={"path":p,"sha256":sha_file(p)}
for seed in (1,2):
    d=os.path.join(OUT,f"339_cache_seed{seed}")
    fs=sorted(f for f in os.listdir(d) if f.endswith(".pt"))
    man["entries_per_seed"][f"seed{seed}"]={"count":len(fs),"dir":d,
      "total_bytes":sum(os.path.getsize(os.path.join(d,f)) for f in fs)}
man["total_entries"]=sum(v["count"] for v in man["entries_per_seed"].values())
man["disk_bytes"]=sum(v["total_bytes"] for v in man["entries_per_seed"].values())
json.dump(man,open(os.path.join(OUT,"339_manifest.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(man["entries_per_seed"],ensure_ascii=False,indent=1))
