
import hashlib, json, os, sys, time
targets = [
  ("cp_seed1", r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1\seed1\run\last.pth", "741d57a2d6ecb7439de17fedbc8462315ab05309da2ba35b9d38d2c3e2dd9d87"),
  ("cp_seed2", r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1\seed2\run\last.pth", "bcf1d9a7a20115835d9e9df3e09a71419f41d1612c2e7b687c47ff3c41952cf0"),
  ("ann_train", r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\annotations\train.json", "32c1cae852e084430ea74503eb660f0707ec6b5e65d0998ce3ec8b448265c883"),
  ("ann_val", r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val\annotations\val.json", "0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd"),
]
out = []
for key, path, expected in targets:
    t0=time.time()
    h=hashlib.sha256(); n=0
    try:
        with open(path,'rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                h.update(chunk); n+=len(chunk)
        got=h.hexdigest()
        rec={"key":key,"path":path,"bytes_read":n,"sha256":got,"expected":expected,
             "match": got==expected, "secs": round(time.time()-t0,1)}
    except Exception as e:
        rec={"key":key,"path":path,"error":f"{type(e).__name__}: {e}"}
    print(key, rec.get("sha256","ERR"), "match=",rec.get("match"), "secs=",rec.get("secs"))
    out.append(rec)
with open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_precheck_C_sha.json","w",encoding="utf-8") as f:
    json.dump({"env_note":"run under yolo conda env python (torch2.5.1+cpu); runtime dir in DSH sandbox unreadable","records":out}, f, ensure_ascii=False, indent=2)
print("DONE")
