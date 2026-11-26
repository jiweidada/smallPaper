
import hashlib, json, os
files = [
 r"C:\Users\jiwei\Desktop\DEIM\engine\deim\postprocessor.py",
 r"C:\Users\jiwei\Desktop\DEIM\engine\deim\matcher.py",
 r"C:\Users\jiwei\Desktop\DEIM\engine\deim\deim_criterion.py",
 r"C:\Users\jiwei\Desktop\DEIM\engine\deim\box_ops.py",
 r"C:\Users\jiwei\Desktop\DEIM\engine\data\dataset\coco_eval.py",
 r"C:\Users\jiwei\Desktop\DEIM\engine\solver\det_engine.py",
 r"C:\Users\jiwei\.conda\envs\yolo\lib\site-packages\faster_coco_eval\core\cocoeval.py",
 r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\相位机制论文_20261018\data\source_snapshot\common.yml",
 r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2\317_A_单位解释残余补全_四臂有限验证方案_v3.txt",
 r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2\332_本地预检六项_交Codex实现规格.txt",
 r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2\333_332勘误_原生后处理与损失配方核实.txt",
]
out={}
for f in files:
    h=hashlib.sha256()
    with open(f,'rb') as fh:
        for c in iter(lambda: fh.read(1<<20), b''): h.update(c)
    out[os.path.basename(f)]={"path":f,"sha256":h.hexdigest(),"bytes":os.path.getsize(f)}
    print(os.path.basename(f), h.hexdigest())
with open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\source_sha256.json","w",encoding="utf-8") as f:
    json.dump(out,f,ensure_ascii=False,indent=2)
