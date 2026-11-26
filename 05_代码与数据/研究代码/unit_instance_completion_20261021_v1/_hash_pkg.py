import os, sys, json, hashlib
pkg=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\单一候选评价依赖补齐与构建核验_20261008_v1\isolated_runtime\pycocoeval"
out={}
for f in sorted(os.listdir(pkg)):
    p=os.path.join(pkg,f)
    if os.path.isfile(p):
        out[f]={"sha256":hashlib.sha256(open(p,'rb').read()).hexdigest(),"bytes":os.path.getsize(p)}
print(json.dumps(out,indent=1))
json.dump(out,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\_win_pycocoeval_files.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
