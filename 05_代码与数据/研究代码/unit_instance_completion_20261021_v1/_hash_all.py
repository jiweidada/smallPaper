import hashlib, os, json, glob
d=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
out={}
for f in sorted(os.listdir(d)):
    p=os.path.join(d,f)
    if os.path.isfile(p) and f.endswith((".py",".json",".md",".txt")):
        h=hashlib.sha256(open(p,'rb').read()).hexdigest()
        out[f]={"sha256":h,"bytes":os.path.getsize(p)}
json.dump(out,open(os.path.join(d,"MANIFEST_sha256.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False,indent=2))
