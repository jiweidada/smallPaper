import os, glob, hashlib, json
base=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文"
target="288eab6c1a82fe9c94f1eede61559aadb48dfe77940b0aee2ea587e355097948"
paths=[]
for r,ds,fs in os.walk(base):
    if "cocoeval.py" in fs: paths.append(os.path.join(r,"cocoeval.py"))
out=[]
for p in sorted(paths):
    h=hashlib.sha256(open(p,'rb').read()).hexdigest()
    out.append({"path":p,"sha256":h,"match":h==target})
print("total cocoeval.py:",len(out))
print("match bound:",sum(1 for x in out if x["match"]))
from collections import Counter
print("distinct SHA prefixes:",Counter(x["sha256"][:8] for x in out))
json.dump(out,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\_local_cocoeval_scan.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
