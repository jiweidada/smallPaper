import sys, json, hashlib, os
sys.dont_write_bytecode=True
d = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\fine_residual_screen24_20261009_v2"
binding=json.load(open(os.path.join(d,"training_source_binding.json"),encoding="utf-8"))
files=binding["files"]
root=r"C:\Users\jiwei\Desktop\DEIM"
def sha(p):
    data=open(p,'rb').read().replace(b'\r\n',b'\n')
    return hashlib.sha256(data).hexdigest()
match=0; mismatch=[]; missing=[]
for rel,exp in files.items():
    p=os.path.join(root,rel)
    if not os.path.exists(p): missing.append(rel); continue
    got=sha(p)
    if got==exp: match+=1
    else: mismatch.append((rel,exp[:12],got[:12]))
print("binding file count:", len(files))
print("match:", match, "mismatch:", len(mismatch), "missing:", len(missing))
print("first mismatches:", mismatch[:8])
print("first missing:", missing[:8])
print("deim.py expected", files.get("engine/deim/deim.py","")[:16], "got", sha(os.path.join(root,"engine/deim/deim.py"))[:16] if os.path.exists(os.path.join(root,"engine/deim/deim.py")) else "NA")
