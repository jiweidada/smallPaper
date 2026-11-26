import sys, os, json, hashlib
d = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\fine_residual_screen24_20261009_v2"
binding=json.load(open(os.path.join(d,"training_source_binding.json"),encoding="utf-8"))
files=binding["files"]
root=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\dn_pre_frozen_diagnostic_20261020_v1\native"
def sha(p): return hashlib.sha256(open(p,'rb').read().replace(b'\r\n',b'\n')).hexdigest()
# list what exists and its status for the key modules
key=[k for k in files if any(s in k for s in ["deim/deim.py","deim/dfine_decoder.py","deim/hybrid_encoder.py","backbone/hgnetv2.py","deim/deim_criterion.py","deim/matcher.py","deim/postprocessor.py","deim/box_ops.py","core/__init__.py","core/yaml_config.py","deim/denoising.py","misc/dist_utils.py"])]
for k in key:
    p=os.path.join(root,k)
    if not os.path.exists(p): print("MISSING", k); continue
    h=sha(p); print("OK " if h==files[k] else "DIFF", h[:12], "exp",files[k][:12], k)
print("\n--- existing engine files in tree ---")
for r,ds,fs in os.walk(os.path.join(root,"engine")):
    for f in fs:
        if f.endswith(".py"): print(os.path.relpath(os.path.join(r,f),root).replace("\\","/"))
