import sys, os, json, hashlib
d = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\fine_residual_screen24_20261009_v2"
binding=json.load(open(os.path.join(d,"training_source_binding.json"),encoding="utf-8"))
files=binding["files"]
cands=[
 r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\dn_pre_frozen_diagnostic_20261020_v1\native",
 r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\local_candidates_20261004_v2\native",
 r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\distribution_context_decoder_20261004_v1\native",
]
def sha(p):
    return hashlib.sha256(open(p,'rb').read().replace(b'\r\n',b'\n')).hexdigest()
for root in cands:
    m=0; mm=0; miss=0; firstmm=[]
    for rel,exp in files.items():
        p=os.path.join(root,rel)
        if not os.path.exists(p): miss+=1; continue
        if sha(p)==exp: m+=1
        else:
            mm+=1
            if len(firstmm)<3: firstmm.append(rel)
    print(os.path.basename(os.path.dirname(root))+"/"+os.path.basename(root), "match",m,"mismatch",mm,"missing",miss, firstmm)
