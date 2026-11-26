import os, sys, json, torch
sys.dont_write_bytecode=True
d=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\336_cache_subset_seed1"
files=sorted(os.listdir(d))
print("files:", len(files))
for f in files:
    o=torch.load(os.path.join(d,f),map_location="cpu",weights_only=True)
    b=o["boxes"]
    # validity as xyxy
    valid=((b[:,2]>b[:,0]) & (b[:,3]>b[:,1])).float().mean().item()
    print(f, "n",b.shape[0], "xyxy_valid_frac", round(valid,4),
          "col2 max", round(float(b[:,2].max()),3), "col3 max", round(float(b[:,3].max()),3),
          "GW", int(o["GW"].sum()), "UW", int(o["UW"].sum()), "gt", int(o["GW"].numel()))
