import sys, os, json, torch
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
p=os.path.join(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1","340_cache_seed1","000001.pt")
e=torch.load(p,map_location="cpu",weights_only=True)
print("keys has crop128:", "crop128" in e)
print("crop shape", list(e["crop128"].shape), e["crop128"].dtype)
print("K_g sum for GW:", [round(float(v),6) for v in e["K_g"][e["G_W"]].sum(1).tolist()] if e["G_W"].any() else "no GW")
nz=[(i//16,i%16) for i in range(256) if e["K_g"][e["G_W"]][0][i]>0] if e["G_W"].any() else []
print("first GW target K_g nonzero cells:", nz)
print("W:", [float(x) for x in e["window_xyxy_640"]])
