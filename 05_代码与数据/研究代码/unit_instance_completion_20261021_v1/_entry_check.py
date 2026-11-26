import os, sys, json, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stats_338 as ST
p=os.path.join(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1","338_cache_seed1","000001.pt")
if os.path.exists(p):
    e=torch.load(p,map_location="cpu",weights_only=True)
    print("keys:",sorted(e.keys()))
    print("image_id",e["image_id"],"n_gt",e["n_gt"],"nGW",e["n_G_W"],"nUW",e["n_U_W"],
          "window",e["window_cell"])
    print("sha stored",e["sha256"][:16],"recomputed",ST.sha_payload(e)[:16],
          "match",ST.sha_payload(e)==e["sha256"])
    print("fullp boxes range",float(e["fullp_boxes"][:,0].min()),float(e["fullp_boxes"][:,2].max()))
    print("K_g sums",float(e["K_g"][e["G_W"]].sum(1).max()) if e["G_W"].any() else "no GW")
else:
    print("not yet")
