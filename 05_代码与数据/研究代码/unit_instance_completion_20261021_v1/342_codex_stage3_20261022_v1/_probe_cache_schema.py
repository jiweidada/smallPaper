
import torch, time, os, json
p = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\342_codex_stage3_20261022_v1\342_cache_val_v2_seed1\342_000000.pt"
print("size MB", round(os.path.getsize(p)/1e6,2))
t=time.perf_counter()
e = torch.load(p, map_location="cpu", weights_only=True)
print("load s", round(time.perf_counter()-t,3))
for k in sorted(e):
    v = e[k]
    if torch.is_tensor(v):
        print(k, tuple(v.shape), v.dtype, int(v.numel()*v.element_size()/1e3), "KB")
    else:
        s = str(v)
        print(k, type(v).__name__, s[:80])
