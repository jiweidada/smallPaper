import sys, os, json, torch, hashlib
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["MKL_THREADING_LAYER"]="SEQUENTIAL"
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\DEIM")
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\fine_residual_screen24_20261009_v2")
import torch._dynamo
torch.set_num_threads(2)
import geometry
from geometry import ResidualGeometryDEIM

base = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1"
out={"item":"补做1 真实模型构建与参数/state_dict 复核","executed":[],"tests":{},"models":{}}
for seed in (1,2):
    rp=os.path.join(base,f"seed{seed}","bundle","resolved.yml")
    cfg=json.load(open(rp,encoding="utf-8"))
    recipe=cfg["ResidualGeometryDEIM"]["recipe"]
    m=ResidualGeometryDEIM(recipe, arm="baseline")
    n=sum(p.numel() for p in m.parameters())
    ck=torch.load(os.path.join(base,f"seed{seed}","run","last.pth"), map_location="cpu", weights_only=True)
    missing,unexpected=m.load_state_dict(ck["model"], strict=False)
    out["models"][f"seed{seed}"]={"params":n,"expected":10231995,"match":n==10231995,
        "missing":len(missing),"unexpected":len(unexpected),
        "missing_keys_sample":list(missing)[:5],"unexpected_keys_sample":list(unexpected)[:5],
        "last_epoch":int(ck.get("last_epoch",-1))}
    out["executed"].append(f"build+load seed{seed}")
print(json.dumps(out,ensure_ascii=False,indent=2))
json.dump(out,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_redo1_real_model.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
