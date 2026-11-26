import sys, os, json, torch, time
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["MKL_THREADING_LAYER"]="SEQUENTIAL"
sys.dont_write_bytecode=True
NATIVE=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\dn_pre_frozen_diagnostic_20261020_v1\native"
EXP=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\fine_residual_screen24_20261009_v2"
sys.path.insert(0,NATIVE); sys.path.insert(0,EXP)
import torch; torch.set_num_threads(2)
from geometry import ResidualGeometryDEIM
from engine.deim.postprocessor import PostProcessor
base=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1"
pj=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_redo1_real_model.json"
out=json.load(open(pj,encoding="utf-8"))
out.setdefault("tests",{})
cfg=json.load(open(os.path.join(base,"seed1","bundle","resolved.yml"),encoding="utf-8"))
m=ResidualGeometryDEIM(cfg["ResidualGeometryDEIM"]["recipe"],arm="baseline")
ck=torch.load(base+r"\seed1\run\last.pth",map_location="cpu",weights_only=True)
m.load_state_dict(ck["model"],strict=True); m.eval()
for p in m.parameters(): p.requires_grad_(False)
before={k:v.clone() for k,v in m.state_dict().items()}
x=torch.rand(1,3,640,640)
t0=time.time()
with torch.no_grad(): o=m(x)
dt=time.time()-t0
after=m.state_dict()
unchanged=all(torch.equal(before[k],after[k]) for k in before) and set(before)==set(after)
gradern=all(p.grad is None for p in m.parameters())
keys=list(o.keys()) if isinstance(o,dict) else type(o).__name__
pl=o.get("pred_logits") if isinstance(o,dict) else None
pb=o.get("pred_boxes") if isinstance(o,dict) else None
out["tests"]["forward_seed1"]={"out_type":type(o).__name__,"keys":keys,
  "pred_logits_shape":list(pl.shape) if pl is not None else None,
  "pred_boxes_shape":list(pb.shape) if pb is not None else None,
  "forward_secs":round(dt,2),"state_dict_unchanged":unchanged,"all_grad_none":gradern,"n_state_keys":len(before)}
out["executed"].append("real forward 1x640x640 seed1 (1 forward)")
pp=PostProcessor(num_classes=10,num_top_queries=300,use_focal_loss=True)
res=pp({"pred_logits":pl,"pred_boxes":pb}, torch.tensor([[640.,640.]]))
out["tests"]["native_postprocessor"]={"num_queries":int(res[0]["labels"].shape[0]),
  "score_min":float(res[0]["scores"].min()),"score_max":float(res[0]["scores"].max()),
  "num_gt_0p001":int((res[0]["scores"]>0.001).sum())}
out["executed"].append("native PostProcessor on real output (1 call)")
json.dump(out,open(pj,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(out["tests"],ensure_ascii=False,indent=2)[:2500])
