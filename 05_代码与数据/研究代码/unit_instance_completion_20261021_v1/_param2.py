import sys, os, json, torch
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="2"
sys.dont_write_bytecode=True
NATIVE=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\dn_pre_frozen_diagnostic_20261020_v1\native"
EXP=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\fine_residual_screen24_20261009_v2"
sys.path.insert(0,NATIVE); sys.path.insert(0,EXP)
import torch; torch.set_num_threads(2)
from geometry import ResidualGeometryDEIM
base=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1"
cfg=json.load(open(os.path.join(base,"seed1","bundle","resolved.yml"),encoding="utf-8"))
m=ResidualGeometryDEIM(cfg["ResidualGeometryDEIM"]["recipe"],arm="baseline")
print("params:",sum(p.numel() for p in m.parameters()))
names=[n for n,_ in m.named_parameters()]
print("anchors-ish params:",[n for n in names if "anchor" in n or "valid_mask" in n])
print("decoder buffers:",[n for n,_ in m.decoder.named_buffers()][:10])
# what params are in model but not in ckpt (should be none)
ck=torch.load(base+r"\seed1\run\last.pth",map_location="cpu",weights_only=True)
sd=ck["model"]
print("params not in ckpt:",[n for n in names if n not in sd][:10])
# count params that are buffers in ckpt-sized model? compare with log 10231995: what if 2 = anchors param?
print("numel of decoder.anchors if param:", sum(p.numel() for n,p in m.named_parameters() if n=="decoder.anchors"))
print("numel decoder buffers total:",sum(b.numel() for b in m.decoder.buffers()))
