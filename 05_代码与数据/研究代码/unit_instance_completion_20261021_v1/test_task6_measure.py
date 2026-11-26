import sys, json, torch, time
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
torch.set_num_threads(2)
from uic_model import FourArm, C, D
import torch.nn as nn
torch.manual_seed(1)
res=json.load(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_task5_6_7_model.json",encoding="utf-8"))

# FLOP counter for conv2d + linear during forward
flops={"conv2d":0,"linear":0,"matmul":0}
hooks=[]
def conv_hook(mod,inp,out):
    oc,oh,ow=out.shape[1],out.shape[2],out.shape[3]
    k=mod.kernel_size; cin=mod.in_channels//mod.groups
    flops["conv2d"]+= 2*oc*oh*ow*cin*k[0]*k[1]
def lin_hook(mod,inp,out):
    flops["linear"]+= 2*out.numel()*mod.in_features
def attach(m):
    hs=[]
    for mod in m.modules():
        if isinstance(mod,nn.Conv2d): hs.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod,nn.Linear): hs.append(mod.register_forward_hook(lin_hook))
    return hs

B=1; rgb=torch.randn(B,3,128,128)
cell=torch.stack(torch.meshgrid(torch.linspace(1/32,1-1/32,16),torch.linspace(1/32,1-1/32,16),indexing='ij'),-1).reshape(-1,2)
Nc=7
P={"boxes_xyxy_local":torch.rand(B,Nc,4)*0.4+0.3,"scores":torch.rand(B,Nc),"cls":torch.randint(0,C,(B,Nc))}
P["boxes_xyxy_local"][...,2:]=P["boxes_xyxy_local"][...,:2]+0.1

per_arm={}
for a in ["N","S","G","R"]:
    m=FourArm(a); m.eval()
    for k in flops: flops[k]=0
    hs=attach(m)
    with torch.no_grad(): m(rgb,P,cell)
    for h in hs: h.remove()
    # latency: warmup + median of 30 runs single window CPU
    with torch.no_grad():
        for _ in range(3): m(rgb,P,cell)
        ts=[]
        for _ in range(30):
            t0=time.perf_counter(); m(rgb,P,cell); ts.append((time.perf_counter()-t0)*1000)
    ts.sort(); med=ts[len(ts)//2]
    per_arm[a]={"params":sum(p.numel() for p in m.parameters()),
                "flops_conv2d":flops["conv2d"],"flops_linear":flops["linear"],
                "flops_total":flops["conv2d"]+flops["linear"],
                "latency_ms_median":round(med,2),"latency_ms_min":round(ts[0],2),"latency_ms_max":round(ts[-1],2)}
res["task6_measure"]=per_arm
res["task6_notes"]={
  "S_new_params_measured":per_arm["S"]["params"]-per_arm["N"]["params"],
  "S_new_params_317":133888,
  "match": per_arm["S"]["params"]-per_arm["N"]["params"]==133888,
  "flops_scope":"单窗 B=1 RGB128^2，仅计 Conv2d+Linear 的乘加(2*MAC)；不含 attention/softmax 等，且为紧凑代理模型，非真实DEIM",
  "latency_scope":"CPU 单窗，torch 2 threads，30 次中位数；GPU 显存与墙钟本地 NA",
  "proxy_warning":"本实现为按317接口的紧凑代理(参数~1.0M)，非BASE24真实DEIM(日志 n_parameters=10231995)。FLOPs/时延只反映代理，不能外推到真实四臂。"
}
json.dump(res,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_task5_6_7_model.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(per_arm,ensure_ascii=False,indent=2))
