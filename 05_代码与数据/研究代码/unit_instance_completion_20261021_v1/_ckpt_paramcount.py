import torch, json
base=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1"
ck=torch.load(base+r"\seed1\run\last.pth",map_location="cpu",weights_only=True)
sd=ck["model"]
tot=sum(v.numel() for v in sd.values() if hasattr(v,"numel"))
print("state_dict tensors:",len(sd),"sum numel:",tot)
# find small tensors
small=[(k,tuple(v.shape)) for k,v in sd.items() if hasattr(v,"numel") and v.numel()<=2]
print("small count:",len(small))
print(small[:10])
# print any non-tensor entries
print("non-tensor:",[k for k,v in sd.items() if not hasattr(v,"numel")][:10])
