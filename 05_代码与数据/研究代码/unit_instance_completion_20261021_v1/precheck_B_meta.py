
import torch, json
torch.set_num_threads(2)
paths = {
 "seed1": r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1\seed1\run\last.pth",
 "seed2": r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1\seed2\run\last.pth",
}
def summarize(o, depth=0):
    if isinstance(o, dict):
        return {k: summarize(v, depth+1) for k,v in list(o.items())[:40]}
    if isinstance(o, (list,tuple)):
        return [summarize(x,depth+1) for x in o[:10]]
    if isinstance(o, torch.Tensor):
        return f"Tensor{tuple(o.shape)}"
    if isinstance(o,(int,float,str,bool)) or o is None:
        return o
    return type(o).__name__
out={}
for name,path in paths.items():
    ck=torch.load(path, map_location='cpu', weights_only=True)
    rec={}
    for k in ['date','name','last_epoch','best_stat','fine_residual_geometry_identity']:
        if k in ck: rec[k]=summarize(ck[k])
    for k in ['criterion','postprocessor','ema','scaler','lr_warmup_scheduler','optimizer']:
        if k in ck:
            rec[k]=summarize(ck[k])
    out[name]=rec
print(json.dumps(out, ensure_ascii=False, indent=2)[:8000])
with open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_precheck_B_ckpt_meta.json","w",encoding="utf-8") as f:
    json.dump(out,f,ensure_ascii=False,indent=2)
