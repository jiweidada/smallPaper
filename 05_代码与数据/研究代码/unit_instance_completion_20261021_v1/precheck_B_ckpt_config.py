
import torch, json, sys
torch.set_num_threads(2)
cp1 = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1\seed1\run\last.pth"
out={}
for name,path in [("seed1",cp1)]:
    rec={}
    try:
        # weights_only=True safe load (torch 2.5 supports it). Try; fallback to map_location cpu
        try:
            ck = torch.load(path, map_location='cpu', weights_only=True)
            rec['loaded_weights_only']=True
        except Exception as e1:
            rec['weights_only_error']=f"{type(e1).__name__}: {e1}"
            ck = torch.load(path, map_location='cpu', weights_only=False)
            rec['loaded_weights_only']=False
        rec['type']=type(ck).__name__
        if isinstance(ck, dict):
            rec['top_keys']=list(ck.keys())
            for k in ('epoch','last_epoch','args','config','cfg','opt','lr_scheduler'):
                if k in ck:
                    v=ck[k]
                    rec[f'has_{k}']=True
                    if isinstance(v,(dict,list,str,int,float)):
                        rec[f'{k}_value']=str(v)[:2000]
                    else:
                        rec[f'{k}_type']=type(v).__name__
            m=ck.get('model', ck)
            if hasattr(m,'keys'):
                rec['model_state_keys_count']=len(m)
    except Exception as e:
        rec['error']=f"{type(e).__name__}: {e}"
    out[name]=rec
print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
with open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_precheck_B_ckpt_config.json","w",encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
