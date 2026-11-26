import os, sys, json, torch
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache as S
from PIL import Image
torch.set_num_threads(2)
ANN=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\annotations\train.json"
IMGDIR=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\images"
CKPT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1\seed1\run\last.pth"
ann=json.load(open(ANN,encoding="utf-8")); imgs={im["id"]:im for im in ann["images"]}
model=S.load_real_model(CKPT)
pid=269
pil=Image.open(os.path.join(IMGDIR,imgs[pid]["file_name"])).convert("RGB")
b1,l1=S.global_forward_640(model,pil)
b2,l2=S.global_forward_640(model,pil)
print("boxes equal:", torch.equal(b1,b2))
print("logits equal:", torch.equal(l1,l2))
print("logits maxdiff:", float((l1-l2).abs().max()))
print("boxes maxdiff:", float((b1-b2).abs().max()))
print("box sha equal:", S.sha_tensor(b1)==S.sha_tensor(b2))
print("logit sha equal:", S.sha_tensor(l1)==S.sha_tensor(l2))
