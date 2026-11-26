import sys, json, os, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v2 as S
from PIL import Image
torch.set_num_threads(2)
ANN=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\annotations\train.json"
IMGDIR=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\images"
CKPT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1\seed1\run\last.pth"
ann=json.load(open(ANN,encoding="utf-8")); imgs={im["id"]:im for im in ann["images"]}
model=S.load_real_model(CKPT)
pid=4113
im=imgs[pid]; pil=Image.open(os.path.join(IMGDIR,im["file_name"])).convert("RGB")
pb,l=S.global_forward_640(model,pil)
b640=S.norm_cxcywh_to_640_xyxy(pb)
print("b640 range x",float(b640[:,0].min()),float(b640[:,2].max()),"y",float(b640[:,1].min()),float(b640[:,3].max()))
rc,cnt=S.select_window(b640)
print("selected",rc,"density grid max",int(cnt.max()),"sum",int(cnt.sum()))
print("density matrix:\n",cnt)
# what is the actual count at the window that has 15 GT?