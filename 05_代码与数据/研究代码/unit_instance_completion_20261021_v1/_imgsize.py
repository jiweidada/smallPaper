import sys, json, os, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v2 as S
from PIL import Image
ANN=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\annotations\train.json"
IMGDIR=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\images"
ann=json.load(open(ANN,encoding="utf-8")); imgs={im["id"]:im for im in ann["images"]}
for pid in [4113,269,5429]:
    im=imgs[pid]
    p=os.path.join(IMGDIR,im["file_name"])
    pil=Image.open(p); 
    print(pid,"ann",im["width"],im["height"],"actual",pil.size)
