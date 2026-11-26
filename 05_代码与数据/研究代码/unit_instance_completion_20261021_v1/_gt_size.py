import sys, json, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v2 as S
from PIL import Image
torch.set_num_threads(2)
ANN=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\annotations\train.json"
IMGDIR=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\images"
ann=json.load(open(ANN,encoding="utf-8")); imgs={im["id"]:im for im in ann["images"]}
by={}
for a in ann["annotations"]: by.setdefault(a["image_id"],[]).append(a)
# GT size distribution in 640 plane
sizes=[]
for pid,items in list(by.items())[:400]:
    im=imgs[pid]; ow,oh=im["width"],im["height"]
    for a in items:
        x,y,w,h=a["bbox"]
        sizes.append((w*640/ow, h*640/oh))
sizes=np.array(sizes)
print("GT w in 640 plane: median",float(np.median(sizes[:,0])),"p90",float(np.percentile(sizes[:,0],90)))
print("GT h in 640 plane: median",float(np.median(sizes[:,1])),"p90",float(np.percentile(sizes[:,1],90)))
print("fraction GT fully inside a 64x64 window (w<=64 and h<=64):",
      float(((sizes[:,0]<=64)&(sizes[:,1]<=64)).mean()))
# for a 64x64 window, a GT of size w,h has chance (64-w)*(64-h)/(640*640) if uniform
print("expected #GT per image fully in a random 64-window:",
      float(np.mean([((sizes[:,0]<=64)&(sizes[:,1]<=64)).mean()*0 + ((64-sizes[:,0]).clip(0)*(64-sizes[:,1]).clip(0)).sum()/(640*640) for _ in [0]])))
