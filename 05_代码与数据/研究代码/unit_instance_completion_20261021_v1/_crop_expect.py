import sys, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v4 as S4
from PIL import Image
Hh,Ww=100,200
img=np.zeros((Hh,Ww,3),dtype=np.float32)
img[...,0]=np.arange(Ww)[None,:]/1.0   # R = source pixel x index (0..199)
img[...,1]=np.arange(Hh)[:,None]/1.0
# use float tensor directly via a PIL-less path: build a fake PIL by scaling to 0..255 then dividing back
pil=Image.fromarray(np.stack([img[...,0]/199*255, img[...,1]/99*255, np.full((Hh,Ww),128)],-1).astype(np.uint8))
c=S4.crop128_from_window(pil,[64.,32.,128.,96.],(Ww,Hh))
# because of uint8 quantization, recover ~ value*199
R=[float(c[0,0,i])*199 for i in (0,64,127)]
G=[float(c[1,i,0])*99 for i in (0,64,127)]
print("R source-index approx:",[round(v,2) for v in R])
print("G source-index approx:",[round(v,2) for v in G])
# 296: edge px = x0 + (u+0.5)*Ww/128 scaled by ow/640, then -0.5 -> centre index
x0,ww=64.,64.; ow=200
for u in (0,63.5,127):
    ex=(x0+(u+0.5)*ww/128)*(ow/640)
    print(f"u={u}: edge={ex:.3f} centre_idx={ex-0.5:.3f}")