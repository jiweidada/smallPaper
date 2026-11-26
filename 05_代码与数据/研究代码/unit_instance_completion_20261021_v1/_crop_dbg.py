import sys, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v4 as S4
from PIL import Image
Hh,Ww=100,200
img=np.zeros((Hh,Ww,3),dtype=np.float32)
img[...,0]=np.arange(Ww)[None,:]/(Ww-1)
img[...,1]=np.arange(Hh)[:,None]/(Hh-1)
img[...,2]=0.5
pil=Image.fromarray((img*255).astype(np.uint8))
Wc=[64.,32.,128.,96.]
c=S4.crop128_from_window(pil,Wc,(Ww,Hh))
print("crop R range overall:",float(c[0].min()),float(c[0].max()))
print("crop G range overall:",float(c[1].min()),float(c[1].max()))
print("row0 R first5:",[round(float(v),4) for v in c[0,0,:5]])
print("row64 R first5:",[round(float(v),4) for v in c[0,64,:5]])
# expected: window x from 64..128 in 640 plane -> orig x 64*200/640=20 .. 128*200/640=40
print("expected orig x range:",20,40,"-> R range",20/199,40/199)
print("expected orig y range:",32*100/640,96*100/640,"-> G range",10/99,30/99)
