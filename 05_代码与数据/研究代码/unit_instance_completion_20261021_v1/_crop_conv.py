import sys, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
from PIL import Image
import torch.nn.functional as F

# Build image where pixel value encodes its own index, so we can see which source pixel each output samples
Hh,Ww=100,200
img=np.zeros((Hh,Ww,3),dtype=np.float32)
img[...,0]=np.arange(Ww)[None,:]      # R = source x index
img[...,1]=np.arange(Hh)[:,None]      # G = source y index
pil=Image.fromarray(img.astype(np.uint8))  # values >255 clipped; use float instead
t=torch.from_numpy(img).permute(2,0,1).unsqueeze(0)

def crop_dbg(W,size=128,grid=640,ow=200,oh=100):
    x0,y0,x1,y1=[float(v) for v in W]; ww=x1-x0; wh=y1-y0
    u=torch.arange(size,dtype=torch.float32)+0.5
    gx=(x0+u*ww/size)*(ow/grid)-0.5
    gy=(y0+u*wh/size)*(oh/grid)-0.5
    xn=((gx+0.5)/Ww*2-1).view(1,1,size).expand(1,size,size)
    yn=((gy+0.5)/Hh*2-1).view(1,size,1).expand(1,size,size)
    g=torch.stack([xn,yn],-1)
    return F.grid_sample(t,g,mode="bilinear",padding_mode="border",align_corners=False)

c=crop_dbg([64.,32.,128.,96.])
print("R (source x) at row0, cols 0/64/127:", [round(float(c[0,0,0,i]),3) for i in (0,64,127)])
print("G (source y) at col0, rows 0/64/127:", [round(float(c[0,1,i,0]),3) for i in (0,64,127)])
print("EXPECT x: 20 .. 40 ; y: 5 .. 15")
# The 296 formula: edge coord = x0 + (u+.5)*Ww/128 in 640 plane; * orig/640 -> orig pixel EDGE coord; -0.5 -> centre index
# grid_sample align_corners=False expects normalized coord where pixel centre i is at (i+0.5)/W*2-1
# so xn = ((edge*orig/640) )/W*2-1  (NOT +0.5 again)
def crop_correct(W,size=128,grid=640,ow=200,oh=100):
    x0,y0,x1,y1=[float(v) for v in W]; ww=x1-x0; wh=y1-y0
    u=torch.arange(size,dtype=torch.float32)+0.5
    ex=(x0+u*ww/size)*(ow/grid)     # edge coord in orig pixels
    ey=(y0+u*wh/size)*(oh/grid)
    xn=(ex/Ww*2-1).view(1,1,size).expand(1,size,size)
    yn=(ey/Hh*2-1).view(1,size,1).expand(1,size,size)
    return F.grid_sample(t,torch.stack([xn,yn],-1),mode="bilinear",padding_mode="border",align_corners=False)
cc=crop_correct([64.,32.,128.,96.])
print("CORRECT R:",[round(float(cc[0,0,0,i]),3) for i in (0,64,127)])
print("CORRECT G:",[round(float(cc[0,1,i,0]),3) for i in (0,64,127)])