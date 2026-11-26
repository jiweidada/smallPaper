import sys, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import torch.nn.functional as F
Hh,Ww=100,200
img=np.zeros((Hh,Ww,3),dtype=np.float32)
img[...,0]=np.arange(Ww)[None,:]; img[...,1]=np.arange(Hh)[:,None]
t=torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
def crop_edge(W,size=128,grid=640,ow=200,oh=100):
    x0,y0,x1,y1=[float(v) for v in W]; ww=x1-x0; wh=y1-y0
    u=torch.arange(size,dtype=torch.float32)+0.5
    ex=(x0+u*ww/size)*(ow/grid)   # edge coordinate in original pixels per 296 L45
    ey=(y0+u*wh/size)*(oh/grid)
    xn=(ex/Ww*2-1).view(1,1,size).expand(1,size,size)
    yn=(ey/Hh*2-1).view(1,size,1).expand(1,size,size)
    return F.grid_sample(t,torch.stack([xn,yn],-1),mode="bilinear",padding_mode="border",align_corners=False)
c=crop_edge([64.,32.,128.,96.])
print("R:",[round(float(c[0,0,0,i]),2) for i in (0,64,127)])
print("G:",[round(float(c[0,1,i,0]),2) for i in (0,64,127)])
print("expected 20..40 / 5..15")
# also test a tiny 1px-shift sensitivity and border
c2=crop_edge([0.,0.,64.,64.])
print("border R:",[round(float(c2[0,0,0,i]),2) for i in (0,127)],"G:",[round(float(c2[0,1,i,0]),2) for i in (0,127)])
