import sys, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
from PIL import Image
import torch.nn.functional as F
# Build an image where pixel value = exact coordinate fraction, but stored at high precision:
# use 16-bit-like by scaling to 0..255 and comparing to the SAME quantised expectation (not analytic).
Hh,Ww=100,200
arr=np.zeros((Hh,Ww,3),dtype=np.uint8)
xs=np.arange(Ww); ys=np.arange(Hh)
arr[...,0]=np.clip(np.round(xs/(Ww-1)*255),0,255)[None,:]
arr[...,1]=np.clip(np.round(ys/(Hh-1)*255),0,255)[:,None]
arr[...,2]=128
pil=Image.fromarray(arr)
base=torch.from_numpy(arr.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
def crop_t(W,size=128,grid=640,ow=200,oh=100):
    x0,y0,x1,y1=[float(v) for v in W]; ww=x1-x0; wh=y1-y0
    u=torch.arange(size,dtype=torch.float32)+0.5
    ex=(x0+u*ww/size)*(ow/grid); ey=(y0+u*wh/size)*(oh/grid)
    xn=(ex/Ww*2-1).view(1,1,size).expand(1,size,size)
    yn=(ey/Hh*2-1).view(1,size,1).expand(1,size,size)
    return F.grid_sample(base,torch.stack([xn,yn],-1),mode="bilinear",padding_mode="border",align_corners=False)
W=[64.,32.,128.,96.]
c=crop_t(W)
# analytic using the SAME quantised image: value at fractional pixel index p = interp between round(p) values
def interp_x(p):
    i0=int(np.floor(p)); i1=min(i0+1,Ww-1); f=p-i0
    v0=np.round(i0/(Ww-1)*255)/255; v1=np.round(i1/(Ww-1)*255)/255
    return v0*(1-f)+v1*f
for u in (0,63.5,127):
    ex=(W[0]+(u+0.5)*(W[2]-W[0])/128)*(Ww/640); p=ex-0.5
    print(f"u={u}: centre_idx={p:.4f} analytic={interp_x(p):.6f} got={float(c[0,0,0,int(u)]):.6f}")