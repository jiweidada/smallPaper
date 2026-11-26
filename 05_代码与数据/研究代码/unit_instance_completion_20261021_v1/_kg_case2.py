import sys, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v4 as S4
# case2: gx exactly 7.0 -> floor gives ix=7, fx=0 -> weights (1,0) on dx=0 only.
K=S4.bilinear_scatter_window((158.0,184.9,1,1),[128.,128.,192.,192.])
print("nonzeros:",[(i//16,i%16,round(float(K[i]),4)) for i in range(256) if K[i]>0])
# manual with same convention
x0,y0,x1,y1=128.,128.,192.,192.
gx=(158.0-x0)/(x1-x0)*16-0.5; gy=(184.9-y0)/(y1-y0)*16-0.5
print("gx",gx,"gy",gy)
ix=int(np.floor(gx)); iy=int(np.floor(gy)); fx=gx-ix; fy=gy-iy
print("ix",ix,"iy",iy,"fx",fx,"fy",fy)
# weights
for dy in (0,1):
    for dx in (0,1):
        w=(fx if dx else 1-fx)*(fy if dy else 1-fy)
        print(" dx",dx,"dy",dy,"w",round(w,4),"cell",(min(max(iy+dy,0),15),min(max(ix+dx,0),15)))
