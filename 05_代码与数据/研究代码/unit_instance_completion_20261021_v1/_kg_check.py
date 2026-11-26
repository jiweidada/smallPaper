import os, sys, json, torch, numpy as np
sys.dont_write_bytecode=True
OUT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
p=os.path.join(OUT,"339_cache_seed1","000001.pt")
e=torch.load(p,map_location="cpu",weights_only=True)
print("image_id",e["image_id"],"window",e["window_cell"],"W",[float(x) for x in e["window_xyxy_640"]])
GW=e["G_W"]; K=e["K_g"]; gt=e["gt_boxes_640"]
print("n_GW",int(GW.sum()))
W=[float(x) for x in e["window_xyxy_640"]]
for i in range(gt.shape[0]):
    if bool(GW[i]):
        cx=float((gt[i,0]+gt[i,2])/2); cy=float((gt[i,1]+gt[i,3])/2)
        print(f"GT{i} center=({cx:.1f},{cy:.1f})")
        # expected window-local coords per 340 formula
        gx=(cx-W[0])/(W[2]-W[0])*16-0.5
        gy=(cy-W[1])/(W[3]-W[1])*16-0.5
        print(f"   window-local cell coords: row(y)={gy:.2f} col(x)={gx:.2f}")
        # current cached K nonzero cells
        nz=torch.nonzero(K[i]>0).flatten().tolist()
        rc=[(int(v)//16,int(v)%16) for v in nz]
        print(f"   cached K nonzero (row,col): {rc}")
        # whole-image-640 version
        gx2=cx/640*16-0.5; gy2=cy/640*16-0.5
        print(f"   whole-640 coords: row={gy2:.2f} col={gx2:.2f}")
