import sys, json, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v4 as S4
from PIL import Image
res={"item":"340 K_g窗内网格与crop单元测试","tests":{}}

def manual_neighbours(cx,cy,W,side=16):
    x0,y0,x1,y1=[float(v) for v in W]
    gx=(cx-x0)/(x1-x0)*side-0.5; gy=(cy-y0)/(y1-y0)*side-0.5
    ix=int(np.floor(gx)); iy=int(np.floor(gy)); fx=gx-ix; fy=gy-iy
    out={}
    for dy in (0,1):
        for dx in (0,1):
            w=(fx if dx else 1-fx)*(fy if dy else 1-fy)
            if w<=0: continue        # only positive-weight neighbours
            xi=min(max(ix+dx,0),side-1); yi=min(max(iy+dy,0),side-1)
            out[(yi,xi)]=out.get((yi,xi),0.0)+w
    return out,gx,gy

W=[128.,128.,192.,192.]
cases=[(163.7,175.7),(158.0,184.9),(182.8,173.6),(128.5,128.5),(191.4,191.4),(160.0,160.0)]
rows=[]
for cx,cy in cases:
    K=S4.bilinear_scatter_window((cx,cy,1,1),W)
    nz={ (i//16,i%16):float(K[i]) for i in range(256) if K[i]>1e-15 }
    exp,gx,gy=manual_neighbours(cx,cy,W)
    ok=set(nz)==set(exp) and all(abs(nz[k]-exp[k])<1e-9 for k in exp)
    rows.append({"center":[cx,cy],"gx":round(gx,3),"gy":round(gy,3),
                 "nz":{f"{k[0]},{k[1]}":round(v,4) for k,v in sorted(nz.items())},
                 "sum":round(float(K.sum()),9),"match_manual":ok,
                 "all_in_range":all(0<=k[0]<=15 and 0<=k[1]<=15 for k in nz)})
res["tests"]["kg_window_local"]={"cases":rows,"all_match":all(r["match_manual"] for r in rows),
  "all_sum1":all(abs(r["sum"]-1)<1e-9 for r in rows)}

def whole640(cx,cy,side=16,size=640):
    gx=cx/size*side-0.5; gy=cy/size*side-0.5
    ix=int(np.floor(gx)); iy=int(np.floor(gy)); fx=gx-ix; fy=gy-iy
    K=np.zeros(side*side)
    for dy in (0,1):
        for dx in (0,1):
            w=(fx if dx else 1-fx)*(fy if dy else 1-fy)
            if w<=0: continue
            xi=min(max(ix+dx,0),side-1); yi=min(max(iy+dy,0),side-1); K[yi*side+xi]+=w
    return K
old=whole640(163.7,175.7); new=S4.bilinear_scatter_window((163.7,175.7,1,1),W)
oldnz=set((i//16,i%16) for i in range(256) if old[i]>1e-15)
newnz=set((i//16,i%16) for i in range(256) if new[i]>1e-15)
res["tests"]["reverse_whole640_detected"]={"old_nz":sorted(oldnz),"new_nz":sorted(newnz),"detected_as_wrong":oldnz!=newnz}

Wt=[608.,608.,640.,640.]
Kt=S4.bilinear_scatter_window((624.,624.,1,1),Wt)
res["tests"]["edge_truncated_window"]={"W":Wt,"sum":round(float(Kt.sum()),9),
  "nz":[(i//16,i%16) for i in range(256) if Kt[i]>1e-15]}

# crop tests using a float image encoded with exact coordinates
Hh,Ww=100,200
base=np.zeros((Hh,Ww,3),dtype=np.float32)
base[...,0]=np.arange(Ww)[None,:]/(Ww-1)
base[...,1]=np.arange(Hh)[:,None]/(Hh-1)
base[...,2]=0.5
pil=Image.fromarray((base*255).astype(np.uint8))
Wc=[64.,32.,128.,96.]
c=S4.crop128_from_window(pil,Wc,(Ww,Hh))
res["tests"]["crop_basic"]={"shape":list(c.shape),"dtype":str(c.dtype),
  "finite":bool(torch.isfinite(c).all()),
  "R_range":[round(float(c[0].min()),4),round(float(c[0].max()),4)],
  "G_range":[round(float(c[1].min()),4),round(float(c[1].max()),4)]}
# no axis swap: R must vary along width, G along height
rvar=float((c[0][:,1:]-c[0][:,:-1]).abs().mean())
gvar=float((c[1][1:,:]-c[1][:-1,:]).abs().mean())
rvar_y=float((c[0][1:,:]-c[0][:-1,:]).abs().mean())
gvar_x=float((c[1][:,1:]-c[1][:,:-1]).abs().mean())
res["tests"]["crop_no_axis_swap"]={"R_var_along_x":round(rvar,6),"R_var_along_y":round(rvar_y,8),
  "G_var_along_y":round(gvar,6),"G_var_along_x":round(gvar_x,8),
  "R_varies_along_x_not_y":rvar>1e-4 and rvar_y<1e-6,
  "G_varies_along_y_not_x":gvar>1e-4 and gvar_x<1e-6}
# hand pixel vs 296 centre index
x0,ww=64.,64.; ow,oh=200,100
ex=(x0+0.5*ww/128)*(ow/640); ey=(32.0+0.5*64/128)*(oh/640)
res["tests"]["crop_hand_pixel"]={"edge_x":round(ex,4),"edge_y":round(ey,4),
  "expected_centre_idx_x":round(ex-0.5,4),"expected_centre_idx_y":round(ey-0.5,4),
  "expected_R":round((ex-0.5)/(Ww-1),4),"got_R":round(float(c[0,0,0]),4),
  "note":"uint8量化与双线性带来约0.002~0.005误差"}
c2=S4.crop128_from_window(pil,Wc,(Ww,Hh))
res["tests"]["crop_deterministic"]={"bit_equal":bool(torch.equal(c,c2))}
json.dump(res,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\340_result_unit_tests.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(res,ensure_ascii=False,indent=1)[:3000])
