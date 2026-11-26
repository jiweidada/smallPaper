import os, sys, json, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import build_341_val as V
torch.set_num_threads(2)
OUT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
imgs, by, order = V.load_val_meta()
SUB = order[:32]
def norm2xyxy(b):
    b=b.float()*640.0; cx,cy,w,h=b.unbind(-1)
    return torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],-1)
def indep_window(b640):
    cx=((b640[:,0]+b640[:,2])/2).numpy(); cy=((b640[:,1]+b640[:,3])/2).numpy()
    best=None; bn=-1
    for y in range(0,640,64):
        for x in range(0,640,64):
            n=int(((cx>=x)&(cx<x+64)&(cy>=y)&(cy<y+64)).sum())
            if n>bn: bn=n; best=(y//64,x//64)
    return best
def complete(gt,W):
    return (gt[:,0]>=W[0]-1e-6)&(gt[:,1]>=W[1]-1e-6)&(gt[:,2]<=W[2]+1e-6)&(gt[:,3]<=W[3]+1e-6)
rows=[]
for pid in SUB:
    e=torch.load(V.entry_path(1,pid),map_location='cpu',weights_only=True)
    rc=indep_window(norm2xyxy(e['final300_boxes']))
    W=[rc[1]*64,rc[0]*64,rc[1]*64+64,rc[0]*64+64]
    GW=complete(e['gt_boxes_640'],W) if e['gt_boxes_640'].numel() else torch.zeros(0,dtype=torch.bool)
    UW=GW&(~e['m0_gt_explained']) if GW.numel() else torch.zeros(0,dtype=torch.bool)
    kg=True
    if GW.any():
        s=e['K_g'][GW].sum(1); kg=bool(torch.allclose(s,torch.ones_like(s),atol=1e-5))
    rows.append({'image_id':pid,'W_match':list(e['window_cell'])==list(rc),
                 'GW_match':int(GW.sum())==e['n_G_W'],'UW_match':int(UW.sum())==e['n_U_W'],'kg_ok':kg})
res={'item':'341 val548子集独立复核','checked':len(rows),
     'all_W_match':all(r['W_match'] for r in rows),'all_GW_match':all(r['GW_match'] for r in rows),
     'all_UW_match':all(r['UW_match'] for r in rows),'all_kg_ok':all(r['kg_ok'] for r in rows),
     'mismatches':[r for r in rows if not (r['W_match'] and r['GW_match'] and r['UW_match'] and r['kg_ok'])]}
print(json.dumps(res,indent=1))
json.dump(res,open(os.path.join(OUT,'341_result_val_indep.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=2)