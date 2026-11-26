import os, sys, json, random, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import build_341_val as V
from PIL import Image, ImageDraw
torch.set_num_threads(2)
OUT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
imgs, by, order = V.load_val_meta()
SUB = order[:32]
cands=[]
for pid in SUB:
    e=torch.load(V.entry_path(1,pid),map_location='cpu',weights_only=True)
    if e['n_G_W']>0: cands.append(pid)
random.seed(2026102101)
pick=random.sample(cands,min(6,len(cands)))
os.makedirs(os.path.join(OUT,'341_vis'),exist_ok=True)
info=[]
for pid in pick:
    e=torch.load(V.entry_path(1,pid),map_location='cpu',weights_only=True)
    im=imgs[pid]
    full=Image.open(os.path.join(V.VAL_IMG, im['file_name'])).convert('RGB')
    W=[float(v) for v in e['window_xyxy_640']]
    canvas=full.resize((640,640)); dr=ImageDraw.Draw(canvas)
    dr.rectangle(W, outline=(0,255,255), width=3)
    for i in range(min(50,e['fullp_boxes'].shape[0])):
        dr.rectangle([float(v) for v in e['fullp_boxes'][i].tolist()], outline=(0,200,0), width=1)
    g=e['gt_boxes_640'].tolist(); GW=e['G_W'].tolist(); U=e['U_W'].tolist()
    for i in range(len(g)):
        col=(160,160,160)
        if GW[i] and not U[i]: col=(0,80,255)
        elif U[i]: col=(255,0,0)
        dr.rectangle(g[i], outline=col, width=2)
    p=os.path.join(OUT,'341_vis',f'341_val_{pid}.png'); canvas.save(p)
    info.append({'image_id':pid,'png':p,'W':e['window_cell'],'n_GW':e['n_G_W'],'n_UW':e['n_U_W']})
json.dump({'item':'341 val548 目视','seed':2026102101,'images':info},open(os.path.join(OUT,'341_result_vis.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=2)
print(json.dumps(info,indent=1))