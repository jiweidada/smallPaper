import os, sys, json, time, torch
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import build_341_val as V
import stage1_cache_v3 as S3
torch.set_num_threads(2)
OUT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
imgs, by, order = V.load_val_meta()
SUB = order[:32]
model = S3.load_real_model(V.CKPT[1])
rows=[]; t0=time.time()
for pid in SUB:
    e = V.build_entry(model, imgs[pid], by.get(pid, []))
    torch.save(e, V.entry_path(1, pid))
    rows.append({'image_id':pid,'n_gt':e['n_gt'],'n_GW':e['n_G_W'],'n_UW':e['n_U_W'],
                 'W':e['window_cell'],'crop_shape':list(e['crop128'].shape),'orig_wh':e['orig_wh']})
res={'item':'341 val548 32张子集','n':len(rows),'subset_ids':SUB,'rows':rows,'secs':round(time.time()-t0,2)}
print(json.dumps({'n':res['n'],'secs':res['secs'],'GW_pos':sum(1 for r in rows if r['n_GW']>0),'UW_pos':sum(1 for r in rows if r['n_UW']>0)},indent=1))
json.dump(res,open(os.path.join(OUT,'341_result_val_subset.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=2)