"""Display-only repair: draw unclipped data on separate panel canvases."""
import importlib,json
C=importlib.import_module('343_context');M=importlib.import_module('343_formal_counts_v2');G=M.G;F=M.F
import torch,numpy as np
from PIL import Image,ImageDraw

def main():
    ann=C.Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val\annotations\val.json')
    data=json.loads(ann.read_text(encoding='utf-8'));by={}
    for a in data['annotations']:by.setdefault(a['image_id'],[]).append(a)
    paths=[]
    for seed in (1,2):
        rng=np.random.default_rng(2026102207+seed);chosen=(0,17,534 if seed==1 else 210)
        for iid in range(548):
            local=torch.tensor(rng.uniform(.05,.95,(4,4)),dtype=torch.float32);logits=torch.tensor(rng.normal(0,2,(4,10)),dtype=torch.float32)
            if iid not in chosen:continue
            e=torch.load(C.S3/f'342_cache_val_v2_seed{seed}'/f'342_{iid:06d}.pt',map_location='cpu',weights_only=True)
            items=by.get(iid,[]);gt=G.orig_xywh_to_orig_xyxy(torch.tensor([a['bbox'] for a in items],dtype=torch.float64).reshape(-1,4))
            out,_=F.fuse(e,local,logits);owner=M.match(out,gt,e['gt_labels'],e['gt_annotation_ids'],.5)
            assert np.array_equal(owner,M.independent_match(out,gt,e['gt_labels'],e['gt_annotation_ids'],.5))
            src=C.Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val\images')/e['file_name']
            with Image.open(src) as im:orig=im.convert('RGB').resize((512,512),Image.BILINEAR)
            crop=Image.fromarray((e['crop128'].permute(1,2,0).numpy().clip(0,1)*255).round().astype('uint8')).resize((512,512),Image.NEAREST)
            predpanel=crop.copy();targetpanel=crop.copy();pd=ImageDraw.Draw(predpanel);td=ImageDraw.Draw(targetpanel)
            full=torch.tensor([0.,0.,640.,640.]);w=e['window_xyxy_640']
            edge=G.win_norm_to_window_pixels(G.win_norm_xyxy(w[None],full),full,(512,512))[0].numpy()
            ImageDraw.Draw(orig).rectangle(tuple(edge),outline='lime',width=3)
            selected=(out['flat_index']>=3000).nonzero().flatten().tolist()
            for j in selected:
                edge=G.win_norm_to_window_pixels(G.win_norm_xyxy(G.orig_xyxy_to_plane640(out['boxes'][j:j+1],e['orig_wh']),w),w,(512,512))[0].numpy()
                pd.rectangle(tuple(edge),outline='lime' if owner[j]>=0 else 'red',width=2)
                if owner[j]>=0:
                    edge=G.win_norm_to_window_pixels(G.win_norm_xyxy(G.orig_xyxy_to_plane640(gt[owner[j]:owner[j]+1],e['orig_wh']),w),w,(512,512))[0].numpy()
                    td.rectangle(tuple(edge),outline='yellow',width=3)
            canvas=Image.new('RGB',(1560,590),'white');d=ImageDraw.Draw(canvas)
            canvas.paste(orig,(4,60));canvas.paste(predpanel,(524,60));canvas.paste(targetpanel,(1044,60))
            d.text((4,5),f'seed={seed}; image={iid}; RANDOM SYNTHETIC new4 selected={len(selected)}; no head executed; panel bounds affect display only',fill='black')
            d.text((4,27),'Original / frozen W',fill='black');d.text((524,27),'Selected new: green TP / red FP; unclipped numerical data',fill='black');d.text((1044,27),'Matched original GT for selected new',fill='black')
            name=f'343_正式计数合成_v3_seed{seed}_{iid:06d}.png'
            with open(C.HERE/name,'xb') as f:canvas.save(f,format='PNG')
            paths.append(str(C.HERE/name))
    C.write('343_result_阶段4合成PNG显示修正.json',{'status':'已生成；替代v2跨面板溢出的显示','png_paths':paths,
        'numerical_result_unchanged':str(C.HERE/'343_result_阶段4正式计数.json'),
        'method':'只在独立512²面板上绘制，图像库自然截断超面板像素；候选框原始坐标、匹配和指标没有裁剪。',
        '已执行':{'cache_loads':6,'image_decodes':6,'fusion_calls':6,'primary_matches':6,'independent_matches':6,'PNG生成':6,'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
        '未执行':['修改数值/阈值/候选','真实训练','阶段5'],'未核':['主控目视核对'],'source_sha256':C.sources()})

if __name__=='__main__':main()
