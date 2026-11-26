"""Readable close-up views of the already verified eligibility groups."""
import importlib,json
C=importlib.import_module('343_context')
import torch,numpy as np
from PIL import Image,ImageDraw
F=importlib.import_module('343_fusion');G=F.G

def main():
    receipt=json.loads((C.HERE/'343_result_阶段4Geo0资格.json').read_text(encoding='utf-8'));paths=[]
    for seed in (1,2):
        for iid in receipt['PNG_ids'][str(seed)]:
            e=torch.load(C.S3/f'342_cache_val_v2_seed{seed}'/f'342_{iid:06d}.pt',map_location='cpu',weights_only=True)
            src=C.Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val\images')/e['file_name']
            with Image.open(src) as im:pil=im.convert('RGB')
            a=Image.new('RGB',(1560,580),'white');d=ImageDraw.Draw(a)
            a.paste(pil.resize((512,512),Image.BILINEAR),(4,48))
            crop=Image.fromarray((e['crop128'].permute(1,2,0).numpy().clip(0,1)*255).round().astype('uint8')).resize((512,512),Image.NEAREST)
            a.paste(crop,(524,48));a.paste(crop,(1044,48))
            full=torch.tensor([0.,0.,640.,640.]);w=e['window_xyxy_640']
            edge=G.win_norm_to_window_pixels(G.win_norm_xyxy(w[None],full),full,(512,512))[0].numpy()
            d.rectangle(tuple(edge+[4,48,4,48]),outline='#00ff00',width=3)
            gt=G.win_norm_to_window_pixels(G.win_norm_xyxy(e['gt_boxes_640'][e['G_W']],w),w,(512,512)).numpy()
            for b in gt:d.rectangle(tuple(b+[524,48,524,48]),outline='yellow',width=2)
            tiny=F.tiny_geo0_group(e)
            boxes=G.win_norm_to_window_pixels(G.win_norm_xyxy(e['gt_boxes_640'][tiny],w),w,(512,512)).numpy()
            for b in boxes:d.rectangle(tuple(b+[1044,48,1044,48]),outline='red',width=2)
            d.text((4,5),f'seed={seed}; image={iid}; complete GT={int(e["G_W"].sum())}; tiny Geo0 eligible GT={int(tiny.sum())}; NOT formal added TP',fill='black')
            d.text((4,27),'Original / frozen W',fill='black');d.text((524,27),'Frozen crop / complete GT (yellow)',fill='black');d.text((1044,27),'Frozen crop / tiny Geo0 eligible GT (red)',fill='black')
            name=f'343_Geo0资格近景_seed{seed}_{iid:06d}.png'
            with open(C.HERE/name,'xb') as f:a.save(f,format='PNG')
            paths.append(str(C.HERE/name))
    C.write('343_result_阶段4Geo0近景PNG.json',{'status':'已生成，已核资格组的显示补充','png_paths':paths,
        'qualification_receipt':str(C.HERE/'343_result_阶段4Geo0资格.json'),
        '已执行':{'cache_loads':6,'image_decodes':6,'PNG生成':6,'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
        '未执行':['正式B计数','真实训练','阶段5'],'未核':['主控目视核对'],'source_sha256':C.sources()})

if __name__=='__main__':main()
