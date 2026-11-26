"""Scientific review panels from immutable original pixels, no enhancement."""
from pathlib import Path
import json, math, hashlib
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
VAL=Path('C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val')
d=json.loads((HERE/'03_漏标审计_样本与总体.json').read_text(encoding='utf-8'))
out=HERE/'原像素审查图_v2';out.mkdir(exist_ok=False)
resampling=getattr(Image,'Resampling',Image)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',23)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
manifest=[]
for start in range(0,len(d['cases']),8):
    cases=d['cases'][start:start+8]
    canvas=Image.new('RGB',(1800,1680),'white');draw=ImageDraw.Draw(canvas)
    for loc,v in enumerate(cases):
        cc=loc%2;rr=loc//2;bx=cc*900;by=rr*420
        p=VAL/'images'/v['file_name'];im=Image.open(p).convert('RGB')
        assert list(im.size)==v['image_wh']
        x,y,w,h=v['bbox'];cx=x+w/2;cy=y+h/2
        ext=max(96,w*2.8,h*2.8)
        l=max(0,int(cx-ext/2));t=max(0,int(cy-ext/2));r=min(im.width,int(cx+ext/2)+1);b=min(im.height,int(cy+ext/2)+1)
        crop=im.crop((l,t,r,b));patch=crop.copy()
        pd=ImageDraw.Draw(patch);pd.rectangle((x-l,y-t,x+w-l,y+h-t),outline=(255,40,40),width=1)
        thumb=im.copy();td=ImageDraw.Draw(thumb);td.rectangle((x,y,x+w,y+h),outline=(255,20,20),width=max(2,im.width//450))
        thumb.thumbnail((435,350),resampling.LANCZOS)
        # Enlarged raw nearest-neighbor pixels are for visibility, not superresolution.
        scale=min(435/crop.width,350/crop.height)
        patch=patch.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))),resampling.NEAREST)
        draw.text((bx+10,by+6),v['case_id'],font=font,fill='black')
        draw.text((bx+100,by+10),f"image {v['image_id']}  box {w:.1f}x{h:.1f} source px",font=small,fill='black')
        canvas.paste(thumb,(bx+5,by+48));canvas.paste(patch,(bx+455,by+48))
        draw.text((bx+5,by+397),'full original image',font=small,fill='black')
        draw.text((bx+455,by+397),'raw pixels, nearest display',font=small,fill='black')
        v['review_crop_bounds']=[l,t,r,b]
    target=out/f'panel_{start//8+1:02d}.png';canvas.save(target)
    manifest.append({'path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'case_ids':[v['case_id'] for v in cases]})
(HERE/'04_原像素图册清单.json').write_text(json.dumps({'panels':manifest,'cases_with_crop_bounds':d['cases'],'display_note':'Only original source pixels; overview downsampled, crop enlarged by nearest. IDs do not expose seed/score/class. Not an independent blind review.'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'panels':len(manifest),'cases':len(d['cases']),'directory':str(out)},ensure_ascii=False))
