"""Display all eight preselected nearest crosssplit retrieval controls."""
import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent;OUT=HERE/'image_duplicates_attempt02'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
rows=read(OUT/'source_rows.json');pick=read(OUT/'pixel_review_selection.json')['closest_controls']
TT=getattr(Image,'Transpose',Image);RS=getattr(Image,'Resampling',Image)
TR=[None,TT.ROTATE_90,TT.ROTATE_180,TT.ROTATE_270,TT.FLIP_LEFT_RIGHT,TT.FLIP_TOP_BOTTOM,TT.TRANSPOSE,TT.TRANSVERSE]
sheet=Image.new('RGB',(960,8*290),'white');dr=ImageDraw.Draw(sheet);sources={}
for j,z in enumerate(pick):
    for col,(key,sp) in enumerate([('val_row','val'),('train_row','train')]):
        r=rows[z[key]];p=Path(r['path']);h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==r['bytes_sha256'];sources[str(p)]=h
        with Image.open(p) as a:im=a.convert('RGB')
        if col and z['transform']:im=im.transpose(TR[z['transform']])
        im.thumbnail((470,250),resample=RS.LANCZOS);x=col*480;yy=j*290;sheet.paste(im,(x,yy+35))
        dr.text((x+5,yy+2),f'C{j+1} {sp} iid={r["image_id"]} transform={z["transform"] if col else 0}',fill='black')
        dr.text((x+5,yy+18),f'DCT={z["DCT_distance"]} dHash={z["dhash_distance"]}',fill='black')
sheet.save(OUT/'closest_crosssplit_controls.png')
(OUT/'controls_display_sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
print('displayed pairs',len(pick))
