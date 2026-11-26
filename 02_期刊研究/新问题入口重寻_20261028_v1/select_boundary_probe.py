"""Freeze outcome-blind finite boundary sample and original-pixel display."""
import csv,json,hashlib,collections,math
from pathlib import Path
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
DATA=Path('C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def key(x):return tuple(int(x[k]) for k in ['image_id','class','short_bin','aspect_bin','neighbour_bin'])
def main():
    p=HERE/'boundary_condition_attempt01/per_GT.csv';rows=list(csv.DictReader(p.open(encoding='utf-8')))
    strata=read(HERE/'boundary_condition_attempt01/strata.json');support={tuple(x['stratum']) for x in strata}
    near=[x for x in rows if x['group']=='near' and int(x['visible_proxy']) and key(x) in support]
    inner=[x for x in rows if x['group']=='inner' and int(x['visible_proxy'])]
    selected=[];used=set()
    for side in range(4):
        ids={int(x['image_id']) for x in near if int(x['near_side'])==side}-used
        ordered=sorted(ids,key=lambda i:hashlib.sha256(f'boundary32:20261028:{side}:{i}'.encode()).hexdigest())[:16]
        for iid in ordered:
            nn=[x for x in near if int(x['image_id'])==iid and int(x['near_side'])==side];keys={key(x) for x in nn}
            ii=[x for x in inner if key(x) in keys]
            local=[z for z in strata if tuple(z['stratum']) in keys]
            for z in local:
                # The original sameimage stratum can have nearGT from other sides.
                # Freeze weights only to selectedside targetGT, as protocol05 specifies.
                z=dict(z);z['near_GT']=sum(key(x)==tuple(z['stratum']) for x in nn)
            selected.append({'image_id':iid,'side':side,'shift':[[32,0],[0,32],[-32,0],[0,-32]][side],
                             'target_GT_ids':sorted(int(x['GTid']) for x in nn),'inner_GT_ids':sorted(int(x['GTid']) for x in ii),
                             'strata':[{'key':list(k),'target_GT_ids':sorted(int(x['GTid']) for x in nn if key(x)==k),
                                        'inner_GT_ids':sorted(int(x['GTid']) for x in ii if key(x)==k)} for k in sorted(keys)]})
            used.add(iid)
    out=HERE/'boundary_shift_selection';out.mkdir(exist_ok=False)
    prot=HERE/'05_边界平移_前向前协议.json'
    sources={str(p):sha(p),str(HERE/'boundary_condition_attempt01/strata.json'):sha(HERE/'boundary_condition_attempt01/strata.json'),str(prot):sha(prot),str(Path(__file__).resolve()):sha(Path(__file__).resolve())}
    ds=read(DATA/'annotations/val.json');ims={x['id']:x for x in ds['images']};ann={x['id']:x for x in ds['annotations']}
    overview={'protocol_sha256':sha(prot),'images':len(selected),'target_GT':sum(len(x['target_GT_ids']) for x in selected),
              'inner_GT':sum(len(x['inner_GT_ids']) for x in selected),'selection_uses_outcomes':False,'selected':selected,'sources':sources}
    with (out/'selection.json').open('x',encoding='utf-8') as f:json.dump(overview,f,ensure_ascii=False,indent=2)
    (HERE/'05_边界平移_前向前协议.sha256').write_text(sha(prot),encoding='utf-8')
    # First two frozen images per side; paired target/control chosen by hash, never score/hit.
    panels=[]
    for side in range(4):
        for z in [x for x in selected if x['side']==side][:2]:
            gg=min(z['target_GT_ids'],key=lambda i:hashlib.sha256(f'pixelcheck:{i}'.encode()).hexdigest())
            st=next(s for s in z['strata'] if gg in s['target_GT_ids'])
            cc=min(st['inner_GT_ids'],key=lambda i:hashlib.sha256(f'pixelcheck:{i}'.encode()).hexdigest())
            for typ,gid in [('near',gg),('inner',cc)]:
                a=ann[gid];im=ims[z['image_id']];x,y,w,h=a['bbox'];cx,cy=x+w/2,y+h/2;win=max(96,min(320,4*max(w,h)))
                left=max(0,math.floor(cx-win/2));top=max(0,math.floor(cy-win/2));right=min(im['width'],math.ceil(cx+win/2));bottom=min(im['height'],math.ceil(cy+win/2))
                pimg=DATA/'images'/im['file_name'];sources[str(pimg)]=sha(pimg)
                with Image.open(pimg) as im0:crop=im0.convert('RGB').crop((left,top,right,bottom))
                d=ImageDraw.Draw(crop);d.rectangle((x-left,y-top,x+w-left,y+h-top),outline=(255,80,50),width=1)
                cell=Image.new('RGB',(380,330),'white');scale=min(370/crop.width,280/crop.height)
                crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),resample=Image.Resampling.NEAREST)
                cell.paste(crop,(5,42));dd=ImageDraw.Draw(cell)
                dd.text((5,5),f'{typ} iid={z["image_id"]} GT={gid} cls={a["category_id"]} side={side}',fill='black')
                dd.text((5,23),f'orig bbox {w:.1f}x{h:.1f}, flags0/0; source pixels only',fill='black')
                panels.append(cell)
    sheet=Image.new('RGB',(380*4,330*4),(230,230,230))
    for i,panel in enumerate(panels):sheet.paste(panel,((i%4)*380,(i//4)*330))
    sheet.save(out/'original_pixel_pairs.png')
    with (out/'pixel_sources.json').open('x',encoding='utf-8') as f:json.dump(sources,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:overview[k] for k in ['images','target_GT','inner_GT','protocol_sha256','selection_uses_outcomes']},ensure_ascii=False))
if __name__=='__main__':main()
