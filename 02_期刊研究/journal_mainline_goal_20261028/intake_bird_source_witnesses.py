"""Verify and render author TRAIN pixels/GT only; no models, matching or fitting."""
import hashlib
import io
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

base = Path(__file__).parent
meta = base / 'smot_source_metadata_20261029'
out = meta / 'witness_intake_20261029'
out.mkdir(exist_ok=True)
downloads = Path('C:/Users/jiwei/Downloads')
data = json.loads((meta / 'author_train_annotations.json').read_text(encoding='utf-8'))
by_image = defaultdict(list)
for a in data['annotations']:
    by_image[a['image_id']].append(a)
font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 19)
small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
receipt = {'scope': 'author TRAIN source pixels and source annotations only',
           'model_forward': False, 'training_or_fitting': False, 'prediction_matching': False,
           'test_read': False, 'registration_run': False,
           'license': {'path': str(downloads / 'LICENSE'),
                       'sha256': hashlib.sha256((downloads / 'LICENSE').read_bytes()).hexdigest()},
           'clips': []}
(out / 'LICENSE').write_bytes((downloads / 'LICENSE').read_bytes())
anchor_frames = {'0042': (2, 3), '0044': (3, 6)}

def annotated(im, rows, crop_origin=(0,0), anchor=None):
    im = im.copy()
    d = ImageDraw.Draw(im)
    ox, oy = crop_origin
    for a in rows:
        x,y,w,h = a['bbox']
        d.rectangle((x-ox,y-oy,x+w-ox,y+h-oy), outline='#00e878', width=2)
    if anchor:
        x,y,w,h = anchor
        d.rectangle((x-ox,y-oy,x+w-ox,y+h-oy), outline='#ff5c37', width=2)
    return im

for vid, name in [(42,'0042'),(44,'0044')]:
    p = downloads / (name+'.zip')
    frames = sorted((im for im in data['images'] if im['video_id'] == vid), key=lambda im: im['frame_id'])
    source_out = out / 'source_images' / name
    source_out.mkdir(parents=True, exist_ok=True)
    rec = {'video': name, 'source_zip': str(p), 'zip_bytes': p.stat().st_size,
           'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'frames': []}
    pixels = {}
    with zipfile.ZipFile(p) as z:
        if sum(i.file_size for i in z.infolist()) > 150_000_000:
            raise ValueError('source read cap exceeded')
        rec['bad_crc_entry'] = z.testzip()
        if rec['bad_crc_entry']:
            raise ValueError('source CRC error')
        rec['archive_files'] = z.namelist()
        for frame in frames:
            leaf = Path(frame['file_name']).name
            if leaf not in z.namelist():
                raise ValueError('author source filename missing: '+leaf)
            raw = z.read(leaf)
            im = Image.open(io.BytesIO(raw)).convert('RGB')
            if im.size != (frame['width'],frame['height']):
                raise ValueError('source image dimensions differ from author annotation')
            (source_out / leaf).write_bytes(raw)
            pixels[frame['frame_id']] = im
            rec['frames'].append({'file':frame['file_name'], 'image_id':frame['id'],
                                  'sha256':hashlib.sha256(raw).hexdigest(), 'size':list(im.size),
                                  'author_gt':by_image[frame['id']]})
    pos_id, later_id = anchor_frames[name]
    chosen = sorted(set([1,pos_id,later_id,frames[len(frames)//2]['frame_id'],frames[-1]['frame_id']]))
    panel = Image.new('RGB',(1536,480*math.ceil(len(chosen)/2)), 'white')
    draw = ImageDraw.Draw(panel)
    for n, f in enumerate(chosen):
        frame = next(im for im in frames if im['frame_id']==f)
        annotated_frame = annotated(pixels[f],by_image[frame['id']]).resize((768,432),Image.Resampling.LANCZOS)
        x,y = (n%2)*768,(n//2)*480
        panel.paste(annotated_frame,(x,y+42))
        draw.text((x+8,y+6),f'{name} / {f:05d}   author GT: {len(by_image[frame["id"]])}',font=font,fill='black')
    panel.save(out/(name+'_source_overview.png'))
    positive = next(im for im in frames if im['frame_id']==pos_id)
    anchor = by_image[positive['id']][0]['bbox']
    x,y,w,h = anchor
    left = max(0,min(3840-512,int(x+w/2)-256))
    top = max(0,min(2160-512,int(y+h/2)-256))
    crop_box = (left,top,left+512,top+512)
    fseq = [pos_id,later_id,frames[-1]['frame_id']]
    cpanel = Image.new('RGB',(1536,570),'white')
    cdraw = ImageDraw.Draw(cpanel)
    for n,f in enumerate(fseq):
        frame = next(im for im in frames if im['frame_id']==f)
        tile = annotated(pixels[f].crop(crop_box),by_image[frame['id']],(left,top),anchor)
        cpanel.paste(tile,(n*512,48))
        cdraw.text((n*512+5,4),f'{name}/{f:05d} - same PIXEL crop (not registered)',font=small,fill='black')
        cdraw.text((n*512+5,25),'green: author GT; red: original anchor',font=small,fill='black')
    cpanel.save(out/(name+'_unregistered_source_crops.png'))
    rec['render_selection'] = chosen
    rec['source_crop_xyxy'] = list(crop_box)
    rec['visual_pair_status'] = 'not_yet_reviewed'
    receipt['clips'].append(rec)

path = out / 'source_intake_receipt.json'
path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'output':str(out), 'clips':[{'video':r['video'],'frames':len(r['frames']),'zip_sha256':r['sha256'],'bad_crc_entry':r['bad_crc_entry']} for r in receipt['clips']], 'analysis':'source-only rendering, unregistered crops'},ensure_ascii=False,indent=2))
