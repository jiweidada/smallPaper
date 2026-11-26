"""Select capacity-only train images using annotation counts, without predictions."""
from pathlib import Path
from PIL import Image
import hashlib,json,shutil,sys,zipfile
sys.stdout.reconfigure(encoding='utf-8')
O=Path(__file__).resolve().parent
DATA=Path(r'D:\数据集\visdrone2019\VisDrone2019-DET-train')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wr(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rows(p):
    out=[]
    for i,s in enumerate(p.read_text(encoding='utf-8-sig').splitlines()):
        if not s.strip():continue
        v=[int(x.strip()) for x in s.split(',') if x.strip()]
        assert len(v)==8,(p.name,i,v)
        x,y,w,h,score,cat,_,_=v
        if score>0 and 1<=cat<=10 and w>0 and h>0:out.append((i+1,x,y,w,h,cat))
    return out
def main():
    assert not (O/'capacity_inputs.zip').exists(),'Existing input package: inspect, do not regenerate'
    files=sorted((DATA/'annotations').glob('*.txt'))
    assert len(files)==6471
    aggregate=hashlib.sha256();ranking=[]
    for p in files:
        aggregate.update(p.name.encode()+b'\0'+bytes.fromhex(sha(p)))
        ranking.append((-len(rows(p)),p.stem,p))
    selected=sorted(ranking)[:2]
    targetdir=O/'capacity_inputs';targetdir.mkdir(exist_ok=False)
    cases=[]
    for _,stem,p in selected:
        image=DATA/'images'/(stem+'.jpg')
        with Image.open(image) as im:W,H=im.size;assert im.mode=='RGB'
        labels=[];boxes=[];area=[];line_numbers=[]
        for line,x,y,w,h,cat in rows(p):
            x1,y1,x2,y2=max(0,x),max(0,y),min(W,x+w),min(H,y+h)
            assert x2>x1 and y2>y1,(stem,line,'zero clipped area')
            labels.append(cat)
            boxes.append([(x1+x2)/(2*W),(y1+y2)/(2*H),(x2-x1)/W,(y2-y1)/H])
            area.append((x2-x1)*800/W*(y2-y1)*800/H)
            line_numbers.append(line)
        for src in [image,p]:shutil.copy2(src,targetdir/src.name)
        cases.append({'stem':stem,'image_file':image.name,'annotation_file':p.name,
          'image_sha256':sha(image),'annotation_sha256':sha(p),'original_width_height':[W,H],
          'valid_gt_count':len(labels),'source_line_numbers':line_numbers,
          'labels':labels,'boxes_cxcywh_normalized':boxes,'resized_box_area':area,
          'orig_size_hw':[H,W],'size_hw':[800,800]})
    manifest={'purpose':'capacity-only selected training inputs; no validation outcomes or accuracy evaluation',
      'dataset':'VisDrone2019-DET-train','annotations_scanned':len(files),
      'annotation_scan_sha256':aggregate.hexdigest(),
      'selection':'descending valid GT count, ascending filename ties; first2; scores>0, class1..10, positive w/h',
      'geometry':'clipped original xyxy then normalized cxcywh; nonuniform torchvision v2 Resize800x800; classes retain ids1..10 in12 outputs',
      'cases':cases,'model_predictions_read':False}
    wr(targetdir/'input_manifest.json',manifest)
    with zipfile.ZipFile(O/'capacity_inputs.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(targetdir.iterdir()):z.write(p,'capacity_inputs/'+p.name)
    wr(O/'capacity_input_package_receipt.json',{'sha256':sha(O/'capacity_inputs.zip'),
       'bytes':(O/'capacity_inputs.zip').stat().st_size,'selected':[{k:c[k] for k in ['stem','valid_gt_count','original_width_height']} for c in cases]})
    print((O/'capacity_input_package_receipt.json').read_text(encoding='utf-8'))
if __name__=='__main__':main()
