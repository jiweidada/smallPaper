"""Bind existing research design and all validation assets; do not run a model."""
from pathlib import Path
from PIL import Image
import ast,hashlib,json,shutil,sys,zipfile
sys.stdout.reconfigure(encoding='utf-8')
O=Path(__file__).resolve().parent;R=O.parent;A21=R/'A21_共享状态路径隔离与基线协议_v1'
DATA=Path(r'D:\数据集\visdrone2019\VisDrone2019-DET-val')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wr(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    assert not (O/'validation_assets.zip').exists(),'Existing package: inspect, do not overwrite'
    last=json.loads((R/'53_A21共享状态路径与研究设计核验回执.json').read_text(encoding='utf-8'))
    for row in last['files']:assert sha(R/row['path'])==row['sha256'],row['path']
    for name in ['scientific_design_PRESPECIFIED.json','scientific_design_seal.json','train_asset_manifest.json',
                 'scientific_cohort_asset_manifest.json','cohort_pixel_control_rules_FROZEN.json']:
        assert not (O/name).exists();shutil.copy2(A21/name,O/name)
    shutil.copy2(A21/'verified_remote_data_controls/cohort_pixel_control_preparation.json',O/'cohort_pixel_control_preparation.json')
    images=[];annotations=[];rows=[];ignored=0
    files=sorted((DATA/'images').glob('*.jpg'));assert len(files)==548
    for index,image in enumerate(files,1):
        annotation=DATA/'annotations'/(image.stem+'.txt')
        with Image.open(image) as im:W,H=im.size
        n=0
        for line in annotation.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():continue
            v=[int(x.strip()) for x in line.split(',') if x.strip()];assert len(v)==8
            x,y,w,h,score,cat,_,_=v
            l,r=max(0,min(W,x)),max(0,min(W,x+w));t,b=max(0,min(H,y)),max(0,min(H,y+h))
            if not(score>0 and 1<=cat<=10 and r>l and b>t):ignored+=1;continue
            n+=1;annotations.append({'id':len(annotations)+1,'image_id':index,'category_id':cat,
              'bbox':[l,t,r-l,b-t],'area':(r-l)*(b-t),'iscrowd':0})
        images.append({'id':index,'file_name':image.name,'width':W,'height':H})
        rows.append({'id':index,'image':image.name,'annotation':annotation.name,'width':W,'height':H,'valid_GT':n,
           'image_sha256':sha(image),'annotation_sha256':sha(annotation),'image_bytes':image.stat().st_size})
    wr('validation_asset_manifest.json',{'dataset':'VisDrone2019-DET-val','image_count':548,'valid_GT':len(annotations),
       'omitted_raw_annotations':ignored,'GT_scope':'valid score>0, class1..10, clipped positive rectangles; official ignore-region policy not reproduced',
       'files':rows,'predictions_inspected':False})
    wr('validation_valid_GT_COCO.json',{'info':{'description':'Prespecified valid-GT diagnostic COCO-style evaluation, not official ignore-region benchmark'},
       'images':images,'annotations':annotations,'categories':[{'id':i,'name':'class_'+str(i)} for i in range(1,11)]})
    with zipfile.ZipFile(O/'validation_assets.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for row in rows:
            for p in [DATA/'images'/row['image'],DATA/'annotations'/row['annotation']]:z.write(p,'val_inputs/'+p.name)
    receipt={'sha256':sha(O/'validation_assets.zip'),'bytes':(O/'validation_assets.zip').stat().st_size,
       'members':1096,'images':548,'valid_GT':len(annotations),'predictions_inspected':False}
    wr('validation_assets_package_receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
