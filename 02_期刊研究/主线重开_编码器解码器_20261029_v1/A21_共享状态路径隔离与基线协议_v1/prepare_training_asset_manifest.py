"""Full local asset hashes and split overlap check; no model or pixel preprocessing."""
from pathlib import Path
from PIL import Image
import hashlib,json,sys
sys.stdout.reconfigure(encoding='utf-8')
O=Path(__file__).resolve().parent;R=O.parent
DATA=Path(r'D:\数据集\visdrone2019')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wr(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    assert not (O/'train_asset_manifest.json').exists(),'Inspect existing manifest; do not regenerate'
    train=DATA/'VisDrone2019-DET-train';val=DATA/'VisDrone2019-DET-val'
    images=sorted((train/'images').glob('*.jpg'));assert len(images)==6471
    rows=[];aggregate=hashlib.sha256();train_sha=set();prefixes=set();count=0;totalbytes=0
    for image in images:
        ann=train/'annotations'/(image.stem+'.txt');assert ann.is_file()
        with Image.open(image) as im:W,H=im.size
        n=0
        for line in ann.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():continue
            v=[int(x.strip()) for x in line.split(',') if x.strip()];assert len(v)==8
            x,y,w,h,score,cat,_,_=v
            if score>0 and 1<=cat<=10 and min(W,x+w)>max(0,x) and min(H,y+h)>max(0,y):n+=1
        ih,ah=sha(image),sha(ann);train_sha.add(ih);prefixes.add(image.stem.split('_')[0])
        aggregate.update(image.name.encode()+b'\0'+bytes.fromhex(ih)+bytes.fromhex(ah))
        rows.append({'image':image.name,'image_sha256':ih,'image_bytes':image.stat().st_size,
          'annotation':ann.name,'annotation_sha256':ah,'width':W,'height':H,'valid_GT':n})
        count+=n;totalbytes+=image.stat().st_size
    candidates=json.loads((R/'A20_VSSD宿主严格装载与容量预检_v1/new_host_prefix_candidates.json').read_text(encoding='utf-8'))
    cases=[]
    for c in candidates:
        image=val/'images'/c['image'];ann=val/'annotations'/(image.stem+'.txt')
        ih=sha(image);assert ih not in train_sha,('Exact train/val image duplicate',image.name)
        cases.append({**c,'image_sha256':ih,'image_bytes':image.stat().st_size,'annotation_sha256':sha(ann),
          'filename_prefix_also_in_train':c['prefix'] in prefixes})
    wr(O/'train_asset_manifest.json',{'dataset':'official VisDrone2019-DET-train','image_count':len(images),
       'valid_GT_count':count,'image_bytes':totalbytes,'aggregate_sha256':aggregate.hexdigest(),
       'valid_GT_rule':'score>0, classes1..10, positive clipped original rectangle',
       'training_recipe_or_forward_allocation':None,'files':rows})
    wr(O/'scientific_cohort_asset_manifest.json',{'dataset':'official VisDrone2019-DET-val','cases':cases,
       'prediction_outcomes_read':False,'fixed_A20_selection_preserved':True,'train_exact_image_hash_overlap':0,
       'train_filename_prefix_overlap_count':sum(c['filename_prefix_also_in_train'] for c in cases),
       'prefix_scope':'Filename prefix is a grouping unit, not a proven independent scene or statistical power guarantee'})
    print(json.dumps({'train_images':len(images),'valid_GT':count,'image_bytes':totalbytes,'aggregate_sha256':aggregate.hexdigest(),
       'cohort_cases':len(cases),'exact_image_overlap':0,'prefix_overlap':sum(c['filename_prefix_also_in_train'] for c in cases)},ensure_ascii=False))
if __name__=='__main__':main()
