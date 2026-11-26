"""Finite annotation-only population audit; no detector, fitting or evaluation run."""
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict,Counter
import hashlib,json,math,time
from PIL import Image

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'A17_新编码器机制与多实例映射审计_v1'
SOURCE=Path('D:/数据集/visdrone2019/VisDrone2019-DET-train')
OUT.mkdir(exist_ok=True)
assert not (OUT/'annotation_started.json').exists()
def write(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
annotations=sorted((SOURCE/'annotations').glob('*.txt'))
assert len(annotations)==6471
protocol={'utc':datetime.now(timezone.utc).isoformat(),'status':'FROZEN_BEFORE_POPULATION_ANNOTATION_COUNTS',
    'question':'How common are multiple valid GT centres occupying one stride4 cell in 800-square UAV training images?',
    'dataset':'All6471officialVisDrone2019DETtrainimages;noofficialval/testselectionornewpredictions.',
    'validity':'score>0,class1..10,cliptoimagebounds,positivewh;noocclusion/truncationfilter.',
    'tiny':'sqrtGTarea<8after800-squareindependentx/yresize,consistentwithexistingDomeaudit.',
    'primary':'FractionoftinyGTwhosecentregridcellatstride4phase0containsanotherGTcentre. Same-classandany-classreportedseparately.',
    'descriptive_only':'Strides8/16andquartercellphaseoffsets0,.25,.5,.75;neighbourswithin4pixels. No post-result choice of a better scale or threshold.',
    'screen':'Primarysame-classfraction>=.05onlyprioritizesfurthermodeldiagnosis;lowerfractiondeprioritizesfixed-cellmultiplicityasbroadmainline. Not significance or proof of representational capacity.',
    'limits':['One feature vector can encode multiple objects;multiple scales and neighbouring query seeds can resolve cell coincidences.',
        'Annotation grid coincidence is not a hard DETR recall ceiling or a causal model failure.',
        'Arbitrary grid phase and duplicate/occluded annotations require explicit accounting.'],
    'new_complete_model_forwards':0,'new_training_updates':0}
write('annotation_protocol.json',protocol)
write('annotation_started.json',{'utc':datetime.now(timezone.utc).isoformat(),'annotation_files':6471})
start=time.monotonic();images=[];inputs=[];total=Counter();totals=defaultdict(Counter)
for index,path in enumerate(annotations):
    image=SOURCE/'images'/(path.stem+'.jpg')
    with Image.open(image) as im:w,h=im.size
    raw=path.read_bytes();rows=[];duplicates=Counter()
    for lineid,line in enumerate(raw.decode('utf-8-sig').splitlines()):
        x=[float(v.strip()) for v in line.split(',') if v.strip()]
        if len(x)<6 or x[4]<=0 or not 1<=int(x[5])<=10:continue
        l,t,bw,bh=x[:4];x1=max(0.,min(w,l));y1=max(0.,min(h,t));x2=max(0.,min(w,l+bw));y2=max(0.,min(h,t+bh))
        if x2<=x1 or y2<=y1:continue
        box=[800*x1/w,800*y1/h,800*x2/w,800*y2/h];size=math.sqrt((box[2]-box[0])*(box[3]-box[1]))
        r={'uid':lineid,'label':int(x[5]),'cx':(box[0]+box[2])*.5,'cy':(box[1]+box[3])*.5,'tiny':size<8}
        rows.append(r);duplicates[(r['label'],x1,y1,x2,y2)]+=1
    tiny=[i for i,r in enumerate(rows) if r['tiny']]
    count=Counter(valid_GT=len(rows),tiny_GT=len(tiny),duplicate_extra_GT=sum(v-1 for v in duplicates.values()))
    grids={}
    for stride in [4,8,16]:
        for phase in [0.,.25,.5,.75]:
            groups=defaultdict(list)
            for i,r in enumerate(rows):groups[(math.floor(r['cx']/stride+phase),math.floor(r['cy']/stride+phase))].append(i)
            anycount=samecount=0
            for i in tiny:
                r=rows[i];members=groups[(math.floor(r['cx']/stride+phase),math.floor(r['cy']/stride+phase))]
                anycount+=len(members)>1
                samecount+=any(j!=i and rows[j]['label']==r['label'] for j in members)
            key='s%d_phase%.2f'%(stride,phase)
            grids[key]={'any_class':int(anycount),'same_class':int(samecount)}
            totals[key].update(grids[key])
    bucket=defaultdict(list)
    for i,r in enumerate(rows):bucket[math.floor(r['cx']/4),math.floor(r['cy']/4)].append(i)
    near_any=near_same=0
    for i in tiny:
        r=rows[i];xx=math.floor(r['cx']/4);yy=math.floor(r['cy']/4);near=[]
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                for j in bucket.get((xx+dx,yy+dy),[]):
                    if j!=i and (rows[j]['cx']-r['cx'])**2+(rows[j]['cy']-r['cy'])**2<=16:near.append(j)
        near_any+=bool(near);near_same+=any(rows[j]['label']==r['label'] for j in near)
    count.update(neighbour_within4_any_class=near_any,neighbour_within4_same_class=near_same)
    total.update(count)
    images.append({'file':image.name,'prefix':path.stem.split('_')[0],'width':w,'height':h,**dict(count),'grid_counts':grids})
    inputs.append({'annotation':path.name,'annotation_sha256':hashlib.sha256(raw).hexdigest(),'image_width':w,'image_height':h})
    if (index+1)%1000==0:write('annotation_progress.json',{'images_complete':index+1,'cap':6471,'new_model_forwards':0})
ratios={k:{**dict(v),'tiny_GT':total['tiny_GT'],'any_fraction':v['any_class']/total['tiny_GT'],
    'same_fraction':v['same_class']/total['tiny_GT']} for k,v in totals.items()}
primary=ratios['s4_phase0.00'];passed=primary['same_fraction']>=.05
summary={'utc':datetime.now(timezone.utc).isoformat(),'status':'COMPLETED_ANNOTATION_ONLY_MULTIPLICITY_POPULATION_AUDIT',
    'images':len(images),'counts':dict(total),'grid_counts':ratios,'primary':primary,
    'primary_image_equal_mean_same_fraction':sum(i['grid_counts']['s4_phase0.00']['same_class']/i['tiny_GT'] for i in images if i['tiny_GT'])/sum(i['tiny_GT']>0 for i in images),
    'resource_screen_passed':bool(passed),
    'decision':'WORTH_SOURCE_SPECIFIC_MODEL_DIAGNOSIS_ONLY_NOT_MAINLINE' if passed else 'FIXED_CELL_MULTIPLICITY_LOW_POPULATION_PREVALENCE_DEPRIORITIZE_AS_BROAD_MAINLINE',
    'new_complete_model_forwards':0,'main_model_updates':0,'mainline_validated':False,'wall_seconds':time.monotonic()-start,
    'interpretation':'Population annotation geometry only;not detection AP,causal error attribution,capacity proof or an approved multiplicity module.'}
write('annotation_per_image.json',images);write('annotation_input_manifest.json',inputs);write('annotation_summary.json',summary)
print(json.dumps({'status':summary['status'],'images':len(images),'counts':dict(total),'primary':primary,
    'decision':summary['decision'],'wall_seconds':summary['wall_seconds']}))
