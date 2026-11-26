"""New-host annotation resource census; old Dome restrictions are not modified."""
from pathlib import Path
from fractions import Fraction as F
from PIL import Image
import hashlib,importlib.util,json,math,os,sys,time
sys.stdout.reconfigure(encoding='utf-8')
O=Path(__file__).resolve().parent;R=O.parent
DATA=Path(r'D:\数据集\visdrone2019\VisDrone2019-DET-val')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wr(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    rules=O/'new_host_annotation_resource_rules.json'
    cfg=json.loads(rules.read_text(encoding='utf-8'))
    assert cfg['minimum_annotation_only_prefix_resource']==20 and cfg['new_model_forwards']==0
    lock=O/'new_host_annotation_started.json'
    with lock.open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'rules_sha256':sha(rules)},f)
    old=R/'a18_neighbour_annotation_preflight.py'
    spec=importlib.util.spec_from_file_location('unchanged_a18_geometry_helpers',old)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    images=sorted((DATA/'images').glob('*.jpg'));assert len(images)==548
    selected_by_prefix={};per_image=[];pairs=[];inputs=[];tiny_count=0;oriented_count=0
    for im in images:
        ann=DATA/'annotations'/(im.stem+'.txt')
        with Image.open(im) as f:W,H=f.size
        raw=[]
        for uid,line in enumerate(ann.read_text(encoding='utf-8-sig').splitlines()):
            if not line.strip():continue
            v=[int(x.strip()) for x in line.split(',') if x.strip()];assert len(v)>=8
            x,y,w,h,score,cat,trunc,occ=v[:8]
            x1,x2=max(0,min(W,x)),max(0,min(W,x+w));y1,y2=max(0,min(H,y)),max(0,min(H,y+h))
            if x2<=x1 or y2<=y1:continue
            b=[F(x1*800,W),F(y1*800,H),F(x2*800,W),F(y2*800,H)]
            raw.append({'uid':uid,'box':b,'area':(b[2]-b[0])*(b[3]-b[1]),'cat':cat,
                'valid':score>0 and 1<=cat<=10,'cx':(b[0]+b[2])/2,'cy':(b[1]+b[3])/2})
        tiny=[t for t in raw if t['valid'] and t['area']<64];tiny_count+=len(tiny);eligible=[]
        for a in tiny:
            guard=m.outward(a['box'])
            for b in tiny:
                if a['uid']==b['uid'] or a['cat']!=b['cat'] or max(a['area'],b['area'])>4*min(a['area'],b['area']):continue
                d2=(a['cx']-b['cx'])**2+(a['cy']-b['cy'])**2
                if d2*d2>16*a['area']*b['area']:continue
                nb=m.outward(b['box'])
                if m.intersects(guard,nb) or any(t['uid']!=b['uid'] and m.intersects(nb,t['box']) for t in raw):continue
                bw,bh=nb[2]-nb[0],nb[3]-nb[1]
                bx=m.rounded_half_up(2*a['cx']-nb[2]);by=m.rounded_half_up(2*a['cy']-nb[3]);bg=[bx,by,bx+bw,by+bh]
                if not(0<=bg[0]<bg[2]<=800 and 0<=bg[1]<bg[3]<=800):continue
                if any(m.intersects(bg,t['box']) for t in raw):continue
                assert not m.intersects(guard,bg)
                eligible.append({'image':im.name,'prefix':im.stem.split('_')[0],'target_uid':a['uid'],'neighbour_uid':b['uid'],
                  'category':a['cat'],'normalized_distance':math.sqrt(float(d2))/float(a['area']*b['area'])**0.25,
                  'target_box800_exact':[str(v) for v in a['box']],'neighbour_box800_exact':[str(v) for v in b['box']],
                  'target_guard':guard,'neighbour_mask':nb,'background_mask':bg})
        eligible.sort(key=lambda z:(z['normalized_distance'],z['target_uid'],z['neighbour_uid']))
        oriented_count+=len(eligible)
        if eligible:
            pairs.append(eligible[0]);selected_by_prefix.setdefault(eligible[0]['prefix'],eligible[0])
        per_image.append({'image':im.name,'tiny_GT':len(tiny),'oriented_pairs':len(eligible)})
        inputs.append({'image':im.name,'width':W,'height':H,'annotation_sha256':sha(ann),'image_bytes':im.stat().st_size})
    counts={'images':len(images),'valid_tiny_GT':tiny_count,'candidate_oriented_pairs':oriented_count,
            'eligible_images':len(pairs),'eligible_prefixes':len(selected_by_prefix)}
    wr('new_host_annotation_inputs.json',inputs);wr('new_host_per_image_resource.json',per_image)
    wr('new_host_eligible_image_pairs.json',pairs);wr('new_host_prefix_candidates.json',list(selected_by_prefix.values()))
    result={'status':'ANNOTATION_RESOURCE_CENSUS_COMPLETE_NOT_EXPERIMENT','counts':counts,
       'annotation_prefix_minimum_met':len(selected_by_prefix)>=20,'rules_sha256':sha(rules),'script_sha256':sha(Path(__file__)),
       'unchanged_geometry_helper_source_sha256':sha(old),'new_model_forwards':0,'model_predictions_read':False,
       'old_Dome_exclusion_or_gate_changed':False,'new_host_training_or_operator_protocol_frozen':False,
       'interpretation':'No effect, causal evidence, novelty, native-state-grid eligibility or power claim'}
    wr('new_host_annotation_resource_result.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
