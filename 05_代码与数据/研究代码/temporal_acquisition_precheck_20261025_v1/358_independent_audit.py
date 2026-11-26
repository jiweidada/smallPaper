"""Independent scalar primary-event audit; never imports the precheck implementation."""
import json, hashlib, math, time
from pathlib import Path
from collections import defaultdict
import numpy as np
H=Path(__file__).resolve().parent
p=json.loads((H/'358_protocol.json').read_text(encoding='utf-8'))
r=json.loads((H/'358_result.json').read_text(encoding='utf-8'))
a=json.loads(Path(p['annotation_path']).read_text(encoding='utf-8'))
ims={v['id']:v for v in a['images']}; groups=defaultdict(list); first={}
for im in ims.values():first[im['uav_sequence']]=min(first.get(im['uav_sequence'],im['uav_frame']),im['uav_frame'])
for i,g in enumerate(a['annotations']):
    im=ims[g['image_id']];groups[(im['uav_sequence'],g['uav_track_id'])].append((im['uav_frame'],i,g))
selected={}
for key,rows in groups.items():
    rows.sort()
    if rows[0][0]<=first[key[0]] or len(rows)<32:continue
    if [z[0] for z in rows[:32]]!=list(range(rows[0][0],rows[0][0]+32)):continue
    if any(math.sqrt(g['bbox'][2]*g['bbox'][3]*640/ims[g['image_id']]['width']*640/ims[g['image_id']]['height'])>=32 for _,_,g in rows[:9]):continue
    selected[key]=rows[:32]
assert len(selected)==104
checks={};start=time.perf_counter()
for key in p['primary_models']:
    z=np.load(Path(p['cache_root'])/key/'predictions.npz',allow_pickle=False)['detections']
    z=z[np.argsort(z[:,0],kind='stable')]
    blocks=np.split(z,np.flatnonzero(np.diff(z[:,0]))+1)
    dt={int(b[0,0]):b[:,1:].tolist() for b in blocks}
    features=np.load(H/('358_geometry_'+key+'.npz'))['max_iou']
    e=[];maxerr=0.;n=0
    for tk,rows in sorted(selected.items()):
        values=[]
        for _,index,g in rows:
            im=ims[g['image_id']];gx,gy,gw,gh=g['bbox'];best=0.
            for x,y,w,h,score in dt[g['image_id']]:
                if im['uav_apply_native_ignore'] and any(x>rx and y>ry and x+w<rx+rw and y+h<ry+rh for rx,ry,rw,rh in im['uav_ignore_boxes']):continue
                iw=max(0.,min(x+w,gx+gw)-max(x,gx));ih=max(0.,min(y+h,gy+gh)-max(y,gy))
                inter=iw*ih;union=w*h+gw*gh-inter
                ov=inter/union if union>0 else 0.
                if ov>best:best=ov
            maxerr=max(maxerr,abs(best-float(features[index])));values.append(best);n+=1
        blind=not any(v>=.7 for v in values[:9]);weak=sum(v>=.3 for v in values[:9])
        later=any(v>=.7 for v in values[9:32])
        if blind and weak>=5 and later:e.append(tk[0]+'/'+str(tk[1]))
    assert maxerr<=1e-12 and sorted(e)==sorted(r['results'][key]['event_tracks'])
    checks[key]={'scalar_GT_frames_checked':n,'max_iou_abs_difference':maxerr,'events':e,'event_count':len(e),'event_fraction':len(e)/104}
launch=json.loads((H/'358_launch_manifest.json').read_text(encoding='utf-8'))
for f in launch['files']:assert hashlib.sha256((H/f['file']).read_bytes()).hexdigest()==f['sha256']
for f in p['inputs']:assert hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()==f['sha256']
common=set(checks['seed0_baseline']['events'])&set(checks['seed1_baseline']['events'])
assert sorted(common)==r['common_event_tracks']
gates={'support':len(selected)>=50 and len({k[0] for k in selected})>=3,'seed0':False,'seed1':False,'paired':len(common)>=5 and len({k.split('/')[0] for k in common})>=2}
for s in [0,1]:
    z=checks['seed'+str(s)+'_baseline'];gates['seed'+str(s)]=z['event_count']>=10 and z['event_fraction']>=.10 and len({v.split('/')[0] for v in z['events']})>=3
assert gates==r['gates']
audit={'passed':True,'scope':'independent scalar IoU for both primary models and every eligible first32 GT frame; independent event/gate calculation; input and launch SHA audit','models':checks,'gates':gates,'common_events':sorted(common),'verified_inputs':len(p['inputs']),'verified_launch_files':len(launch['files']),'elapsed_seconds':time.perf_counter()-start}
with (H/'358_independent_audit.json').open('x',encoding='utf-8') as f:json.dump(audit,f,ensure_ascii=False,indent=2)
print(json.dumps(audit),flush=True)
