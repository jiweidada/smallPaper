"""358: frozen cached-only temporal necessary-condition screen. No torch/model/training."""
import json, hashlib, sys, time, datetime, platform
from pathlib import Path
from collections import defaultdict
import numpy as np
from uavdt_protocol_frozen import NativeScorer, contained_mask, match_frame, curve

HERE = Path(__file__).resolve().parent
P = json.loads((HERE/'358_protocol.json').read_text(encoding='utf-8'))
START = time.perf_counter()

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()

def save(name,obj):
    with (HERE/name).open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def verify_inputs():
    errors=[]
    for row in P['inputs']:
        p=Path(row['path'])
        if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']: errors.append(str(p))
    if errors: raise RuntimeError('Input changed: '+str(errors))
    return {'verified_input_records':len(P['inputs']),'errors':errors}

def overlaps(gt,dt):
    if len(dt)==0: return np.zeros((0,len(gt)),dtype=np.float64)
    wh=np.maximum(0.,np.minimum((dt[:,:2]+dt[:,2:4])[:,None,:],(gt[:,:2]+gt[:,2:4])[None,:,:])-np.maximum(dt[:,None,:2],gt[None,:,:2]))
    inter=wh[:,:,0]*wh[:,:,1]
    union=dt[:,2,None]*dt[:,3,None]+gt[None,:,2]*gt[None,:,3]-inter
    return np.divide(inter,union,out=np.zeros_like(inter),where=union>0)

def first_hit(vals,threshold):
    ids=np.flatnonzero(np.asarray(vals)>=threshold)
    return int(ids[0]+1) if len(ids) else None

def blind_max(rows,vals,threshold):
    best=run=0; prev=None
    for r,v in zip(rows,vals):
        if prev is None or r['frame']!=prev+1: run=0
        run=run+1 if v<threshold else 0
        best=max(best,run);prev=r['frame']
    return best

def temporal(key,features,tracks,eligible):
    geometry=features['max_iou']; tp70=features['native_tp70']; tp50=features['native_tp50']
    track_result=[]; per_seq={}; hist={'.70':defaultdict(int),'.50':defaultdict(int)}
    for tk,rows in sorted(tracks.items()):
        seq=rows[0]['sequence']; ix=np.array([r['ann_index'] for r in rows]); vals=geometry[ix]
        seqres=per_seq.setdefault(seq,{'GT':0,'small_GT':0,'small_geometry_hit70':0,'small_native_tp70':0,'eligible_onsets':0,'event_tracks':[], 'history70':defaultdict(int)})
        small=np.array([r['small'] for r in rows]); seqres['GT']+=len(rows);seqres['small_GT']+=int(small.sum())
        seqres['small_geometry_hit70']+=int(((vals>=.7)&small).sum());seqres['small_native_tp70']+=int(tp70[ix][small].sum())
        el=tk in eligible
        event=bool(el and (vals[:9]<.7).all() and (vals[:9]>=.3).sum()>=5 and (vals[9:32]>=.7).any())
        if el:
            seqres['eligible_onsets']+=1
            if event:seqres['event_tracks'].append(tk)
        record={'track':tk,'sequence':seq,'first_annotated_frame':rows[0]['frame'],'annotated_frames':len(rows),'eligible_onset':el,'first_geometry_hit70_annotated_ordinal':first_hit(vals,.7),'first_geometry_hit50_annotated_ordinal':first_hit(vals,.5),'max_contiguous_geometry_blind_run70':blind_max(rows,vals,.7),'max_contiguous_geometry_blind_run50':blind_max(rows,vals,.5),'event':event}
        if el:
            record.update(first32_max_iou=vals[:32].tolist(),first9_weak_frames=int((vals[:9]>=.3).sum()),first32_hit70=first_hit(vals[:32],.7),first32_hit50=first_hit(vals[:32],.5),first32_native_tp70=tp70[ix[:32]].astype(int).tolist())
        track_result.append(record)
        for j in range(8,len(rows)):
            window=rows[j-8:j+1]
            if not all(r['small'] for r in window) or any(window[k]['frame']!=window[0]['frame']+k for k in range(9)):continue
            for label,thr in [('.70',.7),('.50',.5)]:
                hit=vals[j-8:j+1]>=thr
                kind='current_hit' if hit[-1] else ('current_miss_with_previous_hit' if hit[:-1].any() else 'blind_all9')
                hist[label]['eligible_windows']+=1;hist[label][kind]+=1
                if label=='.70':seqres['history70']['eligible_windows']+=1;seqres['history70'][kind]+=1
    events=[r['track'] for r in track_result if r['event']]
    eventseq=sorted({r['sequence'] for r in track_result if r['event']})
    elrows=[r for r in track_result if r['eligible_onset']]
    onset_summary={}
    for suffix in ['70','50']:
        hits=[r['first32_hit'+suffix] for r in elrows]
        onset_summary[suffix]={'hit_frame1':sum(v==1 for v in hits),'first_hit_frames2_to9':sum(v is not None and 2<=v<=9 for v in hits),'first_hit_frames10_to32':sum(v is not None and 10<=v<=32 for v in hits),'no_hit_by32':sum(v is None for v in hits)}
    return {'model':key,'event_count':len(events),'event_fraction':len(events)/len(eligible),'event_sequences':eventseq,'event_tracks':events,'per_sequence':per_seq,'history':hist,'onset_first_hit':onset_summary,'tracks':track_result,'primary_seed_gate':bool(len(events)>=10 and len(events)/len(eligible)>=.10 and len(eventseq)>=3),'all_GT_geometry_hit70':int((geometry>=.7).sum()),'all_GT_native_tp70':int(tp70.sum()),'all_GT_native_tp50':int(tp50.sum())}

def main():
    verify_inputs()
    ann=json.loads(Path(P['annotation_path']).read_text(encoding='utf-8'))
    scorer=NativeScorer(ann)
    expected=json.loads(Path(P['expected_metrics_path']).read_text(encoding='utf-8'))['models_raw_0_to_1']
    anns_by_img=defaultdict(list); tracks=defaultdict(list); seq_first={}
    for im in ann['images']:seq_first[im['uav_sequence']]=min(seq_first.get(im['uav_sequence'],im['uav_frame']),im['uav_frame'])
    for i,a in enumerate(ann['annotations']):
        im=scorer.images[a['image_id']];w,h=a['bbox'][2:]
        size=float(np.sqrt(w*h*640/im['width']*640/im['height']))
        r={'ann_index':i,'ann_id':a['id'],'image_id':a['image_id'],'sequence':im['uav_sequence'],'frame':im['uav_frame'],'track_id':a['uav_track_id'],'size640':size,'small':size<32}
        anns_by_img[a['image_id']].append(r); tracks[im['uav_sequence']+'/'+str(a['uav_track_id'])].append(r)
    for rows in tracks.values():
        rows.sort(key=lambda r:r['frame'])
        assert len({r['frame'] for r in rows})==len(rows), 'duplicate track/frame annotation'
    eligible={k for k,rows in tracks.items() if rows[0]['frame']>seq_first[rows[0]['sequence']] and len(rows)>=32 and all(rows[j]['frame']==rows[0]['frame']+j for j in range(32)) and all(r['small'] for r in rows[:9])}
    assert len(eligible)==104
    assert len(ann['annotations'])==87719 and len(scorer.images)==5153
    assert sum(r['small'] for rows in tracks.values() for r in rows)==57546
    save('358_cohort.json',{'eligible_tracks':sorted(eligible),'GT':len(ann['annotations']),'small_GT':57546,'all_tracks':len(tracks),'sequences':seq_first,'annotation_index_order':[a['id'] for a in ann['annotations']]})
    features_all={}; audit={}
    for key in P['models']:
        model_start=time.perf_counter();folder=Path(P['cache_root'])/key
        binding=json.loads((folder/'binding.json').read_text(encoding='utf-8'))
        assert binding['epoch_zero_based']==35 and binding['EMA_strict_load'] is True
        assert sha(folder/'predictions.npz')==binding['predictions_sha256']
        with np.load(folder/'predictions.npz',allow_pickle=False) as z:
            assert z['columns'].tolist()==['image_id','x','y','w','h','score']
            raw=z['detections']
        assert raw.shape==(5153*300,6) and np.isfinite(raw).all()
        raw=raw[np.argsort(raw[:,0],kind='stable')]
        blocks=np.split(raw,np.flatnonzero(np.diff(raw[:,0]))+1)
        grouped={int(b[0,0]):b[:,1:] for b in blocks}
        assert set(grouped)==set(scorer.images) and all(len(b)==300 for b in grouped.values())
        metrics_parts={t:[] for t in [.5,.7,.75]}
        feat={'max_iou':np.zeros(87719,np.float64),'native_tp70':np.zeros(87719,np.bool_),'native_tp50':np.zeros(87719,np.bool_)}
        ignored=invalid=invalid_after=0
        for image_id in scorer.order:
            dt=grouped[image_id];im=scorer.images[image_id];gt=scorer.gt[image_id]
            invalid+=int(((dt[:,2]<=0)|(dt[:,3]<=0)).sum())
            if im['uav_apply_native_ignore']:
                mask=contained_mask(dt,im['uav_ignore_boxes']);ignored+=int(mask.sum());dt=dt[~mask]
            invalid_after+=int(((dt[:,2]<=0)|(dt[:,3]<=0)).sum())
            ids=np.array([r['ann_index'] for r in anns_by_img[image_id]],dtype=np.int64)
            assert len(ids)==len(gt)
            ov=overlaps(gt,dt)
            if len(dt):feat['max_iou'][ids]=ov.max(axis=0)
            for t in metrics_parts:
                scores,assignment,_=match_frame(gt,dt,t)
                metrics_parts[t].append((scores,(assignment>=0).astype(np.int8)))
                if t in [.5,.7]:feat['native_tp'+str(int(t*100))][ids[assignment[assignment>=0]]]=True
        met={}
        for t,parts in metrics_parts.items():
            val=curve(np.concatenate([x[0] for x in parts]),np.concatenate([x[1] for x in parts]),scorer.count)[0]
            met['AP'+str(int(100*t))]=val['AP70'];met['Recall'+str(int(100*t))]=val['AR70']
        diffs={k:abs(met[k]-expected[key][k]) for k in met}
        audit[key]={'binding':binding,'metrics':met,'reference':expected[key],'absolute_differences':diffs,'max_absolute_difference':max(diffs.values()),'ignored_predictions':ignored,'nonpositive_extents_original':invalid,'nonpositive_extents_after_ignore':invalid_after,'elapsed_seconds':time.perf_counter()-model_start}
        save('358_reproduction_'+key+'.json',audit[key])
        if max(diffs.values())>1e-10:raise RuntimeError('Reproduction failed '+key+' '+str(diffs))
        np.savez_compressed(HERE/('358_geometry_'+key+'.npz'),**feat)
        features_all[key]=feat
        print(json.dumps({'stage':'reproduction_pass','model':key,'max_abs':max(diffs.values()),'seconds':audit[key]['elapsed_seconds']}),flush=True)
    save('358_reproduction_all.json',audit)
    results={}
    for key in P['models']:
        r=temporal(key,features_all[key],tracks,eligible)
        save('358_tracks_'+key+'.json',r.pop('tracks'));results[key]=r
    common=sorted(set(results['seed0_baseline']['event_tracks'])&set(results['seed1_baseline']['event_tracks']))
    commonseq=sorted({x.split('/')[0] for x in common})
    gates={'support':len(eligible)>=50 and len({x.split('/')[0] for x in eligible})>=3,'seed0':results['seed0_baseline']['primary_seed_gate'],'seed1':results['seed1_baseline']['primary_seed_gate'],'paired':len(common)>=5 and len(commonseq)>=2}
    record={'protocol_sha256':sha(HERE/'358_protocol.json'),'script_sha256':sha(__file__),'status':'necessary_condition_pass_only' if all(gates.values()) else 'stopped_failed_frozen_screen','gates':gates,'eligible_tracks':len(eligible),'common_event_tracks':common,'common_event_sequences':commonseq,'results':results,'verification':verify_inputs(),'execution':{'elapsed_seconds':time.perf_counter()-START,'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'model_loads':0,'image_forwards':0,'optimizer_steps':0,'training_started':False,'prediction_files_read':6,'frames_per_model':5153,'test_frames_read':0},'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    save('358_result.json',record)
    print(json.dumps({'stage':'complete','status':record['status'],'gates':gates,'event_counts':{k:r['event_count'] for k,r in results.items()},'common_events':len(common),'seconds':record['execution']['elapsed_seconds']}),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        save('358_failure.json',{'error':repr(exc),'elapsed_seconds':time.perf_counter()-START,'training_started':False})
        raise
