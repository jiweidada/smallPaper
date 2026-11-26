"""Existing-file GT association only; no model, evaluator or tensor runtime imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,csv,gzip,hashlib,time,collections
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
RESULT=OUT/'attempt02'
PREV=ROOT/'问题优先诊断_20261026_v1'
BASE=ROOT/'data/00_实验计划与证据索引/本次24轮双基线模型下载_20261009_v1'
ANN=Path(r'C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val/annotations/val.json')
LABELS=['persistent_miss','recovered','reverse_loss','both_hit']

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(n,j):
    with open(RESULT/n,'x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvout(n,rr):
    assert rr
    with open(RESULT/n,'x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def rate(n,d):return n/d if d else None
def outcome(a,b):return LABELS[a*2+b]
def sizebin(x):return '[0,4)' if x<4 else '[4,8)' if x<8 else '[8,16)' if x<16 else '[16,32)' if x<32 else '[32,inf)'

def summarize(rr,th):
    n=len(rr);raw=sum(r[f'raw_hit{th}'] for r in rr);ema=sum(r[f'EMA_hit{th}'] for r in rr)
    c=collections.Counter(r[f'transition{th}'] for r in rr)
    # Independent2x2 arithmetic count, separate from string transition counters.
    a=np.array([r[f'raw_hit{th}'] for r in rr],dtype=bool);b=np.array([r[f'EMA_hit{th}'] for r in rr],dtype=bool)
    independent={'recovered':int((~a&b).sum()),'persistent_miss':int((~a&~b).sum()),'reverse_loss':int((a&~b).sum()),'both_hit':int((a&b).sum())}
    assert all(c[k]==v for k,v in independent.items()) and sum(c.values())==n
    assert raw==c['reverse_loss']+c['both_hit'] and ema==c['recovered']+c['both_hit']
    return {'n_gt':n,'raw_hit':raw,'raw_miss':n-raw,'EMA_hit':ema,'EMA_miss':n-ema,
        **{k:c[k] for k in LABELS},'net_hit_gain':ema-raw,
        'recovery_rate_among_raw_miss':rate(c['recovered'],n-raw),
        'persistent_rate_among_raw_miss':rate(c['persistent_miss'],n-raw),
        'reverse_loss_rate_among_raw_hit':rate(c['reverse_loss'],raw),
        'both_hit_rate_among_raw_hit':rate(c['both_hit'],raw)}

def main():
    start=time.perf_counter();frozen=sha(OUT/'protocol.json')
    assert frozen==(OUT/'protocol.sha256').read_text('utf-8').strip()
    RESULT.mkdir(exist_ok=False)
    inputs={}
    def track(p,kind):
        p=Path(p);h=sha(p);inputs[str(p)]={'sha256':h,'bytes':p.stat().st_size,'kind':kind};return h
    ah=track(ANN,'GT_annotation');assert ah=='0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'
    ds=read(ANN);anns={a['id']:a for a in ds['annotations']};images={i['id']:i for i in ds['images']}
    assert len(anns)==38759 and len(images)==548
    for n in ('per_GT.csv','results.json','scope_audit.json','verification.json','delivery_manifest.json','input_manifest.json'):
        track(PREV/n,'previous_verified_diagnostic')
    seal=read(PREV/'delivery_manifest.json')['files']
    for n in ('per_GT.csv','results.json','scope_audit.json','verification.json'):
        assert inputs[str(PREV/n)]['sha256']==seal[n]['sha256']
    prior=read(PREV/'results.json');scope=read(PREV/'scope_audit.json')
    with open(PREV/'per_GT.csv',encoding='utf-8-sig',newline='') as f:raw=list(csv.DictReader(f))
    assert len(raw)==2*len(anns)
    raw_index={(int(r['seed']),int(r['annotation_id'])):r for r in raw}
    assert len(raw_index)==len(raw)
    for seed in (1,2):assert {a for s,a in raw_index if s==seed}==set(anns)
    for aid in anns:
        for key in ('image_id','category_id','short_side_640','area_original','density32','image_n_gt','occlusion','truncation'):
            assert raw_index[1,aid][key]==raw_index[2,aid][key]
    source_code={}
    for n in ('experiment.py','results.py'):
        p=ROOT/'code/fine_residual_screen24_20261009_v2'/n;source_code[n]=track(p,'capture_source_code')
    track(PREV/'audit.py','raw_cache_and_matching_source')
    ema_hits={};ema_legacy={};ema_dt={};bindings={};id0={};verification={}
    for seed in (1,2):
        p=BASE/f'seed{seed}/run/endpoint_epoch24';manifest=read(p/'manifest.json');completion=read(p/'training_completion.json');ep=read(p/'summary.json')
        for n in ('manifest.json','training_completion.json',*manifest['files']):track(p/n,'EMA24_captured_evaluation')
        assert all(inputs[str(p/n)]['sha256']==h for n,h in manifest['files'].items())
        assert inputs[str(p/'manifest.json')]['sha256']==completion['capture_manifest_sha256']
        cp=BASE/f'seed{seed}/run/last.pth';cph=track(cp,'checkpoint_file_hash_only')
        assert cph==completion['last_checkpoint_sha256']==scope['checkpoints'][str(seed)]['sha256']
        assert completion['identity']==ep['identity'] and ep['identity']['seed']==seed and ep['identity']['arm']=='baseline'
        assert ep['identity']['val_annotation_sha256']==ah and ep['epoch_zero_based']==23 and ep['evaluated_weights']=='EMA'
        assert ep['protocol']['max_dets']==[1,10,100] and ep['protocol']['category_ids']==list(range(10))
        assert ep['protocol']['image_ids']==sorted(images) and np.allclose(ep['protocol']['iou_thresholds'],np.linspace(.5,.95,10),atol=1e-12,rtol=0)
        assert all(source_code[n]==ep['identity']['implementation'][n] for n in source_code)
        z=np.load(p/'GT_hits.npz',allow_pickle=False);aids=z['annotation_id'].astype(int);order={int(a):j for j,a in enumerate(aids)}
        assert set(aids)==set(anns) and len(order)==len(anns) and z['hits'].shape==(len(anns),10)
        assert all(int(z['category_id'][order[a]])==anns[a]['category_id'] for a in anns)
        native=np.zeros((len(anns),10),dtype=bool);gtmatch=np.zeros_like(native);details={};seen=[];checked_pairs=0
        with gzip.open(p/'matching_all_area.json.gz','rt',encoding='utf-8') as f:records=json.load(f)
        cells=set()
        for e in records:
            key=(e['image_id'],e['category_id']);assert key not in cells;cells.add(key)
            assert e['aRng']==ep['protocol']['area_ranges'][0] and e['maxDet']==100 and len(e['dtIds'])<=100
            ids=list(map(int,e['gtIds']));dids=list(map(int,e['dtIds']));gpos={a:i for i,a in enumerate(ids)};dpos={a:i for i,a in enumerate(dids)}
            assert len(gpos)==len(ids) and len(dpos)==len(dids)
            gm=np.asarray(e['gtMatches'],dtype=np.int64).reshape(10,len(ids));dm=np.asarray(e['dtMatches'],dtype=np.int64).reshape(10,len(dids))
            gi=np.asarray(e['gtIgnore'],dtype=bool);di=np.asarray(e['dtIgnore'],dtype=bool).reshape(10,len(dids))
            assert not gi.any()
            for a in ids:
                assert (anns[a]['image_id'],anns[a]['category_id'])==key
            seen.extend(ids)
            for t in range(10):
                associated={}
                for j,a in enumerate(ids):
                    did=int(gm[t,j])
                    if did>0 and not gi[j]:
                        assert did in dpos and did not in associated;associated[did]=a;k=dpos[did]
                        assert not di[t,k] and int(dm[t,k])==a
                        gtmatch[order[a],t]=True;checked_pairs+=1
                        if t in (0,5):details[a,t]={'detection_id':did,'score':float(e['dtScores'][k])}
                for k,did in enumerate(dids):
                    a=int(dm[t,k])
                    if a!=0 and not di[t,k]:
                        assert a in gpos and int(gm[t,gpos[a]])==did
                        native[order[a],t]=True
                if 0 in gpos:
                    j=gpos[0];did=int(gm[t,j]);id0.setdefault(str(seed),[]).append({'IoU':float(ep['protocol']['iou_thresholds'][t]),'gtMatched':bool(did>0),'detection_id':did,'reciprocity_valid':bool(did==0 or (did in associated and associated[did]==0 and dm[t,dpos[did]]==0)),'legacy_hit':False})
        assert len(seen)==len(anns) and set(seen)==set(anns)
        assert np.array_equal(native,z['hits']) and native.sum(0).tolist()==ep['TP_by_IoU']
        diff=np.where((native!=gtmatch).any(1))[0]
        assert {int(aids[j]) for j in diff}.issubset({0})
        for th,t in ((50,0),(75,5)):
            ema_hits[seed,th]={int(a):bool(gtmatch[j,t]) for j,a in enumerate(aids)}
            ema_legacy[seed,th]={int(a):bool(native[j,t]) for j,a in enumerate(aids)}
            ema_dt[seed,th]={a:details.get((a,t),{'detection_id':0,'score':None}) for a in anns}
        bindings[str(seed)]={'checkpoint_sha256':cph,'epoch_zero_based':23,'raw_branch':'model','EMA_branch':'ema.module',
            'capture_manifest_bound_to_checkpoint':True,'annotation_sha256':ah,'EMA_environment':ep['environment'],
            'raw_environment':scope['runtime'],'EMA_evaluator_source':ep['evaluator_source'],
            'context_note':'Existing CPU raw inference/Windows evaluation versus native GPU EMA capture; no new matched-backend replay.'}
        verification[str(seed)]={'GT_count':len(seen),'cells':len(cells),'all10_native_hit_columns_replayed_exact':True,
            'reciprocal_matched_pairs_across10thresholds':checked_pairs,'native_vs_gtmatch_different_annotation_ids':[int(aids[j]) for j in diff],
            'legacy_EMA_hits_all10':native.sum(0).tolist(),'per_GT_assignment_EMA_hits_all10':gtmatch.sum(0).tolist()}
        print(f'seed{seed}: all GT identities and10-threshold EMA matching records validated',flush=True)
    paired=[]
    for seed in (1,2):
        for aid,a in sorted(anns.items()):
            r=raw_index[seed,aid]
            assert int(r['image_id'])==a['image_id'] and int(r['category_id'])==a['category_id']
            row={'seed':seed,'image_id':a['image_id'],'annotation_id':aid,'category_id':a['category_id'],
                'short_side_640':float(r['short_side_640']),'size_group':sizebin(float(r['short_side_640'])),
                'area_original':float(r['area_original']),'density32':int(r['density32']),'occlusion':int(r['occlusion']),'truncation':int(r['truncation'])}
            for th in (50,75):
                rh=int(r[f'hit{th}']);eh=int(ema_hits[seed,th][aid]);state=r[f'partition_{th}']
                assert rh==int(state=='T')
                row.update({f'raw_state{th}':state,f'raw_hit{th}':rh,f'EMA_hit{th}':eh,f'transition{th}':outcome(rh,eh),
                    f'legacy_raw_hit{th}':rh if aid!=0 else 0,f'legacy_EMA_hit{th}':int(ema_legacy[seed,th][aid]),
                    f'EMA_matched_detection_id{th}':ema_dt[seed,th][aid]['detection_id'],f'EMA_matched_score{th}':ema_dt[seed,th][aid]['score']})
            paired.append(row)
    all_summary={};legacy_summary={};raw_flows=[];group_rows=[];group_flow_rows=[];image_rows=[];standardized={}
    for seed in (1,2):
        rr=[r for r in paired if r['seed']==seed];all_summary[str(seed)]={};legacy_summary[str(seed)]={}
        strat=collections.defaultdict(list);im=collections.defaultdict(list)
        for r in rr:
            ar=r['area_original'];d=r['density32']
            for typ,label in [('all','all'),('size',r['size_group']),('category',str(r['category_id'])),
                ('size_category',r['size_group']+'|'+str(r['category_id'])),('area','small<1024' if ar<1024 else 'medium[1024,9216)' if ar<9216 else 'large>=9216'),
                ('density32','0' if d==0 else '1-4' if d<5 else '5-14' if d<15 else '15+'),('occlusion',str(r['occlusion']))]:strat[typ,label].append(r)
            im[r['image_id']].append(r)
        for th in (50,75):
            all_summary[str(seed)][str(th)]=summarize(rr,th)
            legacy=[{**r,f'raw_hit{th}':r[f'legacy_raw_hit{th}'],f'EMA_hit{th}':r[f'legacy_EMA_hit{th}'],f'transition{th}':outcome(r[f'legacy_raw_hit{th}'],r[f'legacy_EMA_hit{th}'])} for r in rr]
            legacy_summary[str(seed)][str(th)]=summarize(legacy,th)
            assert all_summary[str(seed)][str(th)]['raw_hit']==next(g for g in prior['groups'] if g['seed']==seed and g['group_type']=='all')['T'+str(th)]
            for (typ,label),items in strat.items():
                summary=summarize(items,th);group_rows.append({'seed':seed,'IoU':th/100,'group_type':typ,'group':label,**summary})
                for state in 'GSKCMT':
                    sub=[r for r in items if r[f'raw_state{th}']==state]
                    hit=sum(r[f'EMA_hit{th}'] for r in sub)
                    flow={'seed':seed,'IoU':th/100,'group_type':typ,'group':label,'raw_state':state,'n_raw_state':len(sub),
                        'EMA_hit':hit,'EMA_miss':len(sub)-hit,'EMA_hit_rate':rate(hit,len(sub))}
                    group_flow_rows.append(flow)
                    if typ=='all':raw_flows.append(flow)
            for iid,items in im.items():image_rows.append({'seed':seed,'image_id':iid,'IoU':th/100,**summarize(items,th)})
            common=[]
            for (typ,label),items in strat.items():
                if typ!='size_category':continue
                G=[r for r in items if r[f'raw_state{th}']=='G'];K=[r for r in items if r[f'raw_state{th}']=='K']
                if G and K:common.append((label,G,K))
            total=sum(len(a)+len(b) for _,a,b in common)
            standardized[f'{seed}_{th}']={'cells':len(common),'G_overlap_n':sum(len(a) for _,a,_ in common),'K_overlap_n':sum(len(b) for _,_,b in common),
                'pooled_weighted_G_recovery_rate':sum((len(a)+len(b))*sum(r[f'EMA_hit{th}'] for r in a)/len(a) for _,a,b in common)/total if total else None,
                'pooled_weighted_K_recovery_rate':sum((len(a)+len(b))*sum(r[f'EMA_hit{th}'] for r in b)/len(b) for _,a,b in common)/total if total else None,
                'weight_population_n':total,'cells_used':[c for c,_,_ in common],'meaning':'descriptive common size x category cells; raw states are selected groups, not randomized treatments'}
    cross={}
    for th in (50,75):
        by={seed:{r['annotation_id']:r for r in paired if r['seed']==seed} for seed in (1,2)}
        c=collections.Counter((by[1][a][f'transition{th}'],by[2][a][f'transition{th}']) for a in anns)
        cross[str(th)]={'matrix':{a:{b:c[a,b] for b in LABELS} for a in LABELS},
            'both_seeds_persistent_miss':c['persistent_miss','persistent_miss'],'both_seeds_recovered':c['recovered','recovered'],
            'both_seeds_reverse_loss':c['reverse_loss','reverse_loss'],'both_seeds_both_hit':c['both_hit','both_hit']}
    image_patterns={}
    for seed in (1,2):
        image_patterns[str(seed)]={}
        for th in (50,75):
            ii=[r for r in image_rows if r['seed']==seed and r['IoU']==th/100]
            image_patterns[str(seed)][str(th)]={'all_images':len(ii),'images_with_recovery':sum(r['recovered']>0 for r in ii),
                'images_with_reverse_loss':sum(r['reverse_loss']>0 for r in ii),'images_with_both':sum(r['recovered']>0 and r['reverse_loss']>0 for r in ii),
                'images_net_gain_positive':sum(r['net_hit_gain']>0 for r in ii),'images_net_gain_negative':sum(r['net_hit_gain']<0 for r in ii),'images_net_gain_zero':sum(r['net_hit_gain']==0 for r in ii)}
    csvout('per_GT_transition.csv',paired);csvout('summary_by_group.csv',group_rows);csvout('raw_state_to_EMA.csv',raw_flows)
    csvout('raw_state_to_EMA_by_group.csv',group_flow_rows);csvout('per_image_transition.csv',image_rows)
    write('results.json',{'protocol_sha256':frozen,'all_GT_primary':all_summary,'legacy_zero_sentinel_sensitivity':legacy_summary,
        'raw_state_flows':raw_flows,'standardized_G_K':standardized,'cross_seed':cross,'image_patterns':image_patterns,'bindings':bindings,
        'id0_reconstruction':id0,'matching_verification':verification,'n_per_GT_rows':len(paired),
        'operations':{'model_imports':0,'checkpoint_deserializations':0,'model_forwards':0,'evaluator_calls':0,'optimizer_steps':0,'GPU':False,'network':False,'test_reads':0},
        'elapsed_seconds':time.perf_counter()-start})
    write('input_manifest.json',{'files':inputs})
    changed=[p for p,v in inputs.items() if sha(p)!=v['sha256']]
    assert not changed and sha(OUT/'protocol.json')==frozen
    write('verification.json',{'input_files_rehashed':len(inputs),'changed_files':changed,'protocol_unchanged':True,'script_sha256':sha(__file__),
        'ID_category_image_bijection':True,'raw_group_features_identical_across_seeds':True,'EMA_all10columns_replayed':True,
        'primary_gtmatch_reciprocity_valid':True,'independent_boolean_and_counter_totals_match':True,'all4transitions_partition_all_GT':True,
        'elapsed_seconds':time.perf_counter()-start})
    print(json.dumps({'primary':all_summary,'raw_state_flows':raw_flows,'cross_seed':cross,'standardized':standardized,'image_patterns':image_patterns},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
