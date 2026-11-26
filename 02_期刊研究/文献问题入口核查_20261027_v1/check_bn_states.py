"""Read-only BN serialized-state prerequisite check, no detector construction."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json, hashlib, gc, time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;E=ROOT/'data/00_实验计划与证据索引'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def quant(x):
    x=np.asarray(x,float)
    return {'n':len(x),'q0_q50_q90_q99_q100':np.quantile(x,[0,.5,.9,.99,1]).tolist()} if len(x) else {'n':0}
def group(p):
    for g in ['backbone','encoder','decoder']:
        if g in p.split('.'):return g
    return 'other'
proto=HERE/'01_BN状态必要条件_静态核查协议.json';ph=sha(proto)
(HERE/'01_BN状态必要条件_静态核查协议.sha256').write_text(ph,encoding='utf-8')
out={'protocol_sha256':ph,'checkpoints':{},'sources':{str(proto):ph,str(Path(__file__).resolve()):sha(Path(__file__).resolve())},'detector_constructed':False,'model_forwards':0,'training':0}
for dr in ['dn_pre_frozen_diagnostic_20261020_v1','local_candidates_20261004_v2']:
    for rel in ['optim/ema.py','backbone/common.py','backbone/hgnetv2.py','deim/hybrid_encoder.py','deim/dfine_decoder.py']:
        p=ROOT/'code'/dr/'native/engine'/rel;out['sources'][str(p)]=sha(p)
start=time.perf_counter()
for epoch in [24,132]:
    for seed in [1,2]:
        if epoch==24:
            p=E/f'本次24轮双基线模型下载_20261009_v1/seed{seed}/run/last.pth'
            cfg=E/f'本次24轮双基线模型下载_20261009_v1/seed{seed}/run/args.json'
        else:
            p=E/f'正式132轮基线模型下载_20261008_v1/seed{seed}/last.pth'
            cfg=E/f'正式132轮基线模型下载_20261008_v1/seed{seed}/run/args.json'
        h=sha(p);out['sources'][str(p)]=h;out['sources'][str(cfg)]=sha(cfg)
        cp=torch.load(p,map_location='cpu',weights_only=True);raw=cp['model'];ema=cp['ema']['module']
        assert set(raw)==set(ema)
        observed_epoch=int(cp.get('last_epoch',cp.get('epoch',-999)));assert observed_epoch==epoch-1,(epoch,observed_epoch)
        conf=json.loads(cfg.read_text(encoding='utf-8'));yc=conf.get('yaml_cfg',{})
        prefixes=sorted(k[:-len('.running_mean')] for k in raw if k.endswith('.running_mean'))
        rows=[]
        for pref in prefixes:
            a=raw[pref+'.running_mean'].double().numpy();b=ema[pref+'.running_mean'].double().numpy()
            v=raw[pref+'.running_var'].double().numpy();w=ema[pref+'.running_var'].double().numpy()
            assert a.shape==b.shape==v.shape==w.shape and np.isfinite(a).all() and np.isfinite(b).all() and (v>=0).all() and (w>=0).all()
            has=pref+'.num_batches_tracked' in raw
            rows.append({'prefix':pref,'group':group(pref),'tracked_field_present':has,'channels':len(a),
                'unequal_mean':int((a!=b).sum()),'unequal_var':int((v!=w).sum()),
                'raw_num_batches_tracked':int(raw[pref+'.num_batches_tracked']) if has else None,
                'EMA_num_batches_tracked':int(ema[pref+'.num_batches_tracked']) if has else None,
                'mean_standardized_delta':quant(np.abs(b-a)/np.sqrt(v+1e-5)),
                'abs_log_variance_ratio':quant(np.abs(np.log((w+1e-5)/(v+1e-5))))})
        summary={}
        for gr in ['backbone','encoder','decoder','other']:
            rr=[r for r in rows if r['group']==gr]
            if rr:summary[gr]={'prefixes':len(rr),'with_tracked_field':sum(r['tracked_field_present'] for r in rr),'channels':sum(r['channels'] for r in rr),
                'unequal_mean_channels':sum(r['unequal_mean'] for r in rr),'unequal_var_channels':sum(r['unequal_var'] for r in rr),
                'max_standardized_mean_delta':max(r['mean_standardized_delta']['q0_q50_q90_q99_q100'][-1] for r in rr),
                'max_abs_log_variance_ratio':max(r['abs_log_variance_ratio']['q0_q50_q90_q99_q100'][-1] for r in rr)}
        name=f'epoch{epoch}_seed{seed}'
        out['checkpoints'][name]={'checkpoint':str(p),'sha256':h,'epoch_index':observed_epoch,'EMA_updates':cp['ema'].get('updates'),
            'HGNetv2_config':yc.get('HGNetv2'),'use_ema':yc.get('use_ema'),'ema_config':yc.get('ema'),
            'rows':rows,'summary':summary,
            'dynamic_serialized_state_difference_exists':any(r['tracked_field_present'] and (r['unequal_mean'] or r['unequal_var']) for r in rows)}
        assert sha(p)==h
        print(json.dumps({'checkpoint':name,'summary':summary,'scope':'state difference only, not inaccurate-stat or performance finding'},ensure_ascii=False),flush=True)
        del cp,raw,ema;gc.collect()
assert all(sha(Path(p))==h for p,h in out['sources'].items())
out['seconds']=time.perf_counter()-start
with (HERE/'01_BN状态必要条件_结果.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({'completed':True,'seconds':out['seconds']},ensure_ascii=False),flush=True)
