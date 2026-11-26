"""287: passive raw/eval natural collector and fixed-PMF pixel-difference diagnosis.
Author may run --preflight only. Root reviews then runs --collect-real.
No active pre insertion, training, GPU, optimizer, heldout or remote access.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import pathlib
import sys
import time
import traceback
import types
sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent
WORK = ROOT.parents[1]
PIXEL = ROOT/'278_natural_development_HL_pixel_probe_isolated.py'
if hashlib.sha256(PIXEL.read_bytes()).hexdigest() != 'a393e5e822d1dd380f38edfb08690ad3497493ba62abb7da136a1ffd2072d88d':
    raise RuntimeError('278 identity')
_spec = importlib.util.spec_from_file_location('_pixel278_for288',PIXEL)
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)
np,torch,need,sha,save = P.np,P.torch,P.need,P.sha,P.save
BASE = WORK/'英文小论文/code/local_candidates_20261004_v2'
NATIVE = BASE/'native'
ASSETS = WORK/'英文小论文/data/08_匹配聚合覆盖探针_20261019_v1'
PACKAGE = ASSETS/'package_manifest.json'
RECIPE = ASSETS/'baseline_recipes.json'
OLD = ROOT/'279_natural_development_HL_pixel_probe'
PROTOCOL = ROOT/'287_A_自然pre任务共享差分_事前协议.txt'
IDS = [1,2,3,7,9,10,12,13,14]
ARMS = ('D','Ddeg','Hsingle','zero')
FIELDS = ('boxes','logits','corners','refs','pre_boxes','pre_logits')
CPROOT = WORK/'英文小论文/data/00_实验计划与证据索引/本次24轮双基线模型下载_20261009_v1'
CPS = {s:CPROOT/f'seed{s}/run/last.pth' for s in (1,2)}
CPSHA = {1:'741d57a2d6ecb7439de17fedbc8462315ab05309da2ba35b9d38d2c3e2dd9d87',
         2:'bcf1d9a7a20115835d9e9df3e09a71419f41d1612c2e7b687c47ff3c41952cf0'}
COUNTS = dict(image_decodes=0,checkpoint_loads=0,detector_builds=0,model_forwards=0,
              backwards=0,optimizer_steps=0,heldout_reads=0,annotation_reads=0)


def libraries_and_sources():
    need(sha(PACKAGE)=='7c4b4c3a6968acf7f5269cc8070caebba24a1b9e5466d41532a366b6b46ad02a','08 package identity')
    need(sha(RECIPE)=='46e5c21aa0308061ac3523871c3c9966309a0a26b4a0777245edff4c3997f4bf','recipe identity')
    sources={'script':pathlib.Path(__file__),'protocol':PROTOCOL,'pixel278':PIXEL,
             'helper136':P.HELPER,'package08':PACKAGE,'recipe08':RECIPE}
    package=json.loads(PACKAGE.read_text(encoding='utf-8-sig'))
    for item in package['files']:
        if item['relative'].startswith('native/'):
            path=BASE/item['relative']
            need(sha(path)==item['sha256'],'native identity '+str(path))
            sources[item['relative']]=path
    need('engine' not in sys.modules,'fresh process, no prior engine import')
    # Same fail-closed optional-profiler import sentinel as frozen diagnostic
    # common.py. EMA/simam use it only in __main__ demos, never model tensors.
    profiler=types.ModuleType('calflops')
    def forbidden_profiler(*args,**kwargs):
        raise RuntimeError('Profiling is outside287; optional calflops must never execute')
    profiler.calculate_flops=forbidden_profiler
    sys.modules['calflops']=profiler
    sys.path.insert(0,str(BASE/'local_deps'))
    sys.path.insert(0,str(NATIVE))
    from engine.backbone.hgnetv2 import HGNetv2
    from engine.deim.hybrid_encoder import HybridEncoder
    from engine.deim.dfine_decoder import DFINETransformer
    from engine.deim.deim import DEIM
    from engine.deim.matcher import HungarianMatcher
    from engine.deim.box_ops import box_iou,box_cxcywh_to_xyxy
    from engine.data.dataset.coco_dataset import ConvertCocoPolysToMask
    for name,module in list(sys.modules.items()):
        if name=='engine' or name.startswith('engine.'):
            if getattr(module,'__file__',None):
                path=pathlib.Path(module.__file__).resolve()
                need(NATIVE.resolve() in path.parents,'wrong native module '+name)
                sources['import:'+name]=path
    api=dict(HGNetv2=HGNetv2,HybridEncoder=HybridEncoder,DFINETransformer=DFINETransformer,
             DEIM=DEIM,Matcher=HungarianMatcher,box_iou=box_iou,xyxy=box_cxcywh_to_xyxy,
             Prepare=ConvertCocoPolysToMask)
    return api,sources


def centroid(patch,points,initial):
    # Exactly278 weights. Return un-clipped centroid for the locked difference.
    border=np.zeros((16,16),dtype=bool)
    border[[0,-1],:]=True; border[:,[0,-1]]=True
    background=patch[border].mean(axis=0)
    weight=np.square(patch-background).sum(axis=-1)
    mass=float(weight.sum())
    center=np.asarray(initial,dtype=np.float64).copy() if mass==0 else (weight[...,None]*points).sum(axis=(0,1))/mass
    return center,mass


def deltas(centers,initial):
    h,l,hd=centers
    return np.stack((4*np.tanh((h-l)/4),4*np.tanh((hd-l)/4),
                     np.clip(h-initial,-4,4),np.zeros(2)))


def read_queries(arrays,pre):
    need(pre.shape==(300,4) and np.isfinite(pre).all() and (pre[:,2:]>0).all(),'all300 pre')
    cs=[]; masses=[]; shifts=[]; windows=[]; clipped=[]
    for box in pre.astype(np.float64)*640:
        c,wh=box[:2],box[2:]
        win=np.r_[c-wh/2-4,c+wh/2+4]
        patches,points=P.patches(arrays,win)
        cm=[centroid(patches[key],points,c) for key in ('H','L','Hdegraded')]
        cc=np.stack([row[0] for row in cm]); dd=deltas(cc,c)
        old=P.readout(patches['H'],points,c)[0]
        need(np.array_equal(dd[2],old),'Hsingle must exactly reproduce278')
        cs.append(cc); masses.append([row[1] for row in cm]); shifts.append(dd); windows.append(win)
        clipped.append(bool((win[:2]<0).any() or (win[2:]>640).any()))
    return dict(centroids=np.stack(cs),mass=np.array(masses),deltas=np.stack(shifts),
                windows=np.stack(windows),window_outside_input=np.array(clipped))


def native_targets(rows,size,api):
    # Invoke native preparation on a metadata-only image object; no image access.
    class SizeOnly:
        pass
    stub=SizeOnly(); stub.size=size
    _,t=api['Prepare'](False)(stub,dict(image_id=0,annotations=rows))
    noncrowd=[a for a in rows if not a.get('iscrowd',0)]
    b=torch.as_tensor([a['bbox'] for a in noncrowd],dtype=torch.float32).reshape(-1,4)
    b[:,2:]+=b[:,:2]; b[:,0::2].clamp_(0,size[0]); b[:,1::2].clamp_(0,size[1])
    keep=(b[:,2]>b[:,0])&(b[:,3]>b[:,1])
    need(torch.equal(b[keep],t['boxes']),'annotation identity/native filter')
    aids=np.array([a['id'] for a,k in zip(noncrowd,keep.tolist()) if k],dtype=np.int64)
    # Match native eval Resize to640 first, then standard cxcywh normalization.
    xy=t['boxes']*torch.tensor([640/size[0],640/size[1]]*2,dtype=torch.float32)
    normalized=P.torchvision.ops.box_convert(xy,'xyxy','cxcywh')/640
    return dict(boxes=normalized,labels=t['labels']),aids


def synthetic(api):
    checks=P.synthetic()
    points=P.grid([-1,-2,15,14]); patch=np.zeros((16,16,3)); patch[8,10]=1
    c,m=centroid(patch,points,np.array([7,6]))
    need(np.array_equal(np.clip(c-[7,6],-4,4),P.readout(patch,points,[7,6])[0]),'oldreader exact')
    need(np.array_equal(deltas(np.stack((c,c,c)),np.array([7,6]))[:2],np.zeros((2,2))),'shared view exact zero')
    cc=np.array([[5.,-6.],[1.,-2.],[1.,-2.]])
    dd=deltas(cc,np.zeros(2))
    need(np.array_equal(dd[0],4*np.tanh(np.array([1.,-1.]))) and np.array_equal(dd[1],[0,0]),'locked positive sign /4')
    need(np.array_equal(dd[2],[4,-4]) and np.array_equal(dd[3],[0,0]),'four arms')
    need(np.array_equal(centroid(np.ones((16,16,3)),points,[12,13])[0],[12,13]),'zero mass fallback')
    rows=[dict(id=1,bbox=[-2,-1,7,5],category_id=0,area=35,iscrowd=0),
          dict(id=2,bbox=[100,1,4,4],category_id=1,area=16,iscrowd=0),
          dict(id=3,bbox=[2,1,4,4],category_id=2,area=16,iscrowd=1),
          dict(id=4,bbox=[1,1,4,4],category_id=3,area=16,iscrowd=0,ignore=1)]
    t,ids=native_targets(rows,(20,10),api)
    need(np.array_equal(ids,[1,4]) and t['boxes'].shape==(2,4),'native clamp/crowd/positive/ignore semantics')
    base=np.array([[.5,.5,.1,.2]],dtype=np.float32); d=np.array([[3.,-2.]])
    # With frozen offset and fixed width,height, pre and late translations agree.
    off=np.array([[.02,-.03,0,0]])
    need(np.allclose((base+off)[0,:2]+d[0]/640,(base[0,:2]+d[0]/640)+off[0,:2],rtol=0,atol=1e-16),'fixed offset translation equality')
    need(all(x==0 for x in COUNTS.values()),'preflight touched real execution')
    checks.update(shared_difference_exact_zero=True,positive_sign_tanh4_locked=True,
                  Hsingle278_bitexact=True,zero_mass_fallback=True,native_annotation_identity=True,
                  fixed_offset_equals_late_translation=True,native_import_without_model=True,
                  optional_profiler='fail-closed sentinel; no tensor operator replaced')
    return checks


def build_models(api,recipes):
    models={}; matchers={}; states={}
    for s in (1,2):
        P.guard(); need(sha(CPS[s])==CPSHA[s],'CP identity')
        y=recipes[str(s)]; need(y['HGNetv2']['pretrained'] is False,'no pretrained download')
        raw=torch.load(CPS[s],map_location='cpu',weights_only=False); COUNTS['checkpoint_loads']+=1
        need(raw['last_epoch']==23 and 'model' in raw,'BASE24 raw epoch23')
        m=api['DEIM'](api['HGNetv2'](**copy.deepcopy(y['HGNetv2'])),
            api['HybridEncoder'](**copy.deepcopy(y['HybridEncoder']),eval_spatial_size=[640,640]),
            api['DFINETransformer'](**copy.deepcopy(y['DFINETransformer']),num_classes=10,eval_spatial_size=[640,640]),num_classes=10)
        COUNTS['detector_builds']+=1
        m.load_state_dict(raw['model'],strict=True); del raw
        m.eval(); m.requires_grad_(False)
        need(not m.training and m.decoder.num_queries==300,'eval300')
        kw=copy.deepcopy(y['DEIMCriterion']['matcher']); kw.pop('type',None)
        matchers[s]=api['Matcher'](**kw,use_focal_loss=True)
        models[s]=m; states[s]=state_hash(m); gc.collect(); P.guard()
    return models,matchers,states


def state_hash(model):
    dig=hashlib.sha256()
    for key,value in model.state_dict().items():
        dig.update(key.encode()); dig.update(str(value.dtype).encode()); dig.update(str(tuple(value.shape)).encode())
        dig.update(value.detach().cpu().contiguous().numpy().tobytes())
    return dig.hexdigest()


def capture(model,x,identity=False):
    P.guard(); need(COUNTS['model_forwards']<20,'20 forward ceiling'); saved={}
    def hook(module,args,output):
        need(isinstance(output,tuple) and len(output)==6,'native six tuple')
        saved.update({key:value.detach().cpu().clone() for key,value in zip(FIELDS,output)})
        return output if identity else None
    handle=model.decoder.decoder.register_forward_hook(hook)
    start=time.monotonic()
    try:
        COUNTS['model_forwards']+=1
        with torch.no_grad():
            public=model(x)
    finally:
        handle.remove()
    seconds=time.monotonic()-start
    need(set(public)=={'pred_boxes','pred_logits'},'eval public output fields')
    need(torch.equal(saved['boxes'][-1],public['pred_boxes']) and torch.equal(saved['logits'][-1],public['pred_logits']),'public/internal parity')
    saved.update({key:value.detach().cpu().clone() for key,value in public.items()})
    need(saved['pre_boxes'].shape==(1,300,4),'pre300')
    need(all(torch.isfinite(v).all() for v in saved.values()),'finite native fields')
    P.guard(); return saved,seconds


def paired_iou(a,b,api):
    return api['box_iou'](api['xyxy'](a),api['xyxy'](b))[0].diag().numpy()


def score(saved,readout,target,annotation_ids,selected_ids,matcher,api):
    # The only GT-dependent call follows the completed all300 pixel readout.
    outputs={key:saved[key] for key in ('pred_boxes','pred_logits')}
    q,g=matcher(outputs,[target])['indices'][0]
    need(len(q)==min(300,len(annotation_ids)),'native min(N,Q)')
    pre=saved['pre_boxes'][0,q]; final=saved['pred_boxes'][0,q]; gt=target['boxes'][g]
    shift=readout['deltas'][q.numpy()]
    err=np.linalg.norm((pre[:,:2]-gt[:,:2]).numpy()[:,None,:]*640+shift,axis=-1)
    pre_iou=paired_iou(pre,gt,api); final_iou=paired_iou(final,gt,api)
    final_arm=[]
    for j in range(4):
        fb=final.clone(); fb[:,:2]+=torch.from_numpy(shift[:,j]/640).to(torch.float32)
        final_arm.append(paired_iou(fb,gt,api))
    ious=np.stack(final_arm,axis=1)
    tiny=(gt[:,2:].min(-1).values*640<4).numpy()
    aids=annotation_ids[g.numpy()]
    groups={'actual_sub4':tiny,'all_native_own':np.ones(len(q),dtype=bool),
            'selected279_64':np.isin(aids,list(selected_ids)), 'actual_sub4_pre_iou0':tiny&(pre_iou==0)}
    stats={}
    for name,mask in groups.items():
        n=int(mask.sum())
        stats[name]=dict(n=n,center_error_mean=err[mask].mean(0).tolist() if n else None,
            cached_final_iou_mean=ious[mask].mean(0).tolist() if n else None,
            cached_IoU05_up=(((ious>=.5)&(final_iou[:,None]<.5))&mask[:,None]).sum(0).tolist(),
            cached_IoU05_down=(((ious<.5)&(final_iou[:,None]>=.5))&mask[:,None]).sum(0).tolist())
    arrays=dict(own_q=q.numpy(),own_g=g.numpy(),own_annotation_ids=aids,center_errors=err,
        native_pre_iou=pre_iou,native_final_iou=final_iou,cached_final_arm_iou=ious,
        target_boxes=target['boxes'].numpy(),target_labels=target['labels'].numpy(),target_annotation_ids=annotation_ids)
    matched=set(aids.tolist())
    return stats,arrays,[int(a) for a in annotation_ids if int(a) not in matched]


def aggregate(records):
    result={}
    for seed in (1,2):
        groups={}
        for name in ('actual_sub4','all_native_own','selected279_64','actual_sub4_pre_iou0'):
            rows=[r for r in records if r['seed']==seed and r['groups'][name]['n']]
            values=np.array([r['groups'][name]['center_error_mean'] for r in rows])
            if not rows:
                groups[name]=dict(images=0,mean=None); continue
            dz=values[:,0]-values[:,3]; dg=values[:,0]-values[:,1]
            groups[name]=dict(images=len(rows),image_ids=[r['image_id'] for r in rows],
                n=sum(r['groups'][name]['n'] for r in rows),image_equal_error=dict(zip(ARMS,values.mean(0).tolist())),
                D_minus_zero_mean=float(dz.mean()),D_minus_Ddeg_mean=float(dg.mean()),
                D_vs_zero_win_tie_loss=[int((dz<0).sum()),int((dz==0).sum()),int((dz>0).sum())],
                D_vs_Ddeg_win_tie_loss=[int((dg<0).sum()),int((dg==0).sum()),int((dg>0).sum())],
                per_image_D_minus_zero=dz.tolist(),per_image_D_minus_Ddeg=dg.tolist())
        result[str(seed)]=groups
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true'); mode.add_argument('--collect-real',action='store_true')
    args=parser.parse_args(); started=time.monotonic(); resources=P.runtime()
    api,sources=libraries_and_sources(); checks=synthetic(api)
    before={key:sha(path) for key,path in sources.items()}
    if args.preflight:
        dest=ROOT/'289_preflight.json'; need(not dest.exists(),'fresh preflight output')
        need(before=={key:sha(path) for key,path in sources.items()},'preflight source unchanged')
        save(dest,dict(state='preflight_passed_no_real_forward',tests=checks,counts=COUNTS,resources=resources,
            sources={key:str(path) for key,path in sources.items()},source_sha256=before,seconds=time.monotonic()-started))
        print('preflight passed: '+str(dest),flush=True); return
    out=ROOT/'289_BASE24_natural_pre_HL_delta'; need(not out.exists(),'fresh289 directory required'); out.mkdir()
    stage='metadata'; records=[]; image_hashes={}; models={}; source_after={}
    try:
        need(sha(OLD/'00_manifest.json')=='a710910f9341999b220d768c83cd4162833194560df7925db5abf97780d7a1ba','279 panel identity')
        old=json.loads((OLD/'00_manifest.json').read_text(encoding='utf-8-sig'))
        selected_ids={r['annotation_id'] for r in old['panel']['selected']}
        need(sorted({r['image_id'] for r in old['panel']['selected']})==IDS,'fixed nine images')
        need(sha(P.PLAN)==P.PLAN_SHA and sha(P.ANNOTATION)==P.ANNOTATION_SHA,'plan/annotation identity')
        plan=json.loads(P.PLAN.read_text(encoding='utf-8-sig'))
        need(set(IDS)<=set(plan['source_pools']['development']) and not set(IDS)&set(plan['source_pools']['heldout']),'development only')
        ann=json.loads(P.ANNOTATION.read_text(encoding='utf-8-sig')); COUNTS['annotation_reads']+=1
        images={r['id']:r for r in ann['images'] if r['id'] in IDS}; annotations={i:[] for i in IDS}
        for row in ann['annotations']:
            if row['image_id'] in annotations: annotations[row['image_id']].append(row)
        del ann; gc.collect()
        paths={i:P.DATA/'images'/images[i]['file_name'] for i in IDS}
        for i,path in paths.items():
            need(str(path)==old['image_paths'][str(i)],'image path binding')
            image_hashes[str(i)]=sha(path); need(image_hashes[str(i)]==old['image_sha256_before'][str(i)],'image identity')
        sources.update(plan=P.PLAN,annotation=P.ANNOTATION,old279_manifest=OLD/'00_manifest.json')
        before.update({key:sha(sources[key]) for key in ('plan','annotation','old279_manifest')})
        recipes=json.loads(RECIPE.read_text(encoding='utf-8-sig'))
        stage='models'; models,matchers,states=build_models(api,recipes)
        save(out/'00_manifest.json',dict(protocol=str(PROTOCOL),sources={k:str(v) for k,v in sources.items()},
            source_sha256=before,image_paths={str(k):str(v) for k,v in paths.items()},image_sha256=image_hashes,
            checkpoint_paths={str(k):str(v) for k,v in CPS.items()},checkpoint_sha256=CPSHA,
            weights='raw model, not EMA',epoch=23,image_ids=IDS,selected279_ids=sorted(selected_ids),
            arms=list(ARMS),resources=resources,synthetic_checks=checks,state_sha_before=states,
            identity_scope='passive observer only; no active pre injection',max_forwards=20))
        for iid in IDS:
            stage=f'image_{iid}'; P.guard(); need(sha(paths[iid])==image_hashes[str(iid)],'predecode identity')
            with P.Image.open(paths[iid]) as source:
                rgb=source.convert('RGB'); rgb.load(); COUNTS['image_decodes']+=1
            need(rgb.size==(images[iid]['width'],images[iid]['height']),'source dimensions')
            arrays=P.views(rgb)
            # Actual native eval input, not float64-array round trip.
            low=rgb.resize((640,640),resample=P.Image.Resampling.BILINEAR)
            x=P.v2.functional.to_dtype(P.v2.functional.pil_to_tensor(low),dtype=torch.float32,scale=True).unsqueeze(0)
            need(x.shape==(1,3,640,640) and x.dtype==torch.float32,'native640 input')
            target,aids=native_targets(annotations[iid],rgb.size,api)
            for seed in (1,2):
                saved,secs=capture(models[seed],x)
                identity=None
                if iid==IDS[0]:
                    repeat,identity_secs=capture(models[seed],x,identity=True)
                    need(set(saved)==set(repeat) and all(torch.equal(saved[k],repeat[k]) for k in saved),'all eight identity fields bitexact')
                    identity=dict(seconds=identity_secs,all_eight_fields_bitexact=True); del repeat
                readout=read_queries(arrays,saved['pre_boxes'][0].numpy())
                stats,scored,unmatched=score(saved,readout,target,aids,selected_ids,matchers[seed],api)
                name=f'seed{seed}_image{iid:04d}'
                payload={**{k:v.numpy() for k,v in saved.items()},**readout,**scored}
                np.savez_compressed(out/(name+'.npz'),**payload)
                row=dict(seed=seed,image_id=iid,forward_seconds=secs,identity=identity,groups=stats,
                    query_crops=300,window_outside_input=int(readout['window_outside_input'].sum()),
                    native_GT=len(aids),unmatched_annotation_ids=unmatched,
                    valid_ignore_GT=sum(bool(a.get('ignore',0)) and a['id'] in set(aids.tolist()) for a in annotations[iid]),
                    cache_only=True,nonzero_intervention_forwards=0)
                records.append(row); save(out/(name+'.json'),row)
                print(json.dumps(dict(image=iid,seed=seed,forwards=COUNTS['model_forwards'],seconds=secs),ensure_ascii=False),flush=True)
                del saved,readout,scored,payload; gc.collect(); P.guard()
            rgb.close(); low.close(); del arrays,x,target; gc.collect()
        need(COUNTS['model_forwards']==20 and COUNTS['image_decodes']==9 and COUNTS['checkpoint_loads']==2,'exact budget counts')
        need(all(state_hash(models[s])==states[s] for s in (1,2)),'model state unchanged')
        need(all(sha(CPS[s])==CPSHA[s] for s in (1,2)),'CP unchanged')
        need(all(sha(paths[i])==image_hashes[str(i)] for i in IDS),'images unchanged')
        source_after={k:sha(v) for k,v in sources.items()}; need(before==source_after,'source unchanged')
        save(out/'01_results.json',dict(state='complete',arms=list(ARMS),aggregate=aggregate(records),records=records,
            counts=COUNTS,resources=resources,seconds=time.monotonic()-started,source_sha_before=before,
            source_sha_after=source_after,sources_unchanged=True,image_sha_unchanged=True,
            model_state_unchanged=True,checkpoint_unchanged=True,training_authorized=False,
            AP=None,active_pre_injection=False,independent_image_panels=1,
            scope='raw/eval natural all300 observed; fixed-PMF cached translation equals late translation'))
        save(out/'02_output_sha256.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
        print('complete: '+str(out),flush=True)
    except Exception:
        save(out/'FAILED.json',dict(stage=stage,counts=COUNTS,seconds=time.monotonic()-started,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
