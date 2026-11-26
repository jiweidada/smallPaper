"""277: user-site-isolated runtime revision of268; unchanged267 pixel method.

No detector/checkpoint/training. Default refuses image access. --synthetic-only
uses in-memory RGB fixtures; only root may invoke --real-images after review.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import pathlib
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent
WORK = ROOT.parents[1]
HELPER = ROOT/'136_C0_pmf_translation_prior_fp32.py'
HELPER_SHA = '10bc190cce57688cd8c04b31190fc356244cc2afd6df5b6c74f8bed4eca6ab26'
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
    raise RuntimeError('frozen resource helper changed')
_spec = importlib.util.spec_from_file_location('_pixel268_helper136', HELPER)
if _spec is None or _spec.loader is None:
    raise RuntimeError('resource helper unavailable')
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)  # Definitions only. No model/history main.
np,torch,psutil = H.np,H.torch,H.psutil
need,sha,save = H.require,H.sha,H.save
import PIL
from PIL import Image
import torchvision
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode

PROTOCOL = ROOT/'277_A_HL探针库路径与用户site隔离修订.txt'
PLAN = WORK/'英文小论文/data/08_匹配聚合覆盖探针_20261019_v1/sample_plan.json'
PLAN_SHA = '825bbfb56b6e2db79209460dd6b22ba06a2450b93ab863e46a3a147c2faacc99'
DATA = pathlib.Path('C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-train')
ANNOTATION = DATA/'annotations/train.json'
ANNOTATION_SHA = '32c1cae852e084430ea74503eb660f0707ec6b5e65d0998ce3ec8b448265c883'
IMAGE_IDS = [1,2,3,5,6,7,9,10,12,13,14,16,17,18,19,20]
SHIFTS = np.array([[2.,0.],[-2.,0.],[0.,2.],[0.,-2.]],dtype=np.float64)
ARMS = ('L','H','Hdegraded')
GRID = 16
RHO = 4.
COUNTS = {key:0 for key in ('real_image_decodes','real_image_hash_calls','annotation_reads','plan_reads',
    'checkpoint_loads','detector_builds','model_forwards','model_backwards','optimizer_steps','heldout_image_reads')}


def guard():
    need(psutil.virtual_memory().available/2**30 > 8, 'RAM must remain above8GiB')
    need(not torch.is_grad_enabled(), 'autograd must stay disabled')


def runtime():
    need(sys.flags.no_user_site == 1, 'invoke Python with -s: user site must be disabled')
    resources = H.resource_limits()
    observed=dict(priority=resources['priority'],affinity=resources['affinity'],torch=torch.__version__,
        torchvision=torchvision.__version__,Pillow=PIL.__version__,numpy=np.__version__,Pillow_path=PIL.__file__)
    need(resources['priority']=='below_normal' and resources['affinity']==[13,15]
         and torch.__version__=='2.5.1+cpu' and torch.version.cuda is None
         and torchvision.__version__=='0.20.1+cpu' and PIL.__version__=='10.4.0'
         and np.__version__=='1.26.4', 'frozen CPU/library versions: '+repr(observed))
    resources.update(Pillow=PIL.__version__,torchvision=torchvision.__version__,python=sys.version,
        executable=sys.executable,numpy_file=np.__file__,no_user_site=sys.flags.no_user_site,RGB='Pillow convert RGB, no EXIF transpose',
        resize='Pillow BILINEAR; synthetic parity vs torchvision PIL antialias=True',
        grid='edge coordinates; pixel centers index+.5; float64 bilinear border replicate')
    return resources


def source_manifest():
    files=dict(script=pathlib.Path(__file__),protocol=PROTOCOL,helper136=HELPER,
        PIL_Image=pathlib.Path(Image.__file__),PIL_core=pathlib.Path(Image.core.__file__),
        torchvision_resize=pathlib.Path(inspect.getfile(v2.Resize)))
    return files,{key:sha(path) for key,path in files.items()}


def sample_bilinear(array, points_edge):
    """Read-only sample of HWC at edge coordinates with replicated boundaries."""
    a=np.asarray(array,dtype=np.float64)
    need(a.ndim==3 and a.shape[2]==3 and np.isfinite(a).all(), 'finite RGB input')
    h,w=a.shape[:2]
    x=points_edge[...,0]-.5; y=points_edge[...,1]-.5
    ix=np.floor(x).astype(np.int64); iy=np.floor(y).astype(np.int64)
    ax=(x-ix)[...,None]; ay=(y-iy)[...,None]
    x0=np.clip(ix,0,w-1); x1=np.clip(ix+1,0,w-1)
    y0=np.clip(iy,0,h-1); y1=np.clip(iy+1,0,h-1)
    return (1-ay)*((1-ax)*a[y0,x0]+ax*a[y0,x1])+ay*((1-ax)*a[y1,x0]+ax*a[y1,x1])


def grid(window):
    window=np.asarray(window,dtype=np.float64)
    need(window.shape==(4,) and (window[2:]>window[:2]).all(), 'positive observation window')
    axis=(np.arange(GRID,dtype=np.float64)+.5)/GRID
    xs=window[0]+axis*(window[2]-window[0]); ys=window[1]+axis*(window[3]-window[1])
    xx,yy=np.meshgrid(xs,ys)
    return np.stack((xx,yy),axis=-1)


def views(rgb):
    """rgb was decoded once. Resize twice; never reopen/redecode its source."""
    need(rgb.mode=='RGB', 'source must be RGB')
    low=rgb.resize((640,640),resample=Image.Resampling.BILINEAR)
    degraded=low.resize(rgb.size,resample=Image.Resampling.BILINEAR)
    return {'H':np.asarray(rgb,dtype=np.float64)/255.,'L':np.asarray(low,dtype=np.float64)/255.,
            'Hdegraded':np.asarray(degraded,dtype=np.float64)/255.}


def patches(arrays, window):
    points=grid(window)
    h,w=arrays['H'].shape[:2]
    source_points=points*np.array([w/640.,h/640.])
    return {arm:sample_bilinear(arrays[arm],points if arm=='L' else source_points) for arm in ARMS},points


def readout(patch, points, initial_center):
    """No GT/category/offset argument. Exactly the fixed265 border contrast rule."""
    need(patch.shape==(16,16,3) and points.shape==(16,16,2), 'readout raster')
    border=np.zeros((16,16),dtype=bool); border[[0,-1],:]=True; border[:,[0,-1]]=True
    need(int(border.sum())==60,'unique outer border')
    background=patch[border].mean(axis=0)
    weight=np.square(patch-background).sum(axis=-1)
    total=float(weight.sum())
    if total==0:
        return np.zeros(2,dtype=np.float64),total
    center=(weight[...,None]*points).sum(axis=(0,1))/total
    return np.clip(center-np.asarray(initial_center),-RHO,RHO),total


def aggregate_gt(errors):
    need(errors.ndim==2 and errors.shape[0]==4 and errors.shape[1]==4, 'four shifts, four arms')
    return errors.mean(axis=0)


def aggregate_images(image_errors):
    need(len(image_errors)>0, 'no participating images')
    values=np.stack([np.asarray(rows).mean(axis=0) for rows in image_errors])
    diff=values[:,1]-values[:,0]
    return dict(participating_images=len(values),arm_order=[*ARMS,'zero'],
        image_equal_weight_mean_error=dict(zip([*ARMS,'zero'],values.mean(axis=0).tolist())),
        H_minus_L_image_mean=float(diff.mean()),H_minus_L_image_median=float(np.median(diff)),
        H_vs_L_image_win_tie_loss=[int((diff<0).sum()),int((diff==0).sum()),int((diff>0).sum())],
        H_minus_Hdegraded_image_mean=float((values[:,1]-values[:,2]).mean()),
        Hdegraded_minus_L_image_mean=float((values[:,2]-values[:,0]).mean()),
        H_minus_zero_image_mean=float((values[:,1]-values[:,3]).mean()),
        L_minus_zero_image_mean=float((values[:,0]-values[:,3]).mean()),
        independent_unit='natural image; four shifts first averaged per GT',automatic_method_GO=None)


def synthetic():
    # Only in-memory arrays/PIL images. No plan, annotation or source image read.
    pts=grid([0,0,16,16]); constant=np.full((16,16,3),.5)
    need(np.array_equal(readout(constant,pts,[8,8])[0],np.zeros(2)), 'constant zero response')
    dot=np.zeros((16,16,3)); dot[7:9,7:9]=1
    need(np.allclose(readout(dot,pts,[8,8])[0],0,atol=1e-12,rtol=0),'symmetric centroid')
    one=np.zeros_like(dot); one[8,10]=1
    need(np.array_equal(readout(one,pts,[8,8])[0],[2.5,.5]),'single point true grid location')
    need(np.array_equal(readout(one,pts,[-10,-10])[0],[4,4]),'delta clipping')
    yy,xx=np.mgrid[0:16,0:16]
    ramp=np.stack([xx,yy,xx+yy],axis=-1).astype(np.float64)
    need(np.array_equal(sample_bilinear(ramp,pts),ramp),'center sampling identity')
    trial=np.array([[[3.25,4.75],[6.6,9.8]]])
    want=np.stack([trial[...,0]-.5,trial[...,1]-.5,trial[...,0]+trial[...,1]-1],axis=-1)
    need(np.allclose(sample_bilinear(ramp,trial),want,rtol=0,atol=1e-12),'bilinear affine exactness')
    need(np.array_equal(sample_bilinear(ramp,np.array([[[-20.,-10.],[90.,30.]]])),
                        np.array([[[0.,0.,0.],[15.,15.,30.]]])),'border replicate')
    base=np.stack([(np.arange(640)[:,None]+np.zeros((640,640)))%256,
                   (np.arange(640)[None,:]+np.zeros((640,640)))%256,np.full((640,640),80)],axis=-1).astype(np.uint8)
    im=Image.fromarray(base,'RGB'); arr=views(im)
    pp,_=patches(arr,[300.125,301.25,311.5,314.75])
    need(np.array_equal(pp['H'],pp['L']) and np.array_equal(pp['H'],pp['Hdegraded']),'640 H=L=HD')
    other=Image.fromarray(base[:127,:191],'RGB')
    pil=other.resize((640,640),Image.Resampling.BILINEAR)
    tv=v2.Resize([640,640],interpolation=InterpolationMode.BILINEAR,antialias=True)(other)
    need(np.array_equal(np.asarray(pil),np.asarray(tv)),'PIL/torchvision resize exact')
    scaled=views(other); pp,xy=patches(scaled,[-3.5,10.2,14.5,28.7])
    need(all(p.shape==(16,16,3) and np.isfinite(p).all() for p in pp.values()),'nonsquare H/L/HD')
    e=np.array([[1,2,3,2],[2,1,2,2],[1,2,3,2],[2,1,2,2]],dtype=float)
    need(np.array_equal(aggregate_gt(e),[1.5,1.5,2.5,2]),'four directions first')
    ag=aggregate_images([np.array([[0,2,1,2]]*8),np.array([[2,0,1,2]])])
    need(ag['H_minus_L_image_mean']==0 and ag['H_vs_L_image_win_tie_loss']==[1,0,1], 'image equal weight')
    need(all(value==0 for value in COUNTS.values()),'synthetic touched real/model sources')
    return dict(constant_zero=True,symmetric_center=True,single_point_grid=True,delta_clip4=True,
        identity_sampling=True,bilinear_affine=True,border_replicate=True,shared_grid16=True,
        resize_PIL_torchvision_bitexact=True,source640_H_L_HD_identical=True,nonsquare_views=True,
        four_directions_then_equal_image_weight=True)


def select_panel():
    need(sha(PLAN)==PLAN_SHA and sha(ANNOTATION)==ANNOTATION_SHA,'plan/annotation identity')
    plan=json.loads(PLAN.read_text(encoding='utf-8-sig')); COUNTS['plan_reads']+=1
    pool=sorted(plan['source_pools']['development'])
    need(pool[:16]==IMAGE_IDS and len(pool)==3235 and not set(pool)&set(plan['source_pools']['heldout'])
         and plan['annotation_sha256']==ANNOTATION_SHA,'fixed development pool')
    ann=json.loads(ANNOTATION.read_text(encoding='utf-8-sig')); COUNTS['annotation_reads']+=1
    images={int(row['id']):row for row in ann['images'] if row['id'] in IMAGE_IDS}
    need(set(images)==set(IMAGE_IDS),'all16 annotation image identities')
    eligible=[]
    for row in ann['annotations']:
        iid=int(row['image_id'])
        if iid not in images or row.get('iscrowd',0) or row.get('ignore',0):
            continue
        bbox=np.asarray(row['bbox'],dtype=np.float64); info=images[iid]
        need(bbox.shape==(4,) and np.isfinite(bbox).all(),'finite COCO boxes')
        need(0<=int(row['category_id'])<10,'class ID binding')
        if (bbox[2:]<=0).any():
            continue
        box640=bbox*np.array([640/info['width'],640/info['height']]*2)
        if min(box640[2:])<4:
            eligible.append(dict(annotation_id=int(row['id']),image_id=iid,category_id=int(row['category_id']),
                xywh_original=bbox.tolist(),xywh640=box640.tolist(),
                GT_outside_image=bool((bbox[:2]<0).any() or bbox[0]+bbox[2]>info['width'] or bbox[1]+bbox[3]>info['height'])))
    need(len({row['annotation_id'] for row in eligible})==len(eligible),'unique eligible annotation IDs')
    selected=sorted(eligible,key=lambda row:row['annotation_id'])[:64]
    panel=dict(fixed_image_ids=IMAGE_IDS,eligible_GT_in_fixed16=len(eligible),selected_GT=len(selected),
        maximum_GT=64,selection='fixed16 image IDs; ascending annotation_id; no pixel/outcome selection',
        images=images,selected=selected,common_to_BASE24_seeds=[1,2],weight_seed_pairing=False,
        scope='controlled natural development panel; not BASE24 failures or augmentation event reconstruction')
    del ann; gc.collect(); guard()
    return panel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--synthetic-only',action='store_true')
    modes.add_argument('--real-images',action='store_true')
    args=parser.parse_args(); started=time.monotonic(); resources=runtime()
    sources,before=source_manifest(); checks=synthetic()
    if args.synthetic_only:
        target=ROOT/'279_synthetic.json'
        need(not target.exists(),'synthetic result already exists; do not overwrite')
        need(before=={key:sha(path) for key,path in sources.items()},'synthetic sources changed')
        result=dict(state='synthetic_passed_real_images_not_run',tests=checks,counts=COUNTS,resources=resources,
            source_paths={key:str(path) for key,path in sources.items()},source_sha256=before,seconds=time.monotonic()-started)
        save(target,result); print(json.dumps(result,ensure_ascii=False),flush=True); return
    out=ROOT/'279_natural_development_HL_pixel_probe'
    need(not out.exists(),'new279 real output only'); out.mkdir(exist_ok=False)
    stage='metadata'; image_hashes={}; artifacts={}; summaries=[]; errors_for_pool=[]
    try:
        panel=select_panel()
        sources.update(plan=PLAN,annotation=ANNOTATION); before.update(plan=PLAN_SHA,annotation=ANNOTATION_SHA)
        image_paths={}
        for iid in IMAGE_IDS:
            path=(DATA/'images'/panel['images'][iid]['file_name']).resolve()
            need(path.parent==(DATA/'images').resolve() and path.is_file(),'image path must stay inside source images')
            image_paths[iid]=path; image_hashes[str(iid)]=sha(path); COUNTS['real_image_hash_calls']+=1
        save(out/'00_manifest.json',dict(panel=panel,resources=resources,synthetic_self_checks=checks,
            source_paths={key:str(path) for key,path in sources.items()},source_sha256=before,
            image_paths={str(iid):str(path) for iid,path in image_paths.items()},image_sha256_before=image_hashes))
        for iid in IMAGE_IDS:
            guard(); stage=f'image_{iid}'; info=panel['images'][iid]
            rows=[row for row in panel['selected'] if row['image_id']==iid]
            if not rows:
                summaries.append(dict(image_id=iid,selected_GT=0,decoded=False)); continue
            need(sha(image_paths[iid])==image_hashes[str(iid)],'image changed before decode')
            COUNTS['real_image_hash_calls']+=1
            with Image.open(image_paths[iid]) as handle:
                need(handle.size==(info['width'],info['height']),'decoded dimensions differ from COCO')
                rgb=handle.convert('RGB'); rgb.load()
            COUNTS['real_image_decodes']+=1
            image_views=views(rgb); rgb.close()
            image_gt_errors=[]; records=[]; saved={}
            for row in rows:
                bbox=np.array(row['xywh640']); center=bbox[:2]+bbox[2:]/2
                trial_errors=[]
                for shift_id,shift in enumerate(SHIFTS):
                    initial=center+shift; true_delta=-shift
                    window=np.r_[initial-bbox[2:]/2-RHO,initial+bbox[2:]/2+RHO]
                    crop,points=patches(image_views,window)
                    estimates={}; energies={}
                    for arm in ARMS:
                        estimates[arm],energies[arm]=readout(crop[arm],points,initial)
                    errs=np.array([np.max(np.abs(estimates[arm]-true_delta)) for arm in ARMS]+[np.max(np.abs(true_delta))])
                    trial_errors.append(errs)
                    tag=f'gt{row["annotation_id"]}_s{shift_id}'
                    for arm in ARMS:
                        saved[tag+'_'+arm]=crop[arm]
                    saved[tag+'_grid_input_px']=points
                    records.append(dict(**row,shift_id=shift_id,initial_center640=initial.tolist(),true_delta_px=true_delta.tolist(),
                        window640=window.tolist(),window_outside_input=bool((window[:2]<0).any() or (window[2:]>640).any()),
                        window_input_area_fraction=float(np.prod(np.maximum(np.minimum(window[2:],640)-np.maximum(window[:2],0),0))/np.prod(window[2:]-window[:2])),
                        estimate_px={arm:estimates[arm].tolist() for arm in ARMS},contrast_mass=energies,
                        Linf_error_px=dict(zip([*ARMS,'zero'],errs.tolist())),
                        H_minus_L_MSE=float(np.square(crop['H']-crop['L']).mean()),
                        Hdegraded_minus_L_MSE=float(np.square(crop['Hdegraded']-crop['L']).mean())))
                image_gt_errors.append(aggregate_gt(np.stack(trial_errors)))
            report=dict(image_id=iid,file_name=info['file_name'],original_WH=[info['width'],info['height']],
                source_sha256=image_hashes[str(iid)],selected_GT=len(rows),controlled_trials=4*len(rows),
                decoded_once=True,GT_equal_weight_mean_error=dict(zip([*ARMS,'zero'],np.stack(image_gt_errors).mean(axis=0).tolist())),records=records)
            stem=f'image_{iid:04d}'; save(out/(stem+'.json'),report); np.savez(out/(stem+'_patches.npz'),**saved)
            artifacts[stem+'.json']=sha(out/(stem+'.json')); artifacts[stem+'_patches.npz']=sha(out/(stem+'_patches.npz'))
            summaries.append({key:value for key,value in report.items() if key!='records'})
            errors_for_pool.append(np.stack(image_gt_errors)); del image_views,saved,crop; gc.collect(); print(stage,'complete',flush=True)
        after={key:sha(path) for key,path in sources.items()}
        image_after={str(iid):sha(path) for iid,path in image_paths.items()}
        COUNTS['real_image_hash_calls']+=len(image_paths)
        need(before==after and image_hashes==image_after,'source/image changed')
        need(COUNTS['real_image_decodes']==len(errors_for_pool)<=16,'decode each participating image once')
        need(sum(row.get('selected_GT',0) for row in summaries)==panel['selected_GT']<=64,'fixed target budget')
        need(all(COUNTS[key]==0 for key in ('checkpoint_loads','detector_builds','model_forwards','model_backwards','optimizer_steps','heldout_image_reads')),'no model/training/heldout')
        result=dict(state='complete' if errors_for_pool else 'complete_empty_fixed_panel',
            main=aggregate_images(errors_for_pool) if errors_for_pool else None,images=summaries,
            selected_GT=panel['selected_GT'],controlled_trials=panel['selected_GT']*4,
            counts=COUNTS,resources=resources,synthetic_self_checks=checks,source_sha256_before=before,source_sha256_after=after,
            image_sha256_before=image_hashes,image_sha256_after=image_after,sources_unchanged=True,
            artifact_sha256=artifacts,seconds=time.monotonic()-started,training_authorized=False,AP=False,
            BASE24_real_failure_validation=False,seed_independent_replications=False,
            limits='fixed hand-designed readout and GT-generated centers/sizes; not deployable evidence or general rejection of learnable phi')
        save(out/'01_results.json',result)
        save(out/'02_output_sha256.json',{path.name:sha(path) for path in out.iterdir() if path.is_file()})
        print(json.dumps(dict(state=result['state'],main=result['main'],counts=COUNTS),ensure_ascii=False),flush=True)
    except BaseException:
        save(out/'FAILED.json',dict(state='error_no_method_conclusion',stage=stage,counts=COUNTS,
            traceback=traceback.format_exc(),seconds=time.monotonic()-started)); raise


if __name__=='__main__':
    main()
