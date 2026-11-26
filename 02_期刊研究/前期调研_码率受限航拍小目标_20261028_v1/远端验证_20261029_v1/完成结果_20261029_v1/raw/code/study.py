"""Pinned YOLO26s baseline experiment. The only arm difference is native-resolution JPEG."""
from pathlib import Path
from copy import copy
from io import BytesIO
import hashlib,json,math,os,random,time
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.torch_utils import unwrap_model
from ultralytics.utils import SETTINGS,events,YAML
import ultralytics.data.utils as dataset_utils

# Keep the isolated run offline. Plots are disabled, so the dataset font check is unnecessary.
SETTINGS.update({'sync':False})
events.enabled=False
dataset_utils.check_font=lambda *args,**kwargs: None

ROOT=Path('/root/c10_region_restore_20261029_v1')
class PreflightComplete(Exception):pass

def state_hash(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        h.update(key.encode());h.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def json_write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str));tmp.replace(path)

def finite_state(model):
    return all(torch.isfinite(v).all().item() for v in model.state_dict().values() if v.is_floating_point())

class NativeJPEGDataset(YOLODataset):
    def configure(self,arm,mode,run_dir):
        self.study_arm=arm;self.study_mode=mode;self.study_run_dir=Path(run_dir)
        self.study_rng={};self.study_reads={}
        # No optional Albumentations transform, independent of whether the optional package is installed.
        def visit(t,seen):
            if id(t) in seen:return
            seen.add(id(t))
            if type(t).__name__=='Albumentations':t.transform=None
            if mode!='train' and type(t).__name__=='LetterBox':
                t.scaleup=True;t.auto=False;t.center=True;t.scale_fill=False
            for child in getattr(t,'transforms',[]):visit(child,seen)
            child=getattr(t,'pre_transform',None)
            if child is not None:visit(child,seen)
        visit(self.transforms,set())
        return self

    def load_image(self,i,rect_mode=True,resize_short=False):
        # Always read the source. A mosaic-buffer hit must not bypass the additional-JPEG intervention.
        was_cached=self.ims[i] is not None
        f=self.im_files[i]
        with Image.open(f) as image:rgb=image.convert('RGB')
        w0,h0=rgb.size
        worker=torch.utils.data.get_worker_info();wid=worker.id if worker else -1
        n=self.study_reads.get(wid,0);self.study_reads[wid]=n+1
        quality=None;apply=False;payload_sha=None
        if self.study_mode=='train':
            if wid not in self.study_rng:self.study_rng[wid]=np.random.default_rng(np.random.SeedSequence([0,20261029,wid+1]))
            rng=self.study_rng[wid]
            selected=rng.random()<.5
            proposed=int(rng.choice([15,20,35,50,70])) if selected else None
            apply=self.study_arm=='DA' and selected
            if apply:
                before_py=random.getstate() if n<8 else None
                before_np=np.random.get_state() if n<8 else None
                buf=BytesIO();rgb.save(buf,format='JPEG',quality=proposed,subsampling=2,optimize=True,progressive=False)
                payload=buf.getvalue();quality=proposed
                with Image.open(BytesIO(payload)) as image:rgb=image.convert('RGB')
                if n<8:
                    assert random.getstate()==before_py
                    after=np.random.get_state()
                    assert before_np[0]==after[0] and np.array_equal(before_np[1],after[1]) and before_np[2:]==after[2:]
                    payload_sha=hashlib.sha256(payload).hexdigest()
        im=np.ascontiguousarray(np.asarray(rgb)[:,:,::-1])
        if n<64:
            event={'worker':wid,'read':n,'source':Path(f).name,'native_hw':[h0,w0],'jpeg_applied':apply,'quality':quality,
                   'placement':'native resolution before any resize/mosaic','payload_sha256':payload_sha}
            path=self.study_run_dir/('source_reads_'+self.study_mode+'_worker'+str(wid)+'.jsonl')
            with path.open('a') as stream:stream.write(json.dumps(event)+'\n')
        if self.study_mode!='train':
            # Let the single fixed LetterBox transform operate directly on native pixels.
            return im,(h0,w0),(h0,w0)
        shape=tuple(self.imgsz) if isinstance(self.imgsz,(tuple,list)) else (self.imgsz,self.imgsz)
        imgsz=max(shape)
        if rect_mode:
            ratio=imgsz/(min(h0,w0) if resize_short else max(h0,w0))
            if ratio!=1:
                if resize_short:
                    w,h=(math.ceil(w0*ratio),imgsz) if h0<w0 else (imgsz,math.ceil(h0*ratio))
                else:w,h=min(math.ceil(w0*ratio),imgsz),min(math.ceil(h0*ratio),imgsz)
                im=cv2.resize(im,(w,h),interpolation=cv2.INTER_LINEAR)
        elif (h0,w0)!=shape:im=cv2.resize(im,shape[::-1],interpolation=cv2.INTER_LINEAR)
        if self.augment and self.cache!='ram':
            self.ims[i],self.im_hw0[i],self.im_hw[i]=im,(h0,w0),im.shape[:2]
            if not was_cached:
                self.buffer.append(i)
                if 1<len(self.buffer)>=self.max_buffer_length:
                    j=self.buffer.pop(0)
                    self.ims[j],self.im_hw0[j],self.im_hw[j]=None,None,None
        return im,(h0,w0),im.shape[:2]

class FixedReferenceValidator(DetectionValidator):
    def __call__(self,trainer=None,model=None):
        assert trainer is not None,'Final scientific evaluation uses the separate unfused fixed-checkpoint evaluator'
        chosen=trainer.ema.ema if trainer.ema else unwrap_model(trainer.model)
        head=chosen.model[-1];old=head.end2end
        if getattr(chosen,'criterion',None) is None:
            chosen.criterion=chosen.init_criterion()  # initialize native dual-head loss before selecting inference branch
        head.end2end=False
        try:return super().__call__(trainer=trainer,model=model)
        finally:head.end2end=old

class StudyTrainer(DetectionTrainer):
    ARM='D0';PREFLIGHT=False
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.args.augmentations=[]
        self.real_updates=0;self.attempted_updates=0;self.amp_skips=0;self.batch_count=0;self.batch_contracts=[]
        self.study_started=time.monotonic()
        self.callbacks['on_pretrain_routine_end'].append(self.audit_ready)
        self.callbacks['on_train_epoch_start'].append(self.audit_epoch_start)
        self.callbacks['on_train_batch_end'].append(self.audit_batch_end)
        self.callbacks['on_fit_epoch_end'].append(self.audit_epoch_end)

    def build_dataset(self,img_path,mode='train',batch=None):
        cfg=copy(self.args);cfg.rect=False;cfg.augmentations=[]
        dataset=build_yolo_dataset(cfg,img_path,batch,self.data,mode=mode,rect=False,stride=32)
        dataset.__class__=NativeJPEGDataset
        return dataset.configure(self.ARM,mode,self.save_dir)

    def get_validator(self):
        return FixedReferenceValidator(self.test_loader,save_dir=self.save_dir,args=copy(self.args),_callbacks=self.callbacks)

    def _build_train_pipeline(self):
        super()._build_train_pipeline()
        self.args.max_det=500
        unwrap_model(self.model).set_head_attr(max_det=500)

    def audit_ready(self,trainer):
        assert self.amp is True,'AMP failed preflight; do not silently change the frozen recipe'
        model=unwrap_model(self.model)
        assert model.model[-1].nc==10 and model.model[-1].cv2 is not None and model.model[-1].one2one_cv2 is not None
        expected=json.loads((ROOT/'receipts/common_initialization.json').read_text())['state_sha256']
        actual=state_hash(model)
        assert actual==expected,('Initial state mismatch',actual,expected)
        self.optimizer.register_step_post_hook(self.actual_step)
        self.initial_state=actual
        YAML.save(self.save_dir/'args.yaml',vars(self.args))
        json_write(self.save_dir/'start_receipt.json',{'status':'STARTED','arm':self.ARM,'preflight':self.PREFLIGHT,'pid':os.getpid(),
                   'initial_state_sha256':actual,'common_checkpoint_sha256':json.loads((ROOT/'receipts/common_initialization.json').read_text())['checkpoint_sha256'],
                   'resolved_args':vars(self.args),'train_count':len(self.train_loader.dataset),'dev_count':len(self.test_loader.dataset),
                   'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'native_stal_and_progressive_loss':True,
                   'jpeg_rng_independent':True,'fresh_native_decode_both_arms':True,'optional_albumentations':[]})
        torch.cuda.reset_peak_memory_stats()

    def actual_step(self,optimizer,args,kwargs):self.real_updates+=1

    def optimizer_step(self):
        old=self.real_updates;self.attempted_updates+=1
        super().optimizer_step()
        self.amp_skips+=int(self.real_updates==old)

    def preprocess_batch(self,batch):
        batch=super().preprocess_batch(batch)
        if len(self.batch_contracts)<8:
            h=hashlib.sha256()
            for key in ('batch_idx','cls','bboxes'):
                tensor=batch[key].detach().cpu().contiguous();h.update(key.encode());h.update(tensor.numpy().tobytes())
            h.update('\n'.join(Path(p).name for p in batch['im_file']).encode())
            self.batch_contracts.append({'batch':len(self.batch_contracts),'geometry_sha256':h.hexdigest(),
                'input_shape':list(batch['img'].shape),'gt_count':int(len(batch['cls'])),
                'pixel_sha256':hashlib.sha256(batch['img'].detach().cpu().contiguous().numpy().tobytes()).hexdigest()})
        return batch

    def audit_epoch_start(self,trainer):
        # The pinned trainer otherwise auto-reduces the batch on first-epoch OOM. It must fail instead.
        self._oom_retries=100

    def audit_batch_end(self,trainer):
        self.batch_count+=1
        if not torch.isfinite(self.loss).all().item():raise RuntimeError('NONFINITE_LOSS: no automatic recovery/restart')
        if self.batch_count%50==0 or self.PREFLIGHT:
            json_write(self.save_dir/'live_status.json',self.status())
        if self.PREFLIGHT and self.batch_count==8:
            assert finite_state(unwrap_model(self.model)) and finite_state(self.ema.ema)
            validation_metrics=self.validator(self)
            assert all(np.isfinite(float(v)) for v in validation_metrics.values()),'Nonfinite preflight validation'
            json_write(self.save_dir/'preflight_result.json',dict(self.status(),status='PASS',batch_contracts=self.batch_contracts,
                model_state_finite=True,ema_state_finite=True,reference_validator_checked=True,
                reference_validator_health_metrics=validation_metrics,reset_required_before_formal=True))
            raise PreflightComplete('Eight disposable preflight batches completed as planned')

    def status(self):
        return {'arm':self.ARM,'preflight':self.PREFLIGHT,'epoch':self.epoch+1,'epoch_budget':self.epochs,
                'batches':self.batch_count,'actual_optimizer_updates':self.real_updates,'attempted_optimizer_updates':self.attempted_updates,
                'amp_skips':self.amp_skips,'ema_updates':getattr(self.ema,'updates',None),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'elapsed_seconds':time.monotonic()-self.study_started,'pid':os.getpid(),'updated_unix':time.time()}

    def audit_epoch_end(self,trainer):
        json_write(self.save_dir/'live_status.json',self.status())
        with (self.save_dir/'epoch_audit.jsonl').open('a') as f:f.write(json.dumps(self.status())+'\n')
        json_write(self.save_dir/'first_batch_contracts.json',self.batch_contracts)

    def _handle_nan_recovery(self,epoch):
        if self.loss is not None and not torch.isfinite(self.loss).all().item():raise RuntimeError('NONFINITE_LOSS: native recovery disabled')
        if self.fitness is not None and not np.isfinite(self.fitness):raise RuntimeError('NONFINITE_FITNESS: native recovery disabled')
        return False

    def save_model(self):
        for model in (unwrap_model(self.model),self.ema.ema):
            assert finite_state(model),'Nonfinite state: refusing native checkpoint repair/clamping'
            assert all(v.abs().max().item()<=65504 for v in model.state_dict().values() if v.is_floating_point()),'FP16 serialization overflow'
        return super().save_model()

    def final_eval(self):
        # Do not evaluate a val-selected best checkpoint or fuse away one2many.
        assert self.epoch+1==100 and self.epochs==100
        ckpt=torch.load(self.last,map_location='cpu',weights_only=False)
        assert ckpt['epoch']==99 and ckpt['ema'].model[-1].cv2 is not None
        with Path(self.last).open('rb') as f:digest=hashlib.sha256(f.read()).hexdigest()
        json_write(self.save_dir/'completed.json',dict(self.status(),status='COMPLETED_FIXED_EPOCH100',
            primary_checkpoint=str(self.last),primary_checkpoint_sha256=digest,checkpoint_selected_by='fixed epoch100 EMA',
            native_reference_health_metrics=self.metrics,scientific_crowd_aware_evaluation_pending=True))

def training_args(arm,batch,preflight=False):
    return dict(data=str(ROOT/'dataset/data.yaml'),epochs=100,seed=0,imgsz=640,batch=batch,nbs=64,
        workers=8,optimizer='AdamW',lr0=.000714,momentum=.9,lrf=.01,weight_decay=.0005,
        warmup_epochs=3.,warmup_momentum=.8,warmup_bias_lr=0.,cos_lr=False,hsv_h=.015,hsv_s=.7,hsv_v=.4,
        translate=.1,scale=.5,fliplr=.5,flipud=0.,degrees=0.,shear=0.,perspective=0.,mosaic=1.,close_mosaic=10,
        mixup=0.,cutmix=0.,copy_paste=0.,rect=False,multi_scale=0.,amp=True,deterministic=True,cache=False,
        compile=False,resume=False,patience=0,device=0,plots=False,save=not preflight,save_period=-1,
        project=str(ROOT/'results'),name=('preflight_'+arm+'_b'+str(batch)) if preflight else arm,
        exist_ok=False,val=not preflight,max_det=500,conf=.001,iou=.7,verbose=False)
