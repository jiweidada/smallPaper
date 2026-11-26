"""CPU raw-model val548 predictions only. No M0, labels, crop or head.

Complete entries are immutable and resumed only after source/payload validation.
Targets remain pending the controller's decision about the inherited M0 defect.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import sys,json,time,hashlib,uuid,argparse
from pathlib import Path
sys.dont_write_bytecode=True
import torch,numpy as np
from PIL import Image
torch.set_num_threads(2)
HERE=Path(__file__).resolve().parent
WORK=HERE.parent
SPEC=WORK.parents[1]/'持续寻找研究主线_20261020_v2'
VAL=Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val')
sys.path.append(str(WORK))
import stage1_cache_v3 as S

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def numeric_sha(e):
    h=hashlib.sha256()
    for name in ('final300_boxes','final300_logits'):
        t=e[name]
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.contiguous().numpy().tobytes())
    for name in ('image_id','orig_wh','file_name','source_image_sha256','checkpoint_sha256'):
        h.update(json.dumps(e[name],sort_keys=True).encode())
    return h.hexdigest()

def state_sha(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items():
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def write(name,obj):
    with open(HERE/name,'x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def run(seed,stop_after_new,tag):
    counts={'annotation_sha_checks':0,'annotation_parses':0,'checkpoint_sha_checks':0,
            'model_loads':0,'global_forward_calls':0,'image_decodes':0,'image_sha_checks':0,
            'cache_loads':0,'cache_writes':0,'existing_validated_skips':0,'optimizer_steps':0}
    t0=time.perf_counter()
    idx=json.loads((SPEC/'327_训练实施输入索引.json').read_text(encoding='utf-8'))
    ap=VAL/'annotations'/'val.json';ah=sha(ap);counts['annotation_sha_checks']+=1
    assert ah==next(a['expected_SHA256_from_existing_audit'] for a in idx['annotations'] if a['split']=='val')
    ann=json.loads(ap.read_text(encoding='utf-8'));counts['annotation_parses']+=1
    ims=ann['images'];assert len(ims)==548 and len({i['id'] for i in ims})==548
    cp=next(c for c in idx['checkpoints'] if c['seed']==seed)
    ch=sha(cp['path']);counts['checkpoint_sha_checks']+=1
    assert ch==cp['expected_SHA256_from_288']
    folder=HERE/f'342_raw_val_seed{seed}'
    folder.mkdir(exist_ok=True)
    model=None;before=None;rows=[];subset=[]
    try:
        for n,im in enumerate(ims,1):
            iid=im['id'];src=VAL/'images'/im['file_name']
            out=folder/f'342_{iid:06d}.pt'
            ih=sha(src);counts['image_sha_checks']+=1
            if out.exists():
                e=torch.load(out,map_location='cpu',weights_only=True);counts['cache_loads']+=1
                assert e['image_id']==iid and e['checkpoint_sha256']==ch and e['source_image_sha256']==ih
                assert numeric_sha(e)==e['raw_payload_sha256'],f'corrupt complete cache: {out}'
                counts['existing_validated_skips']+=1
                rows.append({'image_id':iid,'status':'verified_skip','path':str(out),'sha256':sha(out)})
                continue
            if model is None:
                model=S.load_real_model(cp['path']);counts['model_loads']+=1
                assert all(p.device.type=='cpu' and not p.requires_grad for p in model.parameters())
                before=state_sha(model)
            with Image.open(src) as f:
                pil=f.convert('RGB');counts['image_decodes']+=1
            assert list(pil.size)==[im['width'],im['height']]
            with torch.no_grad():
                pb,logits=S.global_forward_640(model,pil);counts['global_forward_calls']+=1
            assert pb.shape==(300,4) and logits.shape==(300,10)
            assert torch.isfinite(pb).all() and torch.isfinite(logits).all()
            assert not pb.requires_grad and not logits.requires_grad
            e={'image_id':iid,'orig_wh':[im['width'],im['height']],'file_name':im['file_name'],
               'final300_boxes':pb.contiguous(),'final300_logits':logits.contiguous(),
               'source_image_sha256':ih,'checkpoint_sha256':ch,'annotation_sha256':ah,
               'seed':seed,'stage':'342 raw global predictions; no targets/M0'}
            e['raw_payload_sha256']=numeric_sha(e)
            tmp=folder/f'342_{iid:06d}_incomplete_{uuid.uuid4().hex}.pt'
            with open(tmp,'xb') as f:torch.save(e,f)
            os.rename(tmp,out) # On Windows this fails if destination exists.
            counts['cache_writes']+=1
            rows.append({'image_id':iid,'status':'created','path':str(out),'sha256':sha(out)})
            old=WORK/'341_cache_val_seed1'/f'{iid:06d}.pt'
            if seed==1 and old.exists():
                prior=torch.load(old,map_location='cpu',weights_only=True);counts['cache_loads']+=1
                same={k:torch.equal(e[k],prior[k]) for k in ('final300_boxes','final300_logits')}
                subset.append({'image_id':iid,'fields_bit_equal':same,'all_equal':all(same.values())})
            if counts['cache_writes']%32==0 or n==548:
                print(f'seed={seed} completed_position={n}/548 new={counts["cache_writes"]} skipped={counts["existing_validated_skips"]} elapsed={time.perf_counter()-t0:.1f}s',flush=True)
            if stop_after_new is not None and counts['cache_writes']>=stop_after_new:break
        if model is not None:
            after=state_sha(model)
            assert after==before
            assert all(p.grad is None for p in model.parameters())
        else:after=before
        imported={}
        for mod in list(sys.modules.values()):
            path=getattr(mod,'__file__',None)
            if path and path.endswith('.py') and str(WORK.parents[1]) in path:
                p=Path(path)
                if p.is_file():imported[str(p)]=sha(p)
        imported[str(Path(__file__))]=sha(__file__)
        report={'status':'已验证（仅冻结预测子缓存；非完整阶段3缓存）','seed':seed,'tag':tag,
                'intentionally_stopped_after_new':stop_after_new,'annotation_sha256':ah,'checkpoint_sha256':ch,
                'output_folder':str(folder),'existing_complete_entries':len(list(folder.glob('342_*.pt'))),
                'elapsed_seconds':time.perf_counter()-t0,'已执行':counts,
                'raw_global_state_sha_before':before,'raw_global_state_sha_after':after,
                'raw_global_all_grad_none':True if model is not None else '未执行模型调用',
                'old341_prediction_comparison':subset,'rows':rows,'source_sha256':imported,
                '未执行':['M0/完整目标缓存（待主控裁定）','训练','GPU','SSH','队列','heldout3236读取','阶段4至5'],
                '未核':['完整阶段3验收'],'待主控':['M0跨类owner及与341逐位对账的冲突']}
        write(f'342_raw_seed{seed}_{tag}_result.json',report)
        print(json.dumps({'seed':seed,'tag':tag,'complete':report['existing_complete_entries'],'counts':counts}),flush=True)
    except Exception as ex:
        import traceback
        write(f'342_raw_seed{seed}_{tag}_failure.json',{'status':'实现失败','error':repr(ex),
              'traceback':traceback.format_exc(),'已执行':counts,'未执行':['真实训练','阶段4至5'],'未核':['raw缓存完整性']})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[1,2],required=True)
    p.add_argument('--stop-after-new',type=int);p.add_argument('--tag',required=True)
    args=p.parse_args();run(args.seed,args.stop_after_new,args.tag)
