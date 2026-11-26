"""Development rebuild: approved builder/exact M0, immutable 340 raw only."""
import importlib,json,time,types,os,uuid,argparse,traceback
C=importlib.import_module('343_context')
import torch
B=importlib.import_module('342_build_val_v3')
TRAIN=C.Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train')
# Only the image root changes in the approved builder, in memory; old files stay read-only.
B.C=types.SimpleNamespace(**vars(B.C));B.C.VAL=TRAIN
G=B.G
torch.set_num_threads(2)
MUTABLE={'m0_gt_explained','m0_cand_owner','m0_coverage','U_W','n_U_W','sha256'}

def metadata():
    ids,idh=C.development_ids();allowed=set(ids)
    idx=json.loads((C.SPEC/'327_训练实施输入索引.json').read_text(encoding='utf-8'))
    ar=next(a for a in idx['annotations'] if a['split']=='train')
    path=C.Path(ar['path']);ah=C.sha(path);assert ah==ar['expected_SHA256_from_existing_audit']
    data=json.loads(path.read_text(encoding='utf-8'))
    ims={i['id']:i for i in data['images'] if i['id'] in allowed};by={}
    for a in data['annotations']:
        if a['image_id'] in allowed:by.setdefault(a['image_id'],[]).append(a)
    assert set(ims)==allowed
    return ids,ims,by,ah,idx

def entry_path(seed,iid):return C.HERE/f'343_cache_development_seed{seed}'/f'343_{iid:06d}.pt'

def compare_old(old,e):
    equal={k:torch.equal(v,e[k]) if torch.is_tensor(v) else v==e[k] for k,v in old.items() if k not in MUTABLE}
    assert all(equal.values()),{k:v for k,v in equal.items() if not v}
    return equal

def run(seed,tag,limit=None):
    auth=json.loads((C.HERE/'343_追加放行与计数口径.json').read_text(encoding='utf-8'))
    assert auth['source_chain_accepted'] is True
    ids,ims,by,ah,idx=metadata()
    pre=json.loads((C.HERE/'343_result_3b来源预检.json').read_text(encoding='utf-8'))
    seals={(r['seed'],r['image_id']):r['current_340_file_sha256'] for r in pre['entries']}
    cp=idx['checkpoints'][seed-1];assert C.sha(cp['path'])==cp['expected_SHA256_from_288']
    folder=entry_path(seed,ids[0]).parent;folder.mkdir(exist_ok=True)
    count={'old_cache_loads':0,'new_cache_loads':0,'verified_skips':0,'cache_writes':0,'M0_solves':0,
           'image_decodes':0,'image_SHA_reads':0,'global_model_loads':0,'global_forwards':0,'head_forwards':0,'optimizer_steps':0}
    rows=[];start=time.perf_counter();status='已生成，待独立验收';error=None
    try:
        for n,iid in enumerate(ids,1):
            source=C.WORK/f'340_cache_seed{seed}'/f'{iid:06d}.pt';sealed=seals[seed,iid]
            assert C.sha(source)==sealed
            p=entry_path(seed,iid)
            if p.exists():
                e=torch.load(p,map_location='cpu',weights_only=True);count['new_cache_loads']+=1
                assert B.payload_sha(e)==e['stage3_payload_sha256'] and e['raw_cache_file_sha256']==sealed
                assert e['annotation_sha256']==ah and e['seed']==seed and e['image_id']==iid
                count['verified_skips']+=1
                continue
            old=torch.load(source,map_location='cpu',weights_only=True);count['old_cache_loads']+=1
            im=ims[iid];assert old['file_name']==im['file_name'] and old['orig_wh']==[im['width'],im['height']]
            img=TRAIN/'images'/im['file_name'];image_sha=C.sha(img);count['image_SHA_reads']+=1
            raw={'final300_boxes':old['final300_boxes'],'final300_logits':old['final300_logits'],
                'seed':seed,'source_image_sha256':image_sha,'checkpoint_sha256':cp['expected_SHA256_from_288']}
            e=B.build_entry(raw,im,by.get(iid,[]),ah,source);count['M0_solves']+=1;count['image_decodes']+=1
            compare_old(old,e)
            tmp=folder/f'343_{iid:06d}_incomplete_{uuid.uuid4().hex}.pt'
            with open(tmp,'xb') as f:torch.save(e,f)
            os.rename(tmp,p);count['cache_writes']+=1
            rows.append({'image_id':iid,'path':str(p),'file_sha256':C.sha(p),'old_file_sha256':sealed})
            if count['cache_writes']%64==0 or n==3235:
                print(f'development seed={seed} {n}/3235 new={count["cache_writes"]} skips={count["verified_skips"]} elapsed={time.perf_counter()-start:.1f}s',flush=True)
            if limit is not None and count['cache_writes']>=limit:break
    except Exception as ex:
        status='实现失败';error={'exception':repr(ex),'traceback':traceback.format_exc()};print(error['traceback'],flush=True)
    C.write(f'343_result_3b构建_seed{seed}_{tag}.json',{'status':status,'seed':seed,'tag':tag,'rows':rows,
        'new_entries_on_disk':len(list(folder.glob('343_*.pt'))),'counts':count,'已执行':count,
        'elapsed_seconds':time.perf_counter()-start,'error':error,'未执行':['真实训练','GPU','新global前向','阶段5'],
        '未核':['全量独立验收'],'source_sha256':C.sources()})
    print(json.dumps({'seed':seed,'tag':tag,'status':status,'counts':count},ensure_ascii=False),flush=True)
    if error:raise RuntimeError(error['exception'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=(1,2),required=True);p.add_argument('--tag',required=True);p.add_argument('--limit',type=int)
    a=p.parse_args();run(a.seed,a.tag,a.limit)
