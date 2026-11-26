"""Stage-3 complete val cache from immutable CPU raw predictions.

Fresh exact M0 follows 296/317; old caches and all historical source are untouched.
No model/optimizer is constructed by this builder.
"""
import importlib,time,json,argparse,uuid,os,hashlib,difflib
C=importlib.import_module('342_stage3_context')
import torch,numpy as np
from PIL import Image
import geom342 as G
M=importlib.import_module('342_m0_exact')
torch.set_num_threads(2)

def payload_sha(e):
    h=hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            h.update(b'tensor');h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
            h.update(v.contiguous().numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v):
                if k=='stage3_payload_sha256':continue
                h.update(k.encode());add(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(str(len(v)).encode())
            for x in v:add(x)
        else:h.update(json.dumps(v,sort_keys=True).encode())
    add(e);return h.hexdigest()

def entry_path(seed,iid):return C.HERE/f'342_cache_val_v2_seed{seed}'/f'342_{iid:06d}.pt'

def build_entry(raw,im,items,annotation_sha,raw_path):
    iid=im['id'];ow,oh=im['width'],im['height']
    pb=raw['final300_boxes'];logits=raw['final300_logits']
    p300=G.norm_cxcywh_to_640_xyxy(pb)
    scores,flat=logits.sigmoid().flatten().topk(300)
    labels=flat%10;queries=flat//10;keep=scores>.001
    scores=scores[keep];labels=labels[keep];queries=queries[keep];flat=flat[keep]
    full_boxes=p300[queries];full_logits=logits[queries]
    rc,cnt=G.select_window(p300);w=G.window_xyxy(rc)
    xywh=torch.tensor([a['bbox'] for a in items],dtype=torch.float32).reshape(-1,4)
    gt=G.orig_xywh_to_640_xyxy(xywh,ow,oh)
    gl=torch.tensor([a['category_id'] for a in items],dtype=torch.long)
    cb=G.full_plane_cxcywh(full_boxes);gb=G.full_plane_cxcywh(gt)
    cost,iou,legal=G.m0_matrices(cb,full_logits,gb,gl,labels)
    sol=M.exact_assignment(cost.numpy(),legal.numpy(),flat.tolist())
    # Persist large certificate integers as decimal strings: torch 2.5's
    # weights_only unpickler does not implement pickle LONG4. Math stays exact.
    sol=dict(sol)
    for field in ('cost_integer_sum','owner_vector_lex','edge_cost_shift'):
        sol[field]=str(sol[field])
    sol['dual_potentials']=[str(x) for x in sol['dual_potentials']]
    owner=torch.tensor(sol['cand_owner'],dtype=torch.long)
    explained=torch.zeros(len(gt),dtype=torch.bool)
    if (owner>=0).any():explained[owner[owner>=0]]=True
    gw=G.complete_mask(gt,w);uw=gw&~explained
    kg=G.kg_window(gt,gw,w)
    src=C.VAL/'images'/im['file_name']
    assert C.sha(src)==raw['source_image_sha256']
    with Image.open(src) as f:pil=f.convert('RGB')
    assert pil.size==(ow,oh)
    crop=G.crop128(pil,w,(ow,oh))
    result={'image_id':iid,'orig_wh':[ow,oh],'file_name':im['file_name'],
        'final300_boxes':pb,'final300_logits':logits,'fullp_boxes':full_boxes,'fullp_scores':scores,
        'fullp_labels':labels,'fullp_query_index':queries,'fullp_logits_row':full_logits,
        'fullp_flatten_index':flat,'window_cell':list(rc),'window_xyxy_640':w,
        'G_W':gw,'U_W':uw,'m0_gt_explained':explained,'m0_cand_owner':owner,'m0_coverage':sol['coverage'],
        'K_g':kg,'gt_boxes_640':gt,'gt_labels':gl,'gt_annotation_ids':[a['id'] for a in items],
        # Preserve the approved 341 field's G_W-aligned semantics, while also
        # storing unambiguous full-GT values for future evaluation.
        'gt_short_side_640':G.short_side(gt[gw]),'gt_short_side_640_all':G.short_side(gt),
        'gt_short_side_640_annotation_ids':[a['id'] for j,a in enumerate(items) if bool(gw[j])],
        'n_gt':len(gt),'n_G_W':int(gw.sum()),'n_U_W':int(uw.sum()),'crop128':crop,
        'source_image_sha256':raw['source_image_sha256'],'annotation_sha256':annotation_sha,
        'checkpoint_sha256':raw['checkpoint_sha256'],'raw_cache_file_sha256':C.sha(raw_path),
        'm0_cost_matrix':cost,'m0_native_iou':iou,'m0_legal_mask':legal,
        'm0_cand_boxes_norm':cb,'m0_gt_boxes_norm':gb,'m0_exact':sol,'m0_certificate_integer_encoding':'decimal_string',
        'M0_version':'342 exact candidate-instance class + FP32 exact cost + intrinsic flatten-index owner lex',
        'seed':raw['seed']}
    # Legacy numeric subset hash remains available for explicit parity;
    # complete payload hash below covers ALL stored fields.
    h=hashlib.sha256()
    for k in ('fullp_boxes','fullp_scores','fullp_logits_row','G_W','U_W','K_g','gt_boxes_640','crop128'):
        h.update(k.encode());h.update(result[k].contiguous().numpy().tobytes())
    result['sha256']=h.hexdigest();result['stage3_payload_sha256']=payload_sha(result)
    return result

def run(seed,tag,stop_after_new=None):
    assert json.loads((C.HERE/'342_M0修复放行记录.json').read_text(encoding='utf-8'))['user_answer']=='允许修复并按字段披露差异'
    assert json.loads((C.HERE/'342_result_阶段3单元验收.json').read_text(encoding='utf-8'))['status']=='已验证'
    ims,by,ah=C.metadata();folder=C.HERE/f'342_cache_val_v2_seed{seed}';folder.mkdir(exist_ok=True)
    counts={'raw_cache_loads':0,'complete_cache_loads':0,'cache_writes':0,'verified_skips':0,
        'image_decodes':0,'M0_solves':0,'global_forwards':0,'model_loads':0,'optimizer_steps':0}
    rows=[];start=time.perf_counter()
    for n,im in enumerate(ims,1):
        iid=im['id'];p=entry_path(seed,iid);rawpath=C.HERE/f'342_raw_val_seed{seed}'/f'342_{iid:06d}.pt'
        if p.exists():
            e=torch.load(p,map_location='cpu',weights_only=True);counts['complete_cache_loads']+=1
            assert e['image_id']==iid and e['seed']==seed and e['annotation_sha256']==ah
            assert payload_sha(e)==e['stage3_payload_sha256']
            assert e['raw_cache_file_sha256']==C.sha(rawpath)
            counts['verified_skips']+=1;rows.append({'image_id':iid,'status':'verified_skip','sha256':C.sha(p)})
            continue
        raw=torch.load(rawpath,map_location='cpu',weights_only=True);counts['raw_cache_loads']+=1
        assert raw['image_id']==iid and raw['seed']==seed and raw['annotation_sha256']==ah
        e=build_entry(raw,im,by.get(iid,[]),ah,rawpath);counts['M0_solves']+=1;counts['image_decodes']+=1
        tmp=folder/f'342_{iid:06d}_incomplete_{uuid.uuid4().hex}.pt'
        with open(tmp,'xb') as f:torch.save(e,f)
        os.rename(tmp,p);counts['cache_writes']+=1
        rows.append({'image_id':iid,'status':'created','sha256':C.sha(p)})
        if counts['cache_writes']%32==0 or n==548:
            print(f'full seed={seed} position={n}/548 new={counts["cache_writes"]} skipped={counts["verified_skips"]} elapsed={time.perf_counter()-start:.1f}s',flush=True)
        if stop_after_new is not None and counts['cache_writes']>=stop_after_new:break
    res={'status':'已生成，待独立验收','seed':seed,'tag':tag,'completed_entries':len(list(folder.glob('342_*.pt'))),
         'counts':counts,'elapsed_seconds':time.perf_counter()-start,'rows':rows,'source_sha256':C.sources(),
         '已执行':counts,'未执行':['真实训练','GPU','阶段4至5'],'未核':['全量独立验收'],'待主控':['阶段3闸门']}
    C.json_out(f'342_full_seed{seed}_{tag}_result.json',res)
    print(json.dumps({'seed':seed,'tag':tag,'entries':res['completed_entries'],'counts':counts}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[1,2],required=True)
    p.add_argument('--tag',required=True);p.add_argument('--stop-after-new',type=int)
    a=p.parse_args();run(a.seed,a.tag,a.stop_after_new)
