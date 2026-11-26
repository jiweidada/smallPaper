"""Scan all frozen raw inputs, complete #1466, seal all 6470 new development inputs."""
import importlib,json,time,hashlib,types,traceback
C=importlib.import_module('344_context')
import torch
B=importlib.import_module('342_build_val_v3');D=importlib.import_module('343_development_v2')
G=C.load_geometry();M=importlib.import_module('342_m0_exact')
torch.set_num_threads(2)
COUNTS={'old340_loads':0,'new_cache_loads':0,'new_cache_writes':0,'fullP_scans':0,'non_M0_old_comparisons':0,
        'M0_solves':0,'image_decodes':0,'global_model_loads':0,'global_forwards':0,'head_forwards':0,'optimizer_steps':0}

def rebuild(old,im,items,ah,path,checkpoint_sha):
    # A copy of the approved builder, changing only the authorized M0 bad row.
    pb=old['final300_boxes'];logits=old['final300_logits'];ow,oh=old['orig_wh']
    p300=G.raw_query_cxcywh_to_640_xyxy(pb)
    scores,flat=logits.sigmoid().flatten().topk(300);keep=scores>.001
    scores=scores[keep];flat=flat[keep];labels=flat%10;queries=flat//10
    full_boxes=p300[queries];full_logits=logits[queries]
    rc,_=G.select_raw_query_window(p300);w=G.window_xyxy(rc)
    xywh=torch.tensor([a['bbox'] for a in items],dtype=torch.float32).reshape(-1,4)
    gt=G.orig_xywh_to_640_xyxy(xywh,ow,oh);gl=torch.tensor([a['category_id'] for a in items],dtype=torch.long)
    gb=G.full_plane_cxcywh(gt)
    cb,cost,iou,legal,bad=G.fullp_m0_matrices(pb,queries,full_boxes,full_logits,gb,gl,labels)
    sol=dict(M.exact_assignment(cost.numpy(),legal.numpy(),flat.tolist()));COUNTS['M0_solves']+=1
    for field in ('cost_integer_sum','owner_vector_lex','edge_cost_shift'):sol[field]=str(sol[field])
    sol['dual_potentials']=[str(x) for x in sol['dual_potentials']]
    owner=torch.tensor(sol['cand_owner'],dtype=torch.long);explained=torch.zeros(len(gt),dtype=torch.bool)
    if (owner>=0).any():explained[owner[owner>=0]]=True
    gw=G.complete_mask(gt,w);uw=gw&~explained;kg=G.kg_window(gt,gw,w)
    from PIL import Image
    src=C.TRAIN/'images'/im['file_name'];image_sha=C.sha(src)
    with Image.open(src) as f:pil=f.convert('RGB')
    COUNTS['image_decodes']+=1;crop=G.crop128(pil,w,(ow,oh))
    e={'image_id':im['id'],'orig_wh':[ow,oh],'file_name':im['file_name'],
       'final300_boxes':pb,'final300_logits':logits,'fullp_boxes':full_boxes,'fullp_scores':scores,
       'fullp_labels':labels,'fullp_query_index':queries,'fullp_logits_row':full_logits,'fullp_flatten_index':flat,
       'window_cell':list(rc),'window_xyxy_640':w,'G_W':gw,'U_W':uw,'m0_gt_explained':explained,
       'm0_cand_owner':owner,'m0_coverage':sol['coverage'],'K_g':kg,'gt_boxes_640':gt,'gt_labels':gl,
       'gt_annotation_ids':[a['id'] for a in items],'gt_short_side_640':G.short_side(gt[gw]),
       'gt_short_side_640_all':G.short_side(gt),'gt_short_side_640_annotation_ids':[a['id'] for j,a in enumerate(items) if bool(gw[j])],
       'n_gt':len(gt),'n_G_W':int(gw.sum()),'n_U_W':int(uw.sum()),'crop128':crop,
       'source_image_sha256':image_sha,'annotation_sha256':ah,'checkpoint_sha256':checkpoint_sha,
       'raw_cache_file_sha256':C.sha(path),'m0_cost_matrix':cost,'m0_native_iou':iou,'m0_legal_mask':legal,
       'm0_cand_boxes_norm':cb,'m0_gt_boxes_norm':gb,'m0_exact':sol,'m0_certificate_integer_encoding':'decimal_string',
       'M0_version':'342 exact + 344 explicit raw nonpositive w/h exception','seed':2,
       'fullp_nonpositive_mask':bad,'m0_cost_row_evaluated_mask':~bad,
       'm0_nonpositive_reason':'344: raw w<=0 or h<=0; retain fullP/budget, IoU=0, legal=False, cost row not evaluated (stored zero placeholder).'}
    h=hashlib.sha256()
    for k in ('fullp_boxes','fullp_scores','fullp_logits_row','G_W','U_W','K_g','gt_boxes_640','crop128'):
        h.update(k.encode());h.update(e[k].contiguous().numpy().tobytes())
    e['sha256']=h.hexdigest();e['stage3_payload_sha256']=B.payload_sha(e)
    return e

def main():
    t0=time.perf_counter();ids,ims,by,ah,idx=D.metadata()
    pre=json.loads((C.S343/'343_result_3b来源预检.json').read_text(encoding='utf-8'))
    seals={(r['seed'],r['image_id']):r['current_340_file_sha256'] for r in pre['entries']}
    hits=[];raw_unselected=0;scan=[]
    try:
        # Finish the entire exception-scope scan before writing any new target.
        for seed in (1,2):
            for n,iid in enumerate(ids,1):
                source=C.WORK/f'340_cache_seed{seed}'/f'{iid:06d}.pt';assert C.sha(source)==seals[seed,iid]
                old=torch.load(source,map_location='cpu',weights_only=True);COUNTS['old340_loads']+=1
                pb=old['final300_boxes'];lo=old['final300_logits'];assert torch.isfinite(pb).all() and torch.isfinite(lo).all()
                score,flat=lo.sigmoid().flatten().topk(300);flat=flat[score>.001];query=flat//10
                bad=G.fullp_nonpositive_mask(pb,query);COUNTS['fullP_scans']+=1
                assert torch.equal(old['fullp_boxes'],G.raw_query_cxcywh_to_640_xyxy(pb)[query])
                assert torch.equal(old['fullp_labels'],flat%10)
                selected=set(query.tolist());unselected=[q for q in range(300) if q not in selected and bool((pb[q,2:]<=0).any())]
                raw_unselected+=len(unselected)
                for j in bad.nonzero().flatten().tolist():
                    hits.append({'seed':seed,'image_id':iid,'fullp_index':j,'query_index':int(query[j]),
                        'flatten_index':int(flat[j]),'raw_cxcywh':pb[query[j]].tolist(),'score':float(old['fullp_scores'][j])})
                # A positive raw w/h becoming a nonpositive edge is not covered by the exception.
                G.intersecting_fullp_candidates(old['fullp_boxes'],old['window_xyxy_640'],bad)
                scan.append({'seed':seed,'image_id':iid,'raw_file_sha256':seals[seed,iid],'fullp_nonpositive_count':int(bad.sum())})
                if n%256==0 or n==3235:print(f'344 raw scan seed={seed} {n}/3235 hits={len(hits)}',flush=True)
        C.write('344_result_非正框全量扫描.json',{'status':'已验证' if len(hits)==1 else '待主控',
            'fullP_hits':hits,'unselected_raw_nonpositive_queries':raw_unselected,'scope':'命中数按fullP候选实例；未入fullP的raw查询另列，不触发本合同。',
            'per_entry':scan,'已执行':dict(COUNTS),'未执行':['真实训练','GPU','阶段5'],'未核':['主控独立复核'],'source_sha256':C.sources()})
        if len(hits)!=1 or (hits[0]['seed'],hits[0]['image_id'],hits[0]['query_index'])!=(2,1466,62):
            raise RuntimeError('待主控：非正框命中范围不等于授权的唯一实例')
        source=C.WORK/'340_cache_seed2/001466.pt';old=torch.load(source,map_location='cpu',weights_only=True);COUNTS['old340_loads']+=1
        cp=next(p for p in idx['checkpoints'] if p['seed']==2);assert C.sha(cp['path'])==cp['expected_SHA256_from_288']
        e=rebuild(old,ims[1466],by.get(1466,[]),ah,source,cp['expected_SHA256_from_288']);D.compare_old(old,e)
        assert e['n_G_W']==0 and e['n_U_W']==0 and int(e['fullp_nonpositive_mask'].sum())==1
        folder=C.HERE/'344_cache_development_seed2';folder.mkdir()
        target=folder/'344_001466.pt'
        with open(target,'xb') as f:torch.save(e,f)
        COUNTS['new_cache_writes']+=1
        prior=json.loads((C.S343/'343_development6469_逐条SHA清单.json').read_text(encoding='utf-8'))
        prior={(x['seed'],x['image_id']):x for x in prior['entries']}
        manifest=[];diff=[];stats={};windows={}
        for seed in (1,2):
            us=[];gs=[];cross=0;crossgw=0;uwchanged=0;ownerchanged=0;windows[seed]={}
            for n,iid in enumerate(ids,1):
                p=target if (seed,iid)==(2,1466) else C.Path(prior[seed,iid]['path']) if hasattr(C,'Path') else __import__('pathlib').Path(prior[seed,iid]['path'])
                if (seed,iid)!=(2,1466):assert C.sha(p)==prior[seed,iid]['file_sha256']
                e=torch.load(p,map_location='cpu',weights_only=True);COUNTS['new_cache_loads']+=1
                oldpath=C.WORK/f'340_cache_seed{seed}'/f'{iid:06d}.pt'
                assert C.sha(oldpath)==seals[seed,iid]
                old=torch.load(oldpath,map_location='cpu',weights_only=True);COUNTS['old340_loads']+=1
                equal=D.compare_old(old,e);COUNTS['non_M0_old_comparisons']+=1
                assert e['seed']==seed and e['image_id']==iid and e['annotation_sha256']==ah
                assert e['raw_cache_file_sha256']==seals[seed,iid] and B.payload_sha(e)==e['stage3_payload_sha256']
                bad=G.fullp_nonpositive_mask(e['final300_boxes'],e['fullp_query_index'])
                if 'fullp_nonpositive_mask' in e:assert torch.equal(bad,e['fullp_nonpositive_mask'])
                assert int(bad.sum())==int((seed,iid)==(2,1466))
                assert not e['m0_legal_mask'][bad].any() and (e['m0_cand_owner'][bad]==-1).all()
                oc=old['m0_cand_owner'];owned=(oc>=0).nonzero().flatten();bc=owned[old['fullp_labels'][owned]!=old['gt_labels'][oc[owned]]]
                cross+=len(bc);crossgw+=int(old['G_W'][oc[bc]].sum())
                uch=not torch.equal(old['U_W'],e['U_W']);och=not torch.equal(oc,e['m0_cand_owner']);uwchanged+=uch;ownerchanged+=och
                diff.append({'seed':seed,'image_id':iid,'non_M0_old_field_count':len(equal),'all_non_M0_bit_equal':True,
                    'U_W_changed':uch,'owner_changed':och,'fullp_nonpositive_count':int(bad.sum())})
                us.append(e['n_U_W']);gs.append(e['n_G_W']);windows[seed][iid]=tuple(e['window_cell'])
                manifest.append({'seed':seed,'image_id':iid,'path':str(p),'bytes':p.stat().st_size,'file_sha256':C.sha(p),
                    'payload_sha256':e['stage3_payload_sha256'],'source_340_file_sha256':seals[seed,iid],
                    'image_sha256':e['source_image_sha256'],'checkpoint_sha256':e['checkpoint_sha256'],
                    'fullp_nonpositive_indices':bad.nonzero().flatten().tolist(),'fullp_nonpositive_mask_stored': 'fullp_nonpositive_mask' in e})
                if n%256==0 or n==3235:print(f'344 manifest seed={seed} {n}/3235 compared={COUNTS["non_M0_old_comparisons"]}',flush=True)
            stats[str(seed)]={'N':len(us),'G_W_positive':sum(x>0 for x in gs),'U_W_positive':sum(x>0 for x in us),
                'U_W_gt4_images':sum(x>4 for x in us),'overflow_GT':sum(max(0,x-4) for x in us),
                'U_W_total':sum(us),'G_W_total':sum(gs),'old_cross_class_owners':cross,'old_cross_class_owners_in_G_W':crossgw,
                'U_W_changed_images':uwchanged,'owner_changed_images':ownerchanged}
        assert len(manifest)==6470
        same=sum(windows[1][i]==windows[2][i] for i in ids);assert same==1443
        C.write('344_development6470_逐条SHA清单.json',{'entries':manifest,'annotation_sha256':ah,
            'binding_contract':'6469条引用343只读文件；1466为344新文件。逐条文件SHA和payload核对后，由raw w/h重算fullp_nonpositive_mask；旧343条目不重写。',
            '已执行':{'entries':6470},'未执行':['真实训练'],'未核':['主控独立复核'],'source_sha256':C.sources()})
        C.write('344_result_6470补全与非M0对账.json',{'status':'已验证（6470条补全及SHA/非M0字段对账）','stats':stats,
            'same_window_count':same,'per_entry':diff,'new_1466_path':str(target),'new_1466_sha256':C.sha(target),
            'exception':hits[0],'exception_labels_unchanged':True,'exception_G_W':0,'exception_U_W':0,
            '已执行':COUNTS,'wall_seconds':time.perf_counter()-t0,'未执行':['真实训练','GPU','阶段5','全量重新生成M0（仅补1466）'],
            '未核':['#1466独立几何/M0验收（另行执行）','主控独立复核'],'source_sha256':C.sources()})
        print(json.dumps({'status':'PASS','stats':stats,'same_window_count':same,'counts':COUNTS},ensure_ascii=False),flush=True)
    except Exception as ex:
        C.write('344_result_6470补全_实现失败.json',{'status':'待主控' if '待主控' in str(ex) else '实现失败',
            'exception':repr(ex),'traceback':traceback.format_exc(),'已执行':COUNTS,'未执行':['真实训练','GPU','阶段5'],
            '未核':['6470条补全'],'source_sha256':C.sources()});raise

if __name__=='__main__':main()
