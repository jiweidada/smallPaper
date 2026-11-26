"""All 1096 immutable complete entries: independent geometry/targets/crop/M0.

No head/global forwards or optimizer calls. Both seeds receive identical checks.
"""
import importlib,json,time,math,difflib,traceback,sys
C=importlib.import_module('342_stage3_context')
import torch,numpy as np
from PIL import Image,ImageDraw
import geom342 as G
B=importlib.import_module('342_build_val_v2');I=importlib.import_module('342_val_independent')
torch.set_num_threads(2)

def decode(sol):
    out=dict(sol)
    for key in ('cost_integer_sum','owner_vector_lex','edge_cost_shift'):out[key]=int(out[key])
    out['dual_potentials']=[int(x) for x in out['dual_potentials']]
    return out

def independent_window_640(boxes):
    # Same canonical FP32 source plane as the frozen selector; independent
    # cell enumeration and vector counts, rather than histogram bin assignment.
    b=np.asarray(boxes,dtype=np.float32)
    cx=(b[:,0]+b[:,2])/np.float32(2);cy=(b[:,1]+b[:,3])/np.float32(2)
    best=-1;answer=None
    for row in range(10):
        for col in range(10):
            mask=(cx>=col*64)&(cy>=row*64)&(cx<=(col+1)*64 if col==9 else cx<(col+1)*64)&(cy<=(row+1)*64 if row==9 else cy<(row+1)*64)
            count=int(mask.sum())
            if count>best:best=count;answer=(row,col)
    row,col=answer
    return answer,np.array([col*64,row*64,col*64+64,row*64+64],dtype=np.float64)

def independent_native_iou(cb,gb):
    # Separate NumPy representation conversion and area computations, FP32
    # because legal-edge threshold is explicitly native FP32 IoU >= .5.
    c=np.asarray(cb,dtype=np.float32);g=np.asarray(gb,dtype=np.float32)
    c0=c[:,:2]-c[:,2:]*np.float32(.5);c1=c[:,:2]+c[:,2:]*np.float32(.5)
    g0=g[:,:2]-g[:,2:]*np.float32(.5);g1=g[:,:2]+g[:,2:]*np.float32(.5)
    x=np.maximum(0,np.minimum(c1[:,None,0],g1[None,:,0])-np.maximum(c0[:,None,0],g0[None,:,0]))
    y=np.maximum(0,np.minimum(c1[:,None,1],g1[None,:,1])-np.maximum(c0[:,None,1],g0[None,:,1]))
    inter=x*y;ca=(c1[:,0]-c0[:,0])*(c1[:,1]-c0[:,1]);ga=(g1[:,0]-g0[:,0])*(g1[:,1]-g0[:,1])
    return inter/np.maximum(ca[:,None]+ga[None,:]-inter,np.float32(1e-12))

def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return {'count':k,'n':n,'rate':p,'wilson95':[center-half,center+half]}

def png(e,pixels,reference,seed):
    rgb=np.asarray(pixels)
    canvas=Image.new('RGB',(1584,612),'white');d=ImageDraw.Draw(canvas)
    original=Image.fromarray(rgb).resize((512,512),Image.BILINEAR)
    crop=Image.fromarray((e['crop128'].permute(1,2,0).numpy().clip(0,1)*255).round().astype('uint8')).resize((512,512),Image.NEAREST)
    canvas.paste(original,(8,64));canvas.paste(crop,(536,64))
    full=torch.tensor([0.,0.,640.,640.])
    wbox=G.win_norm_to_window_pixels(G.win_norm_xyxy(e['window_xyxy_640'][None],full),full,(512,512))[0].numpy()
    d.rectangle(tuple(wbox+[8,64,8,64]),outline='#00ff00',width=3)
    gw=e['G_W'];gt=e['gt_boxes_640'][gw]
    origedges=G.win_norm_to_window_pixels(G.win_norm_xyxy(gt,full),full,(512,512)).numpy()
    for edge in origedges:d.rectangle(tuple(edge+[8,64,8,64]),outline='#ff3030',width=2)
    w=e['window_xyxy_640'];keep=G.intersecting_candidates(e['fullp_boxes'],w)
    edges=G.win_norm_to_window_pixels(G.win_norm_xyxy(e['fullp_boxes'][keep],w),w,(512,512)).numpy()
    for edge in edges:d.rectangle(tuple(np.clip(edge,0,511)+[536,64,536,64]),outline='#46f0f0',width=1)
    for xy in G.grid_xy().numpy()*512:
        x,y=xy+[536,64];d.ellipse((x-1,y-1,x+1,y+1),fill='white')
    mass=e['K_g'][gw].sum(0).numpy()
    for (x,y),v in zip(G.grid_xy().numpy()*512,mass):
        if v>0:
            x,y=x+536,y+64;r=2+5*np.sqrt(float(v)/max(float(mass.max()),1e-12))
            d.ellipse((x-r,y-r,x+r,y+r),outline='#ff3030',width=2)
    ref=Image.fromarray((reference.transpose(1,2,0).clip(0,1)*255).round().astype('uint8')).resize((512,512),Image.NEAREST)
    canvas.paste(ref,(1064,64))
    d.text((8,8),f'seed={seed}; id={e["image_id"]}; orig_wh={e["orig_wh"]}; W={w.tolist()}',fill='black')
    d.text((8,27),f'G_W={e["n_G_W"]}; U_W={e["n_U_W"]}; P_W={int(keep.sum())}; exact M0; unclipped data',fill='black')
    d.text((8,46),'640-plane source / W+complete GT',fill='black')
    d.text((536,46),'cached crop / P+grid+K_g',fill='black');d.text((1064,46),'independent original-pixel crop',fill='black')
    d.text((8,592),'Green=window; red=complete GT/K_g. Only drawing is clipped; display circle radii are scaled.',fill='black')
    name=f'342_val_seed{seed}_独立复核_{e["image_id"]:06d}.png'
    with open(C.HERE/name,'xb') as f:canvas.save(f,format='PNG')
    return str(C.HERE/name)

def main():
    start=time.perf_counter();ims,by,ah=C.metadata()
    selected=np.random.default_rng(2026102203).choice([i['id'] for i in ims],6,replace=False).tolist()
    counts={'complete_cache_loads':0,'raw_cache_loads':0,'image_decodes':0,'independent_windows':0,
            'independent_G_U_K':0,'independent_crops':0,'independent_M0_certificates':0,
            'old341_compares':0,'PNG生成':0,'global_forwards':0,'head_forwards':0,'optimizer_steps':0}
    allrows={};subset=[];manifests=[];pngs=[];stats={};negative=[]
    try:
        for seed in (1,2):
            rows=[];gwlist=[];ulist=[];norm_min=float('inf');norm_max=-float('inf')
            assert len(list((C.HERE/f'342_cache_val_v2_seed{seed}').glob('342_*.pt')))==548
            for n,im in enumerate(ims,1):
                iid=im['id'];path=B.entry_path(seed,iid)
                e=torch.load(path,map_location='cpu',weights_only=True);counts['complete_cache_loads']+=1
                assert e['image_id']==iid and e['seed']==seed and e['annotation_sha256']==ah
                assert B.payload_sha(e)==e['stage3_payload_sha256']
                rawpath=C.HERE/f'342_raw_val_seed{seed}'/f'342_{iid:06d}.pt'
                raw=torch.load(rawpath,map_location='cpu',weights_only=True);counts['raw_cache_loads']+=1
                assert C.sha(rawpath)==e['raw_cache_file_sha256']
                assert all(torch.equal(e[k],raw[k]) for k in ('final300_boxes','final300_logits'))
                p300=G.norm_cxcywh_to_640_xyxy(e['final300_boxes'])
                expected_pred=I.raw640_boxes(e['final300_boxes'].numpy())
                pred_err=float(np.abs(p300.numpy()-expected_pred).max())
                assert pred_err<8*np.finfo(np.float32).eps*640
                rc,w=independent_window_640(p300.numpy());counts['independent_windows']+=1
                assert list(rc)==e['window_cell'] and np.array_equal(w,e['window_xyxy_640'].numpy())
                assert np.array_equal(w[2:]-w[:2],[64,64])
                items=by.get(iid,[]);ow,oh=e['orig_wh']
                expected_gt=I.gt_boxes(items,ow,oh)
                gt_err=float(np.abs(expected_gt-e['gt_boxes_640'].numpy()).max()) if len(items) else 0.
                assert gt_err<8*np.finfo(np.float32).eps*640
                assert [a['id'] for a in items]==e['gt_annotation_ids']
                assert [a['category_id'] for a in items]==e['gt_labels'].tolist()
                gw=I.complete(e['gt_boxes_640'].numpy(),w)
                owners=e['m0_cand_owner'].numpy();exp=np.zeros(len(items),dtype=bool)
                exp[owners[owners>=0]]=True;uw=gw&~exp
                assert np.array_equal(gw,e['G_W'].numpy()) and np.array_equal(uw,e['U_W'].numpy())
                assert np.array_equal(exp,e['m0_gt_explained'].numpy())
                kg=I.kg(e['gt_boxes_640'].numpy(),gw,w)
                kgerr=float(np.abs(kg-e['K_g'].numpy()).max()) if kg.size else 0.
                assert kgerr<1e-5
                assert np.max(np.abs(kg[gw].sum(1)-1),initial=0)<1e-9
                counts['independent_G_U_K']+=1
                iou=independent_native_iou(e['m0_cand_boxes_norm'].numpy(),e['m0_gt_boxes_norm'].numpy())
                legal=(e['fullp_labels'].numpy()[:,None]==e['gt_labels'].numpy()[None,:])&(iou>=.5)
                assert np.array_equal(legal,e['m0_legal_mask'].numpy())
                iouerr=float(np.abs(iou-e['m0_native_iou'].numpy()).max()) if iou.size else 0.
                assert iouerr<1e-6
                cert=I.certificate(e['m0_cost_matrix'].numpy(),legal,e['fullp_flatten_index'].numpy(),decode(e['m0_exact']))
                counts['independent_M0_certificates']+=1
                matched=np.flatnonzero(owners>=0)
                assert all(legal[j,owners[j]] for j in matched)
                with Image.open(C.VAL/'images'/im['file_name']) as f:pixels=np.asarray(f.convert('RGB'))
                counts['image_decodes']+=1
                reference=I.crop(pixels,w);cerr=float(np.abs(reference-e['crop128'].numpy()).max())
                budget=I.crop_error_budget(ow,oh)
                assert cerr<budget,(seed,iid,cerr,budget)
                counts['independent_crops']+=1
                short=np.minimum(e['gt_boxes_640'][:,2].numpy()-e['gt_boxes_640'][:,0].numpy(),e['gt_boxes_640'][:,3].numpy()-e['gt_boxes_640'][:,1].numpy())
                assert np.array_equal(short,e['gt_short_side_640_all'].numpy())
                assert np.array_equal(short[gw],e['gt_short_side_640'].numpy())
                keep=G.intersecting_candidates(e['fullp_boxes'],e['window_xyxy_640'])
                loc=G.win_norm_xyxy(e['fullp_boxes'][keep],e['window_xyxy_640'])
                if loc.numel():norm_min=min(norm_min,float(loc.min()));norm_max=max(norm_max,float(loc.max()))
                row={'image_id':iid,'G_W':int(gw.sum()),'U_W':int(uw.sum()),'M0_coverage':e['m0_coverage'],
                     'pred_geometry_abs_diff':pred_err,'GT_geometry_abs_diff':gt_err,'K_g_abs_diff':kgerr,
                     'native_IoU_abs_diff':iouerr,'crop_max_abs_diff':cerr,'crop_a_priori_bound':budget,
                     'M0_exact_dual_certificate':cert['exact_residual_dual_feasible']}
                rows.append(row);gwlist.append(int(gw.sum()));ulist.append(int(uw.sum()))
                manifests.append({'seed':seed,'image_id':iid,'path':str(path),'bytes':path.stat().st_size,
                    'file_sha256':C.sha(path),'payload_sha256':e['stage3_payload_sha256'],
                    'image_sha256':e['source_image_sha256'],'checkpoint_sha256':e['checkpoint_sha256'],
                    'annotation_sha256':ah,'M0_cost_sha256':__import__('hashlib').sha256(e['m0_cost_matrix'].numpy().tobytes()).hexdigest()})
                if iid in selected:
                    pngs.append(png(e,pixels,reference,seed));counts['PNG生成']+=1
                    # Wrong row/column and whole-plane K_g fixtures are checked
                    # independently from normal successful comparisons.
                    if kg.size and np.max(kg,initial=0)>0:
                        wrong=kg.reshape(-1,16,16).transpose(0,2,1).reshape(-1,256)
                        detected=bool(np.max(np.abs(wrong-e['K_g'].numpy()))>1e-5)
                        negative.append({'seed':seed,'image_id':iid,'kind':'transpose_K_g','detected':detected})
                old=C.WORK/'341_cache_val_seed1'/f'{iid:06d}.pt'
                if seed==1 and old.exists():
                    prior=torch.load(old,map_location='cpu',weights_only=True);counts['old341_compares']+=1
                    mutable={'m0_gt_explained','m0_cand_owner','m0_coverage','U_W','n_U_W','sha256'}
                    equal={}
                    for key,value in prior.items():
                        if key in mutable:continue
                        assert key in e,key
                        equal[key]=torch.equal(value,e[key]) if torch.is_tensor(value) else value==e[key]
                    assert all(equal.values()),(iid,equal)
                    changed=(prior['m0_cand_owner']!=e['m0_cand_owner']).nonzero().flatten().tolist()
                    uf=(prior['U_W']!=e['U_W']).nonzero().flatten().tolist()
                    subset.append({'image_id':iid,'unchanged_fields_bit_equal':equal,
                        'M0_owner_changed_indices':changed,'M0_owner_before':[int(prior['m0_cand_owner'][j]) for j in changed],
                        'M0_owner_after':[int(e['m0_cand_owner'][j]) for j in changed],
                        'M0_coverage_before':int(prior['m0_coverage']),'M0_coverage_after':e['m0_coverage'],
                        'U_W_flip_GT_indices':uf,'U_W_flip_annotation_ids':[e['gt_annotation_ids'][j] for j in uf],
                        'n_U_W_before':int(prior['n_U_W']),'n_U_W_after':e['n_U_W']})
                if n%64==0 or n==548:print(f'independent seed={seed} checked={n}/548 elapsed={time.perf_counter()-start:.1f}s',flush=True)
            stats[str(seed)]={'G_W_positive':wilson(sum(x>0 for x in gwlist),548),
                'U_W_positive':wilson(sum(x>0 for x in ulist),548),'U_W_gt4':wilson(sum(x>4 for x in ulist),548),
                'total_G_W_GT':sum(gwlist),'total_U_W_GT':sum(ulist),'sum_min4_U_W':sum(min(4,x) for x in ulist),
                'overflow_GT':sum(max(0,x-4) for x in ulist),
                'P_W_coordinate_range':{'scope':f'seed{seed} val548全部窗的相交候选，保持未clip','min':norm_min,'max':norm_max},
                'max_K_g_abs_diff':max(r['K_g_abs_diff'] for r in rows),
                'max_crop_abs_diff':max(r['crop_max_abs_diff'] for r in rows),
                'max_GT_geometry_abs_diff':max(r['GT_geometry_abs_diff'] for r in rows)}
            allrows[str(seed)]=rows
        assert len(subset)==32 and len(manifests)==1096
        # Constructed reverse tests avoid coincidentally symmetric real images.
        fixture=np.zeros((32,256));fixture[0,2*16+7]=1
        assert not np.array_equal(fixture,fixture.reshape(-1,16,16).transpose(0,2,1).reshape(-1,256))
        negative.append({'kind':'constructed_row_column_swap','detected':True})
        result={'status':'已验证（阶段3工程验收）；待主控','stats':stats,'per_image':allrows,
                'old341_32_comparison':subset,'old341_summary':{'all_non_M0_fields_bit_equal':True,
                    'images_with_owner_changes':sum(bool(r['M0_owner_changed_indices']) for r in subset),
                    'images_with_U_W_changes':sum(bool(r['U_W_flip_GT_indices']) for r in subset)},
                'PNG_ids_per_seed':selected,'png_paths':pngs,'reverse_tests':negative,
                'elapsed_seconds':time.perf_counter()-start,'已执行':counts,'source_sha256':C.sources(),
                '未执行':['真实训练','GPU','SSH','队列','heldout3236读取','阶段4至5'],
                '未核':['主控独立复核','按新M0版本重建development3235×2缓存'],
                '待主控':['阶段3闸门：复核统计、M0差异与两seed PNG后再决定阶段4']}
        C.json_out('342_result_val548独立验收.json',result)
        C.json_out('342_val548_逐条SHA清单.json',{'entries':manifests,'已执行':{'entry_count':len(manifests)},
            'source_sha256':C.sources(),'未执行':['真实训练'],'未核':['主控独立复核']})
        old=(C.WORK/'uic_m0.py').read_text(encoding='utf-8');new=(C.HERE/'342_m0_exact.py').read_text(encoding='utf-8')
        C.text_out('342_M0_新旧算法.diff',''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(C.WORK/'uic_m0.py'),tofile=str(C.HERE/'342_m0_exact.py'))))
        print(json.dumps({'status':'pass','stats':stats,'old341_summary':result['old341_summary'],'counts':counts},ensure_ascii=False),flush=True)
    except Exception as ex:
        C.json_out('342_result_val548独立验收_实现失败.json',{'status':'实现失败','error':repr(ex),'traceback':traceback.format_exc(),
            '已执行':counts,'未执行':['真实训练','阶段4至5'],'未核':['val548全量独立验收'],'source_sha256':C.sources()})
        raise

if __name__=='__main__':main()
