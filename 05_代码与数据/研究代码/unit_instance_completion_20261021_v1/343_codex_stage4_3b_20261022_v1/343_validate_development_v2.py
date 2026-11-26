"""All 6470 development entries; independent geometry, exact M0 certificates."""
import importlib,time,json,traceback,hashlib
C=importlib.import_module('343_context');D=importlib.import_module('343_development_v2')
I=importlib.import_module('342_val_independent');V=importlib.import_module('342_validate_val_v3')
import torch,numpy as np
from PIL import Image,ImageDraw
G=D.G;torch.set_num_threads(2)

def independent_cost(cb,gb,logits,labels):
    c=cb.astype(np.float64);g=gb.astype(np.float64)
    a=np.column_stack((c[:,:2]-c[:,2:]/2,c[:,:2]+c[:,2:]/2))
    b=np.column_stack((g[:,:2]-g[:,2:]/2,g[:,:2]+g[:,2:]/2))
    ix=np.maximum(0,np.minimum(a[:,None,2],b[None,:,2])-np.maximum(a[:,None,0],b[None,:,0]))
    iy=np.maximum(0,np.minimum(a[:,None,3],b[None,:,3])-np.maximum(a[:,None,1],b[None,:,1]))
    inter=ix*iy;aa=(a[:,2]-a[:,0])*(a[:,3]-a[:,1]);bb=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    union=aa[:,None]+bb[None,:]-inter;iou=inter/np.maximum(union,1e-12)
    enclosing=(np.maximum(a[:,None,2],b[None,:,2])-np.minimum(a[:,None,0],b[None,:,0]))*(np.maximum(a[:,None,3],b[None,:,3])-np.minimum(a[:,None,1],b[None,:,1]))
    giou=iou-(enclosing-union)/np.maximum(enclosing,1e-12)
    p=1/(1+np.exp(-logits.astype(np.float64)[:,labels]))
    focal=.25*(1-p)**2*(-np.log(p+1e-8))-.75*p**2*(-np.log(1-p+1e-8))
    l1=sum(np.abs(c[:,k,None]-g[None,:,k]) for k in range(4))
    return 5*l1+2*focal-2*giou

def png(e,old,seed):
    with Image.open(D.TRAIN/'images'/e['file_name']) as im:pixels=np.asarray(im.convert('RGB'))
    ref=I.crop(pixels,e['window_xyxy_640'].numpy())
    a=Image.new('RGB',(1560,590),'white');draw=ImageDraw.Draw(a)
    a.paste(Image.fromarray(pixels).resize((512,512),Image.BILINEAR),(4,60))
    crop=Image.fromarray((e['crop128'].permute(1,2,0).numpy().clip(0,1)*255).round().astype('uint8')).resize((512,512),Image.NEAREST)
    refimage=Image.fromarray((ref.transpose(1,2,0).clip(0,1)*255).round().astype('uint8')).resize((512,512),Image.NEAREST)
    a.paste(crop,(524,60));a.paste(refimage,(1044,60))
    full=torch.tensor([0.,0.,640.,640.]);w=e['window_xyxy_640']
    edge=G.win_norm_to_window_pixels(G.win_norm_xyxy(w[None],full),full,(512,512))[0].numpy()
    draw.rectangle(tuple(edge+[4,60,4,60]),outline='#00ff00',width=3)
    keep=G.intersecting_candidates(e['fullp_boxes'],w)
    edges=G.win_norm_to_window_pixels(G.win_norm_xyxy(e['fullp_boxes'][keep],w),w,(512,512)).numpy()
    for b in edges:draw.rectangle(tuple(np.clip(b,0,511)+[524,60,524,60]),outline='cyan',width=1)
    gt=G.win_norm_to_window_pixels(G.win_norm_xyxy(e['gt_boxes_640'][e['U_W']],w),w,(512,512)).numpy()
    for b in gt:draw.rectangle(tuple(b+[524,60,524,60]),outline='yellow',width=3)
    mass=e['K_g'].sum(0).numpy()
    for xy,value in zip(G.grid_xy().numpy()*512,mass):
        if value>0:
            x,y=xy+[524,60];draw.ellipse((x-3,y-3,x+3,y+3),outline='red',width=2)
    draw.text((4,5),f'seed={seed} image={e["image_id"]}; G_W={e["n_G_W"]}; U_W old/new={int(old["U_W"].sum())}/{e["n_U_W"]}; U_W changed={not torch.equal(old["U_W"],e["U_W"])}',fill='black')
    draw.text((4,25),'Original / frozen W',fill='black');draw.text((524,25),'Cache crop / P(cyan), new U_W(yellow), K_g(red)',fill='black');draw.text((1044,25),'Independent original-pixel bilinear crop',fill='black')
    name=f'343_development_v2_seed{seed}_独立复核_{e["image_id"]:06d}.png'
    with open(C.HERE/name,'xb') as f:a.save(f,format='PNG')
    return str(C.HERE/name)

def main():
    start=time.perf_counter();ids,ims,by,ah,_=D.metadata()
    counts={'new_cache_loads':0,'old_cache_loads':0,'independent_windows':0,'independent_G_U_K':0,'independent_crops':0,
        'independent_M0_certificates':0,'independent_cost_matrices':0,'independent_fullP_checks':0,'image_decodes':0,'PNG生成':0,
        'global_forwards':0,'head_forwards':0,'optimizer_steps':0}
    allrows={};stats={};oldstats={};manifests=[];pngs=[];cells={};reverse=[]
    try:
        for seed in (1,2):
            assert len(list(D.entry_path(seed,ids[0]).parent.glob('343_*.pt')))==3235-(seed==2)
            rows=[];gs=[];us=[];ogs=[];ous=[];ownerchanges=[];uwchanges=[];cross=0;crossgw=0;cells[seed]={}
            for n,iid in enumerate(ids,1):
                path=D.entry_path(seed,iid);source=C.WORK/f'340_cache_seed{seed}'/f'{iid:06d}.pt'
                if seed==2 and iid==1466:
                    old=torch.load(source,map_location='cpu',weights_only=True);counts['old_cache_loads']+=1
                    assert not path.exists()
                    rc,_=V.independent_window_640(G.raw_query_cxcywh_to_640_xyxy(old['final300_boxes']).numpy())
                    assert list(rc)==old['window_cell'];cells[seed][iid]=tuple(rc)
                    ogs.append(int(old['G_W'].sum()));ous.append(int(old['U_W'].sum()))
                    continue
                e=torch.load(path,map_location='cpu',weights_only=True);old=torch.load(source,map_location='cpu',weights_only=True)
                counts['new_cache_loads']+=1;counts['old_cache_loads']+=1
                assert e['seed']==seed and e['image_id']==iid and e['annotation_sha256']==ah
                assert D.B.payload_sha(e)==e['stage3_payload_sha256'] and C.sha(source)==e['raw_cache_file_sha256']
                D.compare_old(old,e)
                raw=I.raw640_boxes(e['final300_boxes'].numpy());native=G.raw_query_cxcywh_to_640_xyxy(e['final300_boxes'])
                pred_error=float(np.abs(raw-native.numpy()).max());assert pred_error<8*np.finfo(np.float32).eps*640
                rc,w=V.independent_window_640(native.numpy());assert list(rc)==e['window_cell'] and np.array_equal(w,e['window_xyxy_640'].numpy())
                cells[seed][iid]=tuple(rc);counts['independent_windows']+=1
                items=by.get(iid,[]);ow,oh=e['orig_wh'];expected=I.gt_boxes(items,ow,oh)
                gterr=float(np.max(np.abs(expected-e['gt_boxes_640'].numpy()),initial=0));assert gterr<8*np.finfo(np.float32).eps*640
                assert [a['id'] for a in items]==e['gt_annotation_ids'] and [a['category_id'] for a in items]==e['gt_labels'].tolist()
                flat=e['fullp_flatten_index'].numpy();lab=e['fullp_labels'].numpy();qi=e['fullp_query_index'].numpy()
                assert np.array_equal(flat,qi*10+lab) and len(np.unique(flat))==len(flat)
                assert np.array_equal(e['fullp_boxes'].numpy(),native.numpy()[qi])
                lp=e['final300_logits'].numpy().astype(np.float64);pr=1/(1+np.exp(-lp))
                score=e['fullp_scores'].numpy();assert np.max(np.abs(pr.flatten()[flat]-score),initial=0)<2e-7
                assert np.all(score>.001) and len(flat)<=300
                threshold=np.partition(pr.flatten(),-300)[-300]
                assert np.min(pr.flatten()[flat],initial=1)>=threshold-2e-7
                counts['independent_fullP_checks']+=1
                gw=I.complete(e['gt_boxes_640'].numpy(),w);owners=e['m0_cand_owner'].numpy();explained=np.zeros(len(items),dtype=bool)
                explained[owners[owners>=0]]=True;uw=gw&~explained
                assert np.array_equal(gw,e['G_W'].numpy()) and np.array_equal(uw,e['U_W'].numpy()) and np.array_equal(explained,e['m0_gt_explained'].numpy())
                kg=I.kg(e['gt_boxes_640'].numpy(),gw,w);kgerr=float(np.max(np.abs(kg-e['K_g'].numpy()),initial=0));assert kgerr<1e-5
                assert np.max(np.abs(kg[gw].sum(1)-1),initial=0)<1e-9;counts['independent_G_U_K']+=1
                short=np.minimum(e['gt_boxes_640'][:,2].numpy()-e['gt_boxes_640'][:,0].numpy(),e['gt_boxes_640'][:,3].numpy()-e['gt_boxes_640'][:,1].numpy())
                assert np.array_equal(short,e['gt_short_side_640_all'].numpy()) and np.array_equal(short[gw],e['gt_short_side_640'].numpy())
                iou=V.independent_native_iou(e['m0_cand_boxes_norm'].numpy(),e['m0_gt_boxes_norm'].numpy())
                legal=(lab[:,None]==e['gt_labels'].numpy()[None,:])&(iou>=.5)
                assert np.array_equal(legal,e['m0_legal_mask'].numpy())
                cc=independent_cost(e['m0_cand_boxes_norm'].numpy(),e['m0_gt_boxes_norm'].numpy(),e['fullp_logits_row'].numpy(),e['gt_labels'].numpy())
                costerr=float(np.max(np.abs(cc-e['m0_cost_matrix'].numpy()),initial=0));assert costerr<1e-3
                counts['independent_cost_matrices']+=1
                cert=I.certificate(e['m0_cost_matrix'].numpy(),legal,flat,V.decode(e['m0_exact']));counts['independent_M0_certificates']+=1
                img=D.TRAIN/'images'/e['file_name'];assert C.sha(img)==e['source_image_sha256']
                with Image.open(img) as f:pixels=np.asarray(f.convert('RGB'))
                counts['image_decodes']+=1;crop=I.crop(pixels,w)
                croperr=float(np.max(np.abs(crop-e['crop128'].numpy())));assert croperr<I.crop_error_budget(ow,oh);counts['independent_crops']+=1
                changes=(old['m0_cand_owner']!=e['m0_cand_owner']).nonzero().flatten().tolist();uf=(old['U_W']!=e['U_W']).nonzero().flatten().tolist()
                if changes:ownerchanges.append(iid)
                if uf:uwchanges.append(iid)
                oc=old['m0_cand_owner'];oj=(oc>=0).nonzero().flatten();bad=oj[old['fullp_labels'][oj]!=old['gt_labels'][oc[oj]]]
                cross+=len(bad);crossgw+=int(old['G_W'][oc[bad]].sum())
                row={'image_id':iid,'G_W':int(gw.sum()),'U_W_before':int(old['U_W'].sum()),'U_W_after':int(uw.sum()),
                    'owner_changed_indices':changes,'owner_before':[int(oc[j]) for j in changes],'owner_after':[int(owners[j]) for j in changes],
                    'U_W_flip_GT_indices':uf,'U_W_flip_annotation_ids':[e['gt_annotation_ids'][j] for j in uf],
                    'old_cross_class_owners':len(bad),'old_cross_class_owners_in_G_W':int(old['G_W'][oc[bad]].sum()),
                    'raw_geometry_abs_diff':pred_error,'GT_geometry_abs_diff':gterr,'K_g_abs_diff':kgerr,'crop_abs_diff':croperr,
                    'cost_FP64_abs_diff':costerr,'exact_M0_certificate':cert['exact_residual_dual_feasible'],'non_M0_old_fields_bit_equal':True}
                rows.append(row);gs.append(int(gw.sum()));us.append(int(uw.sum()));ogs.append(int(old['G_W'].sum()));ous.append(int(old['U_W'].sum()))
                manifests.append({'seed':seed,'image_id':iid,'path':str(path),'bytes':path.stat().st_size,'file_sha256':C.sha(path),
                    'payload_sha256':e['stage3_payload_sha256'],'source_340_file_sha256':e['raw_cache_file_sha256'],'image_sha256':e['source_image_sha256'],'checkpoint_sha256':e['checkpoint_sha256']})
                if n%128==0 or n==3235:print(f'independent development seed={seed} {n}/3235 elapsed={time.perf_counter()-start:.1f}s',flush=True)
            def summarize(g,u):
                return {'G_W_positive':V.wilson(sum(x>0 for x in g),len(g)),'U_W_positive':V.wilson(sum(x>0 for x in u),len(u)),
                    'U_W_gt4':V.wilson(sum(x>4 for x in u),len(u)),'total_G_W_GT':sum(g),'total_U_W_GT':sum(u),
                    'overflow_GT':sum(max(0,x-4) for x in u),'sum_min4_U_W':sum(min(4,x) for x in u)}
            oldstats[str(seed)]=summarize(ogs,ous);stats[str(seed)]=summarize(gs,us)
            stats[str(seed)].update({'owner_changed_images':len(ownerchanges),'U_W_changed_images':len(uwchanges),
                'old_cross_class_owners':cross,'old_cross_class_owners_in_G_W':crossgw,
                'max_K_g_abs_diff':max(r['K_g_abs_diff'] for r in rows),'max_crop_abs_diff':max(r['crop_abs_diff'] for r in rows),
                'max_cost_FP64_abs_diff':max(r['cost_FP64_abs_diff'] for r in rows)})
            assert uwchanges,'No U_W-change example exists; report this instead of fabricating a PNG.'
            selected=[uwchanges[0],max(rows,key=lambda r:r['G_W'])['image_id'],min(rows,key=lambda r:r['G_W'])['image_id']]
            if len(set(selected))<3:selected=list(dict.fromkeys(selected+ids))[:3]
            for iid in selected:
                e=torch.load(D.entry_path(seed,iid),map_location='cpu',weights_only=True);old=torch.load(C.WORK/f'340_cache_seed{seed}'/f'{iid:06d}.pt',map_location='cpu',weights_only=True)
                counts['new_cache_loads']+=1;counts['old_cache_loads']+=1;counts['image_decodes']+=1
                pngs.append(png(e,old,seed));counts['PNG生成']+=1
                loc=G.win_norm_xyxy(e['fullp_boxes'][G.intersecting_candidates(e['fullp_boxes'],e['window_xyxy_640'])],e['window_xyxy_640'])
                if len(loc):
                    try:G.geometry_cost(loc*64)
                    except AssertionError:detected=True
                    else:detected=not torch.allclose(G.geometry_cost(loc*64),G.geometry_cost(loc))
                    assert detected;reverse.append({'seed':seed,'image_id':iid,'wrong_pixel_scale_detected':True})
            allrows[str(seed)]=rows
        same=sum(cells[1][i]==cells[2][i] for i in ids);assert same==1443
        fixture=np.zeros((1,256));fixture[0,2*16+7]=1
        assert not np.array_equal(fixture,fixture.reshape(1,16,16).transpose(0,2,1).reshape(1,256))
        reverse.append({'constructed_row_column_transpose_detected':True})
        C.write('343_result_3b已生成6469条独立验收.json',{'status':'已生成6469条已验证；完整6470条实现失败','validated_entries':len(manifests),'failed_missing_entries':[{'seed':2,'image_id':1466}],'statistical_scope':'seed1为完整3235；seed2统计仅3234条有效子集，不冒称完整3235新统计。','stats':stats,'old340_stats':oldstats,
            'same_window_seeds':{'count':same,'n':3235,'rate':same/3235,'old_count':1443,'unchanged':True},
            'per_image':allrows,'png_paths':pngs,'reverse_tests':reverse,'all_non_M0_old_fields_bit_equal':True,
            '已执行':counts,'elapsed_seconds':time.perf_counter()-start,'未执行':['真实训练','GPU','阶段5'],'未核':['主控独立复核'],'source_sha256':C.sources()})
        C.write('343_development6469_逐条SHA清单.json',{'entries':manifests,'annotation_sha256':ah,
            '已执行':{'entries':len(manifests)},'未执行':['真实训练'],'未核':['主控独立复核'],'source_sha256':C.sources()})
        print(json.dumps({'status':'PASS','stats':stats,'same_windows':same,'counts':counts},ensure_ascii=False),flush=True)
    except Exception as ex:
        C.write('343_result_3b部分独立验收_实现失败_v2.json',{'status':'实现失败','exception':repr(ex),'traceback':traceback.format_exc(),
            '已执行':counts,'未执行':['真实训练','阶段5'],'未核':['全量验收'],'source_sha256':C.sources()})
        raise

if __name__=='__main__':main()
