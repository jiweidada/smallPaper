"""Independent scalar/NumPy geometry, exact certificate, negative controls, six PNGs."""
import importlib,json,time,traceback
C=importlib.import_module('344_context')
import torch,numpy as np
from PIL import Image,ImageDraw
I=importlib.import_module('342_val_independent');V=importlib.import_module('342_validate_val_v3')
Q=importlib.import_module('343_validate_development_v2');B=importlib.import_module('342_build_val_v3')
G=C.load_geometry();torch.set_num_threads(2)
COUNT={'cache_loads':0,'old340_loads':0,'independent_cost_matrices':0,'independent_exact_certificates':0,'independent_windows':0,
    'independent_G_U_K':0,'image_decodes':0,'independent_crops':0,'PNG_writes':0,'negative_controls':0,
    'global_forwards':0,'head_forwards':0,'optimizer_steps':0}

def png(e,pixels,ref,keep,bad):
    a=Image.new('RGB',(1560,606),'white');d=ImageDraw.Draw(a)
    a.paste(Image.fromarray(pixels).resize((512,512),Image.Resampling.BILINEAR),(4,60))
    for off,arr in ((524,e['crop128'].numpy()),(1044,ref)):
        a.paste(Image.fromarray((arr.transpose(1,2,0).clip(0,1)*255).round().astype('uint8')).resize((512,512),Image.Resampling.NEAREST),(off,60))
    full=torch.tensor([0.,0.,640.,640.]);w=e['window_xyxy_640']
    wb=G.win_norm_to_window_pixels(G.win_norm_xyxy(w[None],full),full,(512,512))[0].numpy()
    d.rectangle(tuple(wb+[4,60,4,60]),outline='#00ff00',width=3)
    boxes=G.win_norm_to_window_pixels(G.win_norm_xyxy(e['fullp_boxes'][keep],w),w,(512,512)).numpy()
    for b in boxes:d.rectangle(tuple(np.clip(b,0,511)+[524,60,524,60]),outline='cyan',width=1)
    for b in G.win_norm_to_window_pixels(G.win_norm_xyxy(e['gt_boxes_640'][e['G_W']],w),w,(512,512)).numpy():
        d.rectangle(tuple(b+[524,60,524,60]),outline='yellow',width=3)
    for xy,m in zip(G.grid_xy().numpy()*512,e['K_g'].sum(0).numpy()):
        if m>0:
            x,y=xy+[524,60];d.ellipse((x-3,y-3,x+3,y+3),outline='red',width=2)
    # Native signed raw edges are drawn in their given order, never repaired.
    signed=G.global_queries_to_orig_xyxy(e['final300_boxes'],(512,512))
    for j in bad.nonzero().flatten().tolist():
        x0,y0,x1,y1=signed[e['fullp_query_index'][j]].tolist()
        d.line([(x0+4,y0+60),(x1+4,y0+60),(x1+4,y1+60),(x0+4,y1+60),(x0+4,y0+60)],fill='magenta',width=4)
        d.text((x0+4,y0+70),'raw h<=0; retained fullP; excluded P_W',fill='magenta')
    d.text((4,6),f'344 seed={e["seed"]} id={e["image_id"]} W={w.tolist()}; G_W={e["n_G_W"]} U_W={e["n_U_W"]} P_W={int(keep.sum())}; nonpositive={int(bad.sum())}',fill='black')
    d.text((4,30),'Original / frozen W / signed bad raw (magenta)',fill='black')
    d.text((524,30),'Stored crop / valid P_W / complete GT / K_g',fill='black')
    d.text((1044,30),'Independent original-pixel bilinear crop',fill='black')
    d.text((4,584),'Only visual drawing is clipped. Stored coordinates remain native and unclipped.',fill='black')
    path=C.HERE/f'344_geometry_seed{e["seed"]}_{e["image_id"]:06d}.png'
    with open(path,'xb') as f:a.save(f,format='PNG')
    COUNT['PNG_writes']+=1;return str(path)

def main():
    start=time.perf_counter();manifest=json.loads((C.HERE/'344_development6470_逐条SHA清单.json').read_text(encoding='utf-8'))
    rows={(r['seed'],r['image_id']):r for r in manifest['entries']};ids=C.development_ids()
    stats=json.loads((C.S343/'343_result_3b已生成6469条独立验收.json').read_text(encoding='utf-8'))
    picks=[]
    for seed in (1,2):
        rich=max(stats['per_image'][str(seed)],key=lambda r:r['G_W'])['image_id']
        chosen=[ids[0],rich,1466 if seed==2 else ids[1]]
        assert len(set(chosen))==3;picks.extend((seed,i) for i in chosen)
    result=[];pngs=[];reverse=[]
    try:
        for seed,iid in picks:
            row=rows[seed,iid];assert C.sha(row['path'])==row['file_sha256']
            e=torch.load(row['path'],map_location='cpu',weights_only=True);COUNT['cache_loads']+=1
            assert B.payload_sha(e)==row['payload_sha256']
            pb=e['final300_boxes'].numpy();query=e['fullp_query_index'].numpy();boxes=e['fullp_boxes'].numpy();w=e['window_xyxy_640'].numpy()
            # Scalar raw sign and rectangle intersection, independent from Torch helpers.
            bad=np.array([float(pb[q,2])<=0 or float(pb[q,3])<=0 for q in query],dtype=bool)
            keep=np.array([not bad[j] and min(float(b[2]),float(w[2]))>max(float(b[0]),float(w[0])) and
                min(float(b[3]),float(w[3]))>max(float(b[1]),float(w[1])) for j,b in enumerate(boxes)],dtype=bool)
            actual_bad=G.fullp_nonpositive_mask(e['final300_boxes'],e['fullp_query_index'])
            actual_keep=G.intersecting_fullp_candidates(e['fullp_boxes'],e['window_xyxy_640'],actual_bad)
            assert np.array_equal(bad,actual_bad.numpy()) and np.array_equal(keep,actual_keep.numpy())
            edges=I.raw640_boxes(pb);rawdiff=float(np.max(np.abs(edges[query]-boxes),initial=0));assert rawdiff<8*np.finfo(np.float32).eps*640
            rc,ww=V.independent_window_640(G.raw_query_cxcywh_to_640_xyxy(e['final300_boxes']).numpy())
            assert list(rc)==e['window_cell'] and np.array_equal(w,ww);COUNT['independent_windows']+=1
            local=np.array([[(float(b[k])-float(w[k%2]))/float(w[k%2+2]-w[k%2]) for k in range(4)] for b in boxes[keep]]).reshape(-1,4)
            g_local=G.win_norm_xyxy(e['fullp_boxes'][actual_keep],e['window_xyxy_640']).numpy()
            locdiff=float(np.max(np.abs(local-g_local),initial=0));assert locdiff<1e-6
            good=~bad;cb=e['m0_cand_boxes_norm'].numpy();gb=e['m0_gt_boxes_norm'].numpy()
            ii=np.zeros_like(e['m0_native_iou'].numpy());ii[good]=V.independent_native_iou(cb[good],gb)
            legal=(e['fullp_labels'].numpy()[:,None]==e['gt_labels'].numpy()[None,:])&(ii>=.5)&good[:,None]
            assert np.array_equal(legal,e['m0_legal_mask'].numpy())
            cc=Q.independent_cost(cb[good],gb,e['fullp_logits_row'].numpy()[good],e['gt_labels'].numpy())
            costdiff=float(np.max(np.abs(cc-e['m0_cost_matrix'].numpy()[good]),initial=0));assert costdiff<1e-3
            assert not e['m0_legal_mask'][actual_bad].any() and (e['m0_native_iou'][actual_bad]==0).all()
            cert=I.certificate(e['m0_cost_matrix'].numpy(),legal,e['fullp_flatten_index'].tolist(),V.decode(e['m0_exact']))
            COUNT['independent_cost_matrices']+=1;COUNT['independent_exact_certificates']+=1
            gw=I.complete(e['gt_boxes_640'].numpy(),w);owner=e['m0_cand_owner'].numpy();explained=np.zeros(len(gw),dtype=bool)
            explained[owner[owner>=0]]=True;uw=gw&~explained
            assert np.array_equal(gw,e['G_W'].numpy()) and np.array_equal(uw,e['U_W'].numpy())
            kg=I.kg(e['gt_boxes_640'].numpy(),gw,w);kgdiff=float(np.max(np.abs(kg-e['K_g'].numpy()),initial=0));assert kgdiff<1e-5
            COUNT['independent_G_U_K']+=1
            src=C.TRAIN/'images'/e['file_name'];assert C.sha(src)==row['image_sha256']
            with Image.open(src) as f:pixels=np.asarray(f.convert('RGB'))
            COUNT['image_decodes']+=1;crop=I.crop(pixels,w);COUNT['independent_crops']+=1
            cropdiff=float(np.max(np.abs(crop-e['crop128'].numpy())));assert cropdiff<I.crop_error_budget(*e['orig_wh'])
            pngs.append(png(e,pixels,crop,actual_keep,actual_bad))
            if (seed,iid)==(2,1466):
                assert np.flatnonzero(bad).tolist()==[259] and query[259]==62 and e['n_G_W']==e['n_U_W']==0
                assert owner[259]==-1 and not keep[259] and e['fullp_nonpositive_mask'][259]
                assert not e['m0_cost_row_evaluated_mask'][259]
                old=torch.load(C.WORK/'340_cache_seed2/001466.pt',map_location='cpu',weights_only=True);COUNT['old340_loads']+=1
                for k in ('G_W','U_W','K_g','fullp_boxes','fullp_scores','fullp_labels','crop128'):assert torch.equal(e[k],old[k])
                # Wrong mask, NaN, wrong representation, illegal owner must be detected.
                tests={
                    'nonpositive_mask_omitted':lambda:G.intersecting_fullp_candidates(e['fullp_boxes'],e['window_xyxy_640'],torch.zeros_like(actual_bad)),
                    'positive_box_in_exception':lambda:G.intersecting_fullp_candidates(torch.tensor([[0.,0.,1.,1.]]),e['window_xyxy_640'],torch.tensor([True])),
                    'nonfinite_raw':lambda:G.fullp_nonpositive_mask(torch.full((300,4),float('nan')),e['fullp_query_index']),
                    'xyxy_fed_as_raw':lambda:G.fullp_m0_matrices(G.raw_query_cxcywh_to_640_xyxy(e['final300_boxes']),e['fullp_query_index'],e['fullp_boxes'],e['fullp_logits_row'],e['m0_gt_boxes_norm'],e['gt_labels'],e['fullp_labels']),
                }
                for name,fn in tests.items():
                    try:fn()
                    except (AssertionError,RuntimeError):reverse.append({'test':name,'rejected':True});COUNT['negative_controls']+=1
                    else:raise AssertionError('negative control accepted: '+name)
                corrupted=V.decode(e['m0_exact']);corrupted['cand_owner']=list(corrupted['cand_owner']);corrupted['cand_owner'][259]=0
                try:I.certificate(e['m0_cost_matrix'].numpy(),legal,e['fullp_flatten_index'].tolist(),corrupted)
                except (AssertionError,KeyError):reverse.append({'test':'illegal_nonpositive_owner_certificate','rejected':True});COUNT['negative_controls']+=1
                else:raise AssertionError('corrupted certificate accepted')
            result.append({'seed':seed,'image_id':iid,'fullp_nonpositive_indices':np.flatnonzero(bad).tolist(),'P_W':int(keep.sum()),
                'raw_geometry_max_abs_diff':rawdiff,'local_geometry_max_abs_diff':locdiff,'cost_FP64_max_abs_diff':costdiff,
                'K_g_max_abs_diff':kgdiff,'crop_max_abs_diff':cropdiff,'exact_certificate':cert,'file_sha256':row['file_sha256']})
        assert len(pngs)==6
        C.write('344_result_1466独立验收与6PNG.json',{'status':'已验证（独立实现及反向测试）；PNG待本轮目视记录',
            'per_case':result,'negative_controls':reverse,'png_paths':pngs,'已执行':COUNT,'wall_seconds':time.perf_counter()-start,
            '未执行':['真实训练','GPU','阶段5'],'未核':['PNG本轮目视（另记）','主控独立复核'],'source_sha256':C.sources()})
        print(json.dumps({'status':'PASS','counts':COUNT,'png_paths':pngs},ensure_ascii=False),flush=True)
    except Exception as ex:
        C.write('344_result_1466独立验收_实现失败.json',{'status':'实现失败','exception':repr(ex),'traceback':traceback.format_exc(),
            '已执行':COUNT,'未执行':['真实训练','GPU','阶段5'],'未核':['1466独立验收'],'source_sha256':C.sources()});raise

if __name__=='__main__':main()
