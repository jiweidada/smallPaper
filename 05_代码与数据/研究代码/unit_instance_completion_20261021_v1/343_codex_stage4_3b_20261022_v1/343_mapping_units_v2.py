"""Both seeds: independent mapping, round trips, reverse cases, six PNGs."""
import importlib,json
C=importlib.import_module('343_context')
import torch,numpy as np
from PIL import Image,ImageDraw
F=importlib.import_module('343_fusion');G=F.G
torch.set_num_threads(2)

def independent(local,w,wh):
    result=[]
    for cx,cy,bw,bh in local:
        ox,oy,ex,ey=map(float,w);ow,oh=wh
        center_x=ox+float(cx)*(ex-ox);center_y=oy+float(cy)*(ey-oy)
        half_x=float(bw)*(ex-ox)/2;half_y=float(bh)*(ey-oy)/2
        result.append([(center_x-half_x)*ow/640,(center_y-half_y)*oh/640,
                       (center_x+half_x)*ow/640,(center_y+half_y)*oh/640])
    return np.array(result,dtype=np.float64)

def main():
    rng=np.random.default_rng(2026102204);rows=[];pngs=[];bad_reject=0;bad_diff=0;reordered=[]
    from engine.deim.postprocessor import PostProcessor
    pp=PostProcessor(num_classes=10,num_top_queries=300,use_focal_loss=True,score_threshold=.001)
    low=torch.full((4,10),-12.,dtype=torch.float32)
    for seed in (1,2):
        for iid in range(548):
            path=C.S3/f'342_cache_val_v2_seed{seed}'/f'342_{iid:06d}.pt'
            e=torch.load(path,map_location='cpu',weights_only=True)
            local=torch.tensor(rng.uniform(.05,.95,(4,4)),dtype=torch.float64)
            got=G.local_cxcywh_to_orig_xyxy(local,e['window_xyxy_640'],e['orig_wh'])
            ref=independent(local.numpy(),e['window_xyxy_640'].numpy(),e['orig_wh'])
            err=float(np.abs(got.numpy()-ref).max());assert err<1e-9
            back=G.orig_xyxy_to_local_cxcywh(got,e['window_xyxy_640'],e['orig_wh'])
            cycle=float((local-back).abs().max());assert cycle<1e-12
            try:
                wrong=G.plane640_to_orig_xyxy(G.win_norm_xyxy_to_640(local,e['window_xyxy_640']),e['orig_wh'])
            except AssertionError:bad_reject+=1
            else:
                assert float((wrong-got).abs().max())>1e-3;bad_diff+=1
            try:G.local_cxcywh_to_orig_xyxy(local*64,e['window_xyxy_640'],e['orig_wh'])
            except AssertionError:pass
            else:raise AssertionError('pixel-scale local input accepted')
            o,_=F.fuse(e);lo,_=F.fuse(e,local.float(),low)
            same_order=all(torch.equal(o[k],lo[k]) for k in o)
            oi=torch.argsort(o['flat_index']);li=torch.argsort(lo['flat_index'])
            assert all(torch.equal(o[k][oi],lo[k][li]) for k in o)
            if not same_order:reordered.append({'seed':seed,'image_id':iid})
            with torch.no_grad():native=pp({'pred_boxes':e['final300_boxes'][None],
                'pred_logits':e['final300_logits'][None]},torch.tensor([e['orig_wh']],dtype=torch.float32))[0]
            counts={};keep=[]
            for j,lab in enumerate(native['labels']):
                c=int(lab)
                if counts.get(c,0)<100:keep.append(j);counts[c]=counts.get(c,0)+1
            for k in ('boxes','scores','labels'):assert torch.equal(native[k][keep],o[k])
            rows.append({'seed':seed,'image_id':iid,'independent_max_abs_diff':err,'round_trip_max_abs_diff':cycle,
                         'global_native_PP_bit_equal':True,'low_new4_prediction_set_bit_equal':True,'low_new4_order_bit_equal':same_order})
            if iid in (0,174,534):
                src=C.Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val\images')/e['file_name']
                with Image.open(src) as im:pil=im.convert('RGB')
                canvas=Image.new('RGB',(1536,570),'white');d=ImageDraw.Draw(canvas)
                thumb=pil.resize((768,512),Image.BILINEAR);canvas.paste(thumb,(0,48));canvas.paste(thumb,(768,48))
                ow,oh=e['orig_wh']
                # Both displays use shared original->640->display scale; the
                # independent numeric edges above are used for the right panel.
                a=G.orig_xyxy_to_plane640(got,e['orig_wh'])
                a=G.win_norm_to_window_pixels(G.win_norm_xyxy(a,torch.tensor([0.,0.,640.,640.])),
                    torch.tensor([0.,0.,640.,640.]),(768,512)).numpy()
                b=ref*np.array([768/ow,512/oh,768/ow,512/oh])
                for edge in a:d.rectangle(tuple(edge+[0,48,0,48]),outline='red',width=2)
                for edge in b:d.rectangle(tuple(edge+[768,48,768,48]),outline='red',width=2)
                d.text((8,8),f'seed={seed}; image={iid}; left shared inverse; right independent center/extent construction',fill='black')
                name=f'343_映射_v2_seed{seed}_{iid:06d}.png'
                with open(C.HERE/name,'xb') as f:canvas.save(f,format='PNG')
                pngs.append(str(C.HERE/name))
        print(f'mapping seed={seed}:548 checked',flush=True)
    # Explicit non-square/truncated fixture, preserving actual width/height.
    w=torch.tensor([600.,624.,640.,640.]);local=torch.tensor([[.2,.7,.5,.4]],dtype=torch.float64)
    got=G.local_cxcywh_to_orig_xyxy(local,w,[1920,721])
    assert np.abs(got.numpy()-independent(local.numpy(),w.numpy(),[1920,721])).max()<1e-9
    C.write('343_result_阶段4映射与融合_v2.json',{'status':'已验证','per_image':rows,'png_paths':pngs,
        'reverse':{'cxcywh_as_xyxy_rejected':bad_reject,'numerically_different':bad_diff,'pixel_scale_rejected':1096},
        'max_independent_abs_diff':max(r['independent_max_abs_diff'] for r in rows),
        'max_round_trip_abs_diff':max(r['round_trip_max_abs_diff'] for r in rows),
        '已执行':{'val_v2_cache_loads':1096,'mapping_tests':1096,'round_trip_tests':1096,'native_PostProcessor_calls':1096,
                   'synthetic_fusion_calls':2192,'PNG生成':6,'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
        '未执行':['训练产物评估','真实训练','GPU','阶段5'],'未核':['主控独立核对'],
        'native_topk_tie_reordered_images':reordered,'low_new4_COCO_metric_contract':'未执行；单独在全量评估中核对','source_sha256':C.sources()})

if __name__=='__main__':main()
