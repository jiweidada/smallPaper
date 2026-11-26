"""Unambiguous 317 eligibility only. Formal added-TP matching awaits main control."""
import importlib,json,copy,time
C=importlib.import_module('343_context')
import torch,numpy as np
from PIL import Image,ImageDraw
F=importlib.import_module('343_fusion');G=F.G
I=importlib.import_module('342_val_independent')
torch.set_num_threads(2)

def independent(raw,gt,gw,short):
    # Distinct NumPy, per-GT maximum over all 300 raw boxes of ANY class.
    q=np.asarray(raw,dtype=np.float32)*np.float32(640)
    edges=np.empty_like(q)
    edges[:,0]=q[:,0]-q[:,2]/np.float32(2);edges[:,2]=q[:,0]+q[:,2]/np.float32(2)
    edges[:,1]=q[:,1]-q[:,3]/np.float32(2);edges[:,3]=q[:,1]+q[:,3]/np.float32(2)
    area=np.maximum(edges[:,2]-edges[:,0],0)*np.maximum(edges[:,3]-edges[:,1],0)
    maxima=[]
    for a,b,c,d in np.asarray(gt,dtype=np.float32):
        ix=np.maximum(0,np.minimum(edges[:,2],c)-np.maximum(edges[:,0],a))
        iy=np.maximum(0,np.minimum(edges[:,3],d)-np.maximum(edges[:,1],b))
        inter=ix*iy;den=np.maximum(area+(c-a)*(d-b)-inter,np.float32(1e-12))
        maxima.append(float(np.max(inter/den)))
    maxima=np.asarray(maxima)
    allgeo=np.asarray(gw,dtype=bool)&(maxima<.5)
    tiny=allgeo&np.asarray([float(x)<4 for x in short])
    return tiny,allgeo,maxima

def png(e,tiny,seed):
    path=C.Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val\images')/e['file_name']
    with Image.open(path) as im:src=im.convert('RGB')
    canvas=Image.new('RGB',(1400,760),'white');d=ImageDraw.Draw(canvas)
    canvas.paste(src.resize((700,700),Image.BILINEAR),(0,50));canvas.paste(src.resize((700,700),Image.BILINEAR),(700,50))
    full=torch.tensor([0.,0.,640.,640.])
    scaled=G.win_norm_to_window_pixels(G.win_norm_xyxy(e['gt_boxes_640'],full),full,(700,700)).numpy()
    w=G.win_norm_to_window_pixels(G.win_norm_xyxy(e['window_xyxy_640'][None],full),full,(700,700))[0].numpy()
    for offset in (0,700):d.rectangle(tuple(w+[offset,50,offset,50]),outline='#00ff00',width=2)
    for j,edge in enumerate(scaled):
        if bool(e['G_W'][j]):d.rectangle(tuple(edge+[0,50,0,50]),outline='yellow',width=1)
        if tiny[j]:
            d.rectangle(tuple(edge+[700,50,700,50]),outline='red',width=3)
            d.text((float(edge[0])+700,float(edge[1])+50),str(e['gt_annotation_ids'][j]),fill='red')
    d.text((5,5),f'seed={seed} image={e["image_id"]}; left yellow=complete GT; right red=tiny Geo0 eligibility (NOT added TP)',fill='black')
    d.text((5,25),f'G_W={int(e["G_W"].sum())}; tiny Geo0={int(tiny.sum())}; tiny from gt_short_side_640_all; green=frozen W',fill='black')
    name=f'343_Geo0资格_seed{seed}_{e["image_id"]:06d}.png'
    with open(C.HERE/name,'xb') as f:canvas.save(f,format='PNG')
    return str(C.HERE/name)

def main():
    start=time.perf_counter();rows=[];short_reverse=0;maxdiff=0.;stats={};chosen={};paths=[]
    for seed in (1,2):
        one=[]
        for iid in range(548):
            e=torch.load(C.S3/f'342_cache_val_v2_seed{seed}'/f'342_{iid:06d}.pt',map_location='cpu',weights_only=True)
            assert len(e['gt_annotation_ids'])==len(e['gt_short_side_640_all'])==len(e['gt_labels'])
            ref,allref,mx=independent(e['final300_boxes'].numpy(),e['gt_boxes_640'].numpy(),e['G_W'].numpy(),e['gt_short_side_640_all'].numpy())
            got=F.tiny_geo0_group(e).numpy();allgot=F.tiny_geo0_group(e,tiny_only=False).numpy()
            assert np.array_equal(got,ref) and np.array_equal(allgot,allref)
            raw=G.raw_query_cxcywh_to_640_xyxy(e['final300_boxes'])
            actual=G.pairwise_iou(raw,e['gt_boxes_640'],allow_raw_a=True).max(0).values.numpy()
            diff=float(np.max(np.abs(mx-actual),initial=0));maxdiff=max(maxdiff,diff)
            bad=dict(e);bad['gt_short_side_640_all']=e['gt_short_side_640']
            if len(bad['gt_short_side_640_all'])!=len(e['gt_annotation_ids']):
                try:F.tiny_geo0_group(bad)
                except AssertionError:short_reverse+=1
                else:raise AssertionError('G_W-only short-side vector accepted as all-GT')
            eligible=[aid for aid,yes in zip(e['gt_annotation_ids'],got) if yes]
            row={'seed':seed,'image_id':iid,'tiny_Geo0_count':len(eligible),'tiny_Geo0_annotation_ids':eligible,
                'all_GT_Geo0_complete_window_count':int(allgot.sum()),'independent_IoU_max_abs_diff':diff}
            rows.append(row);one.append(row)
        order=sorted(one,key=lambda r:(-r['tiny_Geo0_count'],r['image_id']))
        ids=[r['image_id'] for r in order[:2]]+[534 if seed==1 else 210]
        if len(set(ids))<3:ids=[r['image_id'] for r in order[:3]]
        chosen[str(seed)]=ids
        for iid in ids:
            e=torch.load(C.S3/f'342_cache_val_v2_seed{seed}'/f'342_{iid:06d}.pt',map_location='cpu',weights_only=True)
            paths.append(png(e,F.tiny_geo0_group(e).numpy(),seed))
        stats[str(seed)]={'tiny_Geo0_eligible_GT':sum(r['tiny_Geo0_count'] for r in one),
            'images_with_tiny_Geo0':sum(r['tiny_Geo0_count']>0 for r in one),
            'all_GT_Geo0_complete_window':sum(r['all_GT_Geo0_complete_window_count'] for r in one)}
        print(f'Geo0 seed={seed}:548 independently checked',flush=True)
    # Explicit format reversal, and .5 equality does not qualify as Geo0.
    raw=torch.tensor([[.3,.6,.1,.05]]).repeat(300,1)
    edges=G.raw_query_cxcywh_to_640_xyxy(raw)
    correct=G.pairwise_iou(edges,edges[:1],allow_raw_a=True)
    wrong=G.pairwise_iou(raw*640,edges[:1],allow_raw_a=True)
    assert correct.max()==1 and wrong.max()<.5
    C.write('343_result_阶段4Geo0资格.json',{'status':'已验证（资格组，非正式新增TP）','stats':stats,'per_image':rows,
        'max_independent_IoU_abs_diff':maxdiff,'PNG_ids':chosen,'png_paths':paths,
        'reverse_tests':{'G_W_only_short_side_vector_rejected':short_reverse,'raw_cxcywh_as_xyxy_detected':True},
        'definition':'G_W & 全300 raw任意类最大IoU<.5 & gt_short_side_640_all<4；annotation_ids逐一对应；冻结1e-6完整窗容差保留。',
        '待主控':['正式B新增覆盖的参照及TP匹配顺序','全GT Geo0正式TP、IoU.75、FP、原TP新增/丢失口径'],
        '已执行':{'cache_loads':1102,'independent_eligibility_checks':1096,'PNG生成':6,'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
        '未执行':['正式B计数','真实训练','阶段5'],'未核':['主控独立核对'],'wall_seconds':time.perf_counter()-start,
        'source_sha256':C.sources()})

if __name__=='__main__':main()
