"""Frozen counting mechanics and val-v2 synthetic end-to-end only."""
import importlib,json,time,copy
C=importlib.import_module('343_context');M=importlib.import_module('343_formal_counts');F=M.F;G=M.G
import torch,numpy as np
from PIL import Image,ImageDraw
torch.set_num_threads(2)

def fixture_out(boxes,labels,scores,flat):
    return {'boxes':torch.tensor(boxes,dtype=torch.float64).reshape(-1,4),'labels':torch.tensor(labels),
        'scores':torch.tensor(scores,dtype=torch.float64),'flat_index':torch.tensor(flat)}

def units():
    tests=[]
    def check(name,out,gt,lab,aid,threshold,expected):
        gt=torch.tensor(gt,dtype=torch.float64).reshape(-1,4);lab=torch.tensor(lab)
        a=M.match(out,gt,lab,aid,threshold);b=M.independent_match(out,gt,lab,aid,threshold)
        assert np.array_equal(a,b) and a.tolist()==expected
        tests.append({'case':name,'owner_GT_indices':a.tolist(),'threshold':threshold})
    o=fixture_out([[0,0,1,2]],[0],[.8],[7])
    check('IoU_exactly_.5_is_TP',o,[[0,0,2,2]],[0],[0],.5,[0])
    check('same_box_at_.75_is_FP',o,[[0,0,2,2]],[0],[0],.75,[-1])
    check('wrong_class_is_FP',o,[[0,0,1,2]],[1],[0],.5,[-1])
    check('empty_GT',o,[],[],[],.5,[-1])
    dup=fixture_out([[0,0,2,2],[0,0,2,2]],[0,0],[.9,.8],[3000,3010])
    check('duplicate_prediction_counts_unique_GT_once',dup,[[0,0,2,2]],[0],[0],.5,[0,-1])
    tie=fixture_out([[0,0,2,2],[0,0,2,2]],[0,0],[.9,.9],[7,2])
    check('score_tie_flat_index_and_IoU_tie_annotation_id',tie,[[0,0,2,2],[0,0,2,2]],[0,0],[9,0],.5,[0,1])
    steal=fixture_out([[0,0,2,1.8]],[0],[.9],[3000])
    check('match_all_GT_before_subgroup',steal,[[0,0,2,2],[0,0,2,1.8]],[0,0],[0,1],.5,[1])
    # Original GT xywh conversion independent and deliberate malformed width.
    xywh=torch.tensor([[1.,2.,3.,4.]],dtype=torch.float64)
    assert torch.equal(G.orig_xywh_to_orig_xyxy(xywh),torch.tensor([[1.,2.,4.,6.]],dtype=torch.float64))
    try:G.orig_xywh_to_orig_xyxy(torch.tensor([[1.,2.,-3.,4.]]))
    except AssertionError:tests.append({'case':'negative_GT_width_rejected','detected':True})
    else:raise AssertionError('negative GT width accepted')
    return tests

def picture(e,out,owner,gt,seed):
    src=C.Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val\images')/e['file_name']
    with Image.open(src) as im:pil=im.convert('RGB')
    a=Image.new('RGB',(1560,590),'white');d=ImageDraw.Draw(a)
    a.paste(pil.resize((512,512),Image.BILINEAR),(4,60))
    crop=Image.fromarray((e['crop128'].permute(1,2,0).numpy().clip(0,1)*255).round().astype('uint8')).resize((512,512),Image.NEAREST)
    a.paste(crop,(524,60));a.paste(crop,(1044,60));w=e['window_xyxy_640'];full=torch.tensor([0.,0.,640.,640.])
    edge=G.win_norm_to_window_pixels(G.win_norm_xyxy(w[None],full),full,(512,512))[0].numpy()
    d.rectangle(tuple(edge+[4,60,4,60]),outline='#00ff00',width=3)
    candidates=(out['flat_index']>=3000).nonzero().flatten().tolist()
    for j in candidates:
        edge=G.win_norm_to_window_pixels(G.win_norm_xyxy(G.orig_xyxy_to_plane640(out['boxes'][j:j+1],e['orig_wh']),w),w,(512,512))[0].numpy()
        color='lime' if owner[j]>=0 else 'red'
        d.rectangle(tuple(edge+[524,60,524,60]),outline=color,width=2)
        if owner[j]>=0:
            target=G.win_norm_to_window_pixels(G.win_norm_xyxy(G.orig_xyxy_to_plane640(gt[owner[j]:owner[j]+1],e['orig_wh']),w),w,(512,512))[0].numpy()
            d.rectangle(tuple(target+[1044,60,1044,60]),outline='yellow',width=3)
    d.text((4,5),f'seed={seed} image={e["image_id"]}; RANDOM SYNTHETIC new4; selected instances={len(candidates)}; no head executed',fill='black')
    d.text((4,27),'Original / frozen W',fill='black');d.text((524,27),'New predictions: green TP / red FP (all-GT IoU.5)',fill='black');d.text((1044,27),'Matched original GT for selected new predictions',fill='black')
    name=f'343_正式计数合成_seed{seed}_{e["image_id"]:06d}.png'
    with open(C.HERE/name,'xb') as f:a.save(f,format='PNG')
    return str(C.HERE/name)

def main():
    start=time.perf_counter();tests=units()
    ann=C.Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val\annotations\val.json')
    data=json.loads(ann.read_text(encoding='utf-8'));by={}
    for a in data['annotations']:by.setdefault(a['image_id'],[]).append(a)
    results={};summaries={};pngs=[];maxgterr=0.;independent_calls=0;low_equal=True
    for seed in (1,2):
        rng=np.random.default_rng(2026102207+seed);rows=[]
        for iid in range(548):
            e=torch.load(C.S3/f'342_cache_val_v2_seed{seed}'/f'342_{iid:06d}.pt',map_location='cpu',weights_only=True)
            items=by.get(iid,[]);aids=[a['id'] for a in items];assert aids==e['gt_annotation_ids']
            gt=G.orig_xywh_to_orig_xyxy(torch.tensor([a['bbox'] for a in items],dtype=torch.float64).reshape(-1,4))
            ref=np.array([[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in items]).reshape(-1,4)
            err=float(np.max(np.abs(gt.numpy()-ref),initial=0));assert err==0;maxgterr=max(maxgterr,err)
            local=torch.tensor(rng.uniform(.05,.95,(4,4)),dtype=torch.float32)
            outs={};pres={}
            outs['baseline'],pres['baseline']=F.fuse(e)
            outs['low_new4'],pres['low_new4']=F.fuse(e,local,torch.full((4,10),-12.))
            outs['random_new4'],pres['random_new4']=F.fuse(e,local,torch.tensor(rng.normal(0,2,(4,10)),dtype=torch.float32))
            raw=G.raw_query_cxcywh_to_640_xyxy(e['final300_boxes'])
            mx=G.pairwise_iou(raw,e['gt_boxes_640'],allow_raw_a=True).max(0).values.numpy();gw=e['G_W'].numpy();tiny=e['gt_short_side_640_all'].numpy()<4
            groups={'tiny_complete_Geo05':(mx<.5)&gw&tiny,'all_image_Geo05':mx<.5,
                'all_sizes_complete_Geo05':(mx<.5)&gw,'tiny_complete_Geo075':(mx<.75)&gw&tiny}
            one={'seed':seed,'image_id':iid,'groups':{k:int(v.sum()) for k,v in groups.items()},'cases':{}}
            owners={}
            for threshold in (.5,.75):
                own={}
                for mode,out in outs.items():
                    own[mode]=M.match(out,gt,e['gt_labels'],aids,threshold)
                    independent=M.independent_match(out,gt,e['gt_labels'],aids,threshold);independent_calls+=1
                    assert np.array_equal(own[mode],independent),(seed,iid,mode,threshold)
                owners[threshold]=own
                for mode,out in outs.items():
                    result=M.metrics(out,own[mode],outs['baseline'],own['baseline'],groups,aids,pres[mode])
                    one['cases'].setdefault(mode,{})[str(threshold)]=result
                for key in ('TP','FP','B','matched_annotation_ids','GT_added_vs_global_baseline','GT_lost_vs_global_baseline'):
                    assert one['cases']['baseline'][str(threshold)][key]==one['cases']['low_new4'][str(threshold)][key],(seed,iid,key)
            if iid in (0,17,534 if seed==1 else 210):pngs.append(picture(e,outs['random_new4'],owners[.5]['random_new4'],gt,seed))
            rows.append(one)
            if (iid+1)%128==0:print(f'formal seed={seed} {iid+1}/548',flush=True)
        results[str(seed)]=rows;summaries[str(seed)]={}
        for mode in ('baseline','low_new4','random_new4'):
            summaries[str(seed)][mode]={}
            for threshold in (.5,.75):
                metrics=[r['cases'][mode][str(threshold)] for r in rows]
                keys=[k for k,v in metrics[0].items() if isinstance(v,int)]
                summaries[str(seed)][mode][str(threshold)]={k:sum(m[k] for m in metrics) for k in keys}
        print(f'formal seed={seed}: independent matching passed',flush=True)
    C.write('343_result_阶段4正式计数.json',{'status':'已验证（合成新框工程测试）','constructed_tests':tests,'per_seed':results,'summary':summaries,
        'GT_original_xywh_independent_max_abs_diff':maxgterr,'low_new4_formal_counts_equal_baseline_both_seeds':low_equal,
        'png_paths':pngs,'policy_record':str(C.HERE/'343_追加放行与计数口径.json'),
        'policy_supplement':str(C.HERE/'343_计数口径补充冻结.json'),
        'B075_note':'0.75栏B是固定主Geo0(.5)资格组在IoU.75下的新增TP；tiny_complete_Geo075另列，主B仍只取0.5栏。',
        'candidate_baseline_note':'baseline未提供new4；其new_candidate_instances_final_unselected等于40仅为统一40候选槽位计数，不代表执行了新头。',
        '已执行':{'cache_loads':1096,'fusion_calls':3288,'natural_primary_matches':6576,'natural_independent_matches':independent_calls,
            'constructed_matching_tests':7,'negative_format_tests':1,'PNG生成':6,'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
        '未执行':['八run训练后科学指标','真实训练','GPU','阶段5'],'未核':['主控独立核对'],'wall_seconds':time.perf_counter()-start,'source_sha256':C.sources()})

if __name__=='__main__':main()
