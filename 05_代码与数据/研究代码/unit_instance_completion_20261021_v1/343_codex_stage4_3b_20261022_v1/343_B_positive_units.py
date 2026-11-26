"""Positive B, duplicate/wrong-class/full-GT ordering, and displacement fixtures."""
import importlib
C=importlib.import_module('343_context');M=importlib.import_module('343_formal_counts_v2');G=M.G;F=M.F
import torch,numpy as np
from PIL import Image,ImageDraw

def fixture():
    raw=torch.tensor([[.8,.8,.01,.01]]).repeat(300,1);logits=torch.full((300,10),-12.);logits[0,0]=0.
    rc,_=G.select_raw_query_window(G.raw_query_cxcywh_to_640_xyxy(raw));w=G.window_xyxy(rc)
    gt=torch.tensor([[520.,520.,523.,523.]])
    return {'final300_boxes':raw,'final300_logits':logits,'orig_wh':[640,640],'window_xyxy_640':w,
        'gt_boxes_640':gt,'gt_labels':torch.tensor([0]),'gt_annotation_ids':[0],
        'gt_short_side_640_all':G.short_side(gt),'G_W':G.complete_mask(gt,w)}

def groups(e):
    return {'tiny_complete_Geo05':F.tiny_geo0_group(e).numpy(),'all_sizes_complete_Geo05':F.tiny_geo0_group(e,tiny_only=False).numpy()}

def main():
    rows=[];paths=[]
    for case in range(6):
        e=fixture();gt=e['gt_boxes_640'];labels=e['gt_labels'];aids=e['gt_annotation_ids']
        local=G.win_norm_cxcywh(gt,e['window_xyxy_640']).repeat(4,1);nl=torch.full((4,10),-12.);nl[0,0]=2.
        if case==1:nl[1,0]=1. # duplicate -> one unique B
        if case==2:nl[0,0]=-12.;nl[0,1]=2. # wrong-class new prediction
        if case==3:
            e['final300_boxes'][0]=G.full_plane_cxcywh(gt)[0];e['final300_logits'][0,0]=-12.;e['final300_logits'][0,1]=2.
        if case==4:
            e['final300_logits'].fill_(-12.)
            for q in range(300):e['final300_logits'][q,q%10]=4.
            e['final300_boxes'][299]=G.full_plane_cxcywh(gt)[0];e['gt_labels']=torch.tensor([9]);labels=e['gt_labels']
            e['final300_logits'][299,9]=3.;local.fill_(.4);nl.fill_(20.)
        if case==5:
            # Higher-IoU non-group GT is matched first; no subgroup-only matching.
            gt=torch.tensor([[520.,520.,523.,523.],[520.,520.,523.,522.7]])
            e['gt_boxes_640']=gt;e['gt_labels']=torch.tensor([0,0]);labels=e['gt_labels'];aids=[0,1];e['gt_annotation_ids']=aids
            e['gt_short_side_640_all']=G.short_side(gt);e['G_W']=G.complete_mask(gt,e['window_xyxy_640'])
            local=G.win_norm_cxcywh(gt[1:],e['window_xyxy_640']).repeat(4,1)
        base,_=F.fuse(e);out,pre=F.fuse(e,local,nl)
        own=M.match(out,gt,labels,aids,.5);ref=M.independent_match(out,gt,labels,aids,.5);assert np.array_equal(own,ref)
        bowner=M.match(base,gt,labels,aids,.5);grp=groups(e)
        if case==5:grp['tiny_complete_Geo05']=np.array([True,False]) # explicit group-order fixture
        result=M.metrics(out,own,base,bowner,grp,aids,pre)
        expected=[1,1,0,0,0,0][case];assert result['B']==expected,(case,result)
        if case==4:assert result['GT_lost_vs_global_baseline']==1 and result['new_candidate_instances_final_selected']==40
        if case==3:assert result['GT_added_vs_global_baseline']==1 and not grp['tiny_complete_Geo05'][0]
        rows.append({'case':case,'expected_B':expected,'metrics':result,'independent_owner_equal':True})
        image=Image.new('RGB',(1040,590),'#202020');d=ImageDraw.Draw(image)
        for x in range(0,513,32):d.line((x+4,60,x+4,572),fill='#404040');d.line((x+524,60,x+524,572),fill='#404040')
        for y in range(60,573,32):d.line((4,y,516,y),fill='#404040');d.line((524,y,1036,y),fill='#404040')
        target=G.win_norm_to_window_pixels(G.win_norm_xyxy(gt,e['window_xyxy_640']),e['window_xyxy_640'],(512,512)).numpy()
        for edge in target:d.rectangle(tuple(edge+[4,60,4,60]),outline='yellow',width=2)
        new=(out['flat_index']>=3000).nonzero().flatten().tolist()
        for j in new:
            edge=G.win_norm_to_window_pixels(G.win_norm_xyxy(out['boxes'][j:j+1],e['window_xyxy_640']),e['window_xyxy_640'],(512,512))[0].numpy()
            d.rectangle(tuple(edge+[524,60,524,60]),outline='lime' if own[j]>=0 else 'red',width=2)
        d.text((4,5),f'Constructed case={case}; B={result["B"]}; FP={result["FP"]}; original TP lost={result["GT_lost_vs_global_baseline"]}',fill='white')
        d.text((4,26),'Left: target GT; right: selected synthetic new instances; green TP / red FP; no head execution',fill='white')
        name=f'343_B阳性构造_{case}.png'
        with open(C.HERE/name,'xb') as f:image.save(f,format='PNG')
        paths.append(str(C.HERE/name))
    C.write('343_result_阶段4主B阳性构造.json',{'status':'已验证','cases':rows,'png_paths':paths,
        '已执行':{'constructed_cases':6,'fusion_calls':12,'primary_matches':12,'independent_matches':6,'PNG生成':6,'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
        '未执行':['真实训练','阶段5'],'未核':['主控独立复核'],'source_sha256':C.sources()})

if __name__=='__main__':main()
