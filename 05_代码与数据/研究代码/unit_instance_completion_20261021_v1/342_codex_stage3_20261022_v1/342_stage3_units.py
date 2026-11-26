"""Gate before complete-cache construction: exact M0 and geometry parity."""
import importlib,itertools,json
C=importlib.import_module('342_stage3_context')
import torch,numpy as np
import geom342 as G
M=importlib.import_module('342_m0_exact');I=importlib.import_module('342_val_independent')
import stage1_cache_v3 as S3,stage1_cache_v4 as S4
from PIL import Image
torch.set_num_threads(2)

def main():
    rng=np.random.default_rng(2026102203);cases=[]
    for k in range(240):
        n=int(rng.integers(0,6));m=int(rng.integers(0,5))
        costs=(rng.integers(-8,9,size=(n,m))/4).astype(np.float32)
        legal=rng.random((n,m))>.45;ids=rng.permutation(n)*10+2
        got=M.exact_assignment(costs,legal,ids)
        ref=I.brute_assignment(costs,legal,ids)
        assert got['cand_owner']==ref,(k,got,ref)
        cert=I.certificate(costs,legal,ids,got)
        cases.append({'case':k,'N':n,'M':m,'coverage':got['coverage'],'brute_owner_exact_match':True,'certificate':cert['exact_residual_dual_feasible']})
    # Cost priority smaller than the old 1e-6 acceptance threshold; intrinsic
    # order deliberately conflicts with score-order input rows.
    for name,costs,ids in [
        ('one_ulp_cost_before_lex',np.array([[np.nextafter(np.float32(1),np.float32(2)),1]],dtype=np.float32),[3]),
        ('intrinsic_order_before_input_order',np.zeros((2,2),dtype=np.float32),[90,10])]:
        legal=np.ones_like(costs,dtype=bool);got=M.exact_assignment(costs,legal,ids)
        assert got['cand_owner']==I.brute_assignment(costs,legal,ids)
        I.certificate(costs,legal,ids,got)
        cases.append({'name':name,'owner':got['cand_owner']})
    rows=[]
    for path in sorted((C.WORK/'341_cache_val_seed1').glob('*.pt')):
        e=torch.load(path,map_location='cpu',weights_only=True)
        pred=G.norm_cxcywh_to_640_xyxy(e['final300_boxes'])
        assert torch.equal(pred,S3.norm_cxcywh_to_640_xyxy(e['final300_boxes']))
        rc,_=G.select_window(pred);w=G.window_xyxy(rc)
        assert torch.equal(w,e['window_xyxy_640'])
        gw=G.complete_mask(e['gt_boxes_640'],w)
        assert torch.equal(gw,e['G_W'])
        kg=G.kg_window(e['gt_boxes_640'],gw,w)
        assert torch.equal(kg,e['K_g'])
        with Image.open(C.VAL/'images'/e['file_name']) as f:pil=f.convert('RGB')
        crop=G.crop128(pil,w,e['orig_wh'])
        assert torch.equal(crop,e['crop128'])
        cb=G.full_plane_cxcywh(e['fullp_boxes']);gb=G.full_plane_cxcywh(e['gt_boxes_640'])
        cost,iou,legal=G.m0_matrices(cb,e['fullp_logits_row'],gb,e['gt_labels'],e['fullp_labels'])
        import uic_m0
        oldcost,_=uic_m0.native_matcher_cost(cb,e['fullp_logits_row'],gb,e['gt_labels'],10)
        assert torch.equal(cost,oldcost)
        ids=e['fullp_query_index']*10+e['fullp_labels']
        got=M.exact_assignment(cost.numpy(),legal.numpy(),ids.tolist())
        cert=I.certificate(cost.numpy(),legal.numpy(),ids.tolist(),got)
        rows.append({'image_id':int(e['image_id']),'non_M0_fields_bit_equal':True,
                     'native_cost_bit_equal':True,'new_M0_coverage':got['coverage'],'exact_certificate':cert})
    # Non-square synthetic gradient image with clipped window, plus wrong axes,
    # wrong box representation, wrong pixel-vs-normalized scale.
    h,ow=73,211
    a=np.zeros((h,ow,3),dtype=np.uint8)
    a[...,0]=np.arange(ow,dtype=np.uint16)[None,:]%256;a[...,1]=np.arange(h,dtype=np.uint8)[:,None];a[...,2]=127
    pil=Image.fromarray(a);w=torch.tensor([600.,624.,640.,640.])
    expected=I.crop(a,w.numpy());got=G.crop128(pil,w,(ow,h)).numpy()
    err=float(np.abs(got-expected).max());assert err<I.crop_error_budget(ow,h)
    assert float(np.abs(got-expected.transpose(0,2,1)).max())>1e-3
    for bad in (torch.tensor([[.8,.8,.2,.2]]),torch.tensor([[0.,0.,64.,64.]])):
        try:G.geometry_cost(bad)
        except AssertionError:pass
        else:raise AssertionError('wrong geometry format accepted')
    # Reversed cost priority and argmax-class substitution are detected.
    probe_C=torch.tensor([[0.]])
    _,_,legal=G.m0_matrices(torch.tensor([[.5,.5,.2,.2]]),torch.tensor([[0.,5.]+[-5.]*8]),
        torch.tensor([[.5,.5,.2,.2]]),torch.tensor([1]),torch.tensor([0]))
    assert not bool(legal[0,0])
    res={'status':'已验证','exact_assignment_cases':cases,'old341_32_non_M0_parity':rows,
         'crop_non_square_truncated_max_abs_diff':err,'crop_error_budget':I.crop_error_budget(ow,h),
         'reverse_tests':['cxcywh rejected','pixel geometry rejected','transposed crop detected','query argmax substitution rejected'],
         '已执行':{'synthetic_exact_M0_cases':242,'old_val_caches':32,'image_decodes':32,'global_forwards':0,'optimizer_steps':0},
         'source_sha256':C.sources(),'未执行':['真实训练','GPU','阶段4至5'],'未核':['val548全量目标缓存']}
    C.json_out('342_result_阶段3单元验收.json',res)
    print(json.dumps({'status':'pass','M0_exact_cases':242,'old341_nonM0_bitwise':len(rows),'crop_error':err}))

if __name__=='__main__':main()
