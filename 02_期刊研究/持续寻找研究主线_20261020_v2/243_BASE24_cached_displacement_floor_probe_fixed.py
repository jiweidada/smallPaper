"""242 wiring repair of frozen 230; unchanged 229 geometry/encoding probe.
Written only; root reviews/runs.
Never run a detector, load a checkpoint, differentiate, update or read heldout.
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
os.environ['MPLBACKEND'] = 'Agg'; sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key] = '2'
import argparse, collections, gc, hashlib, importlib.util, json, pathlib, time, traceback
ROOT = pathlib.Path(__file__).resolve().parent
PROTOCOL = ROOT/'229_绝对位移标度_缓存诊断事前协议.txt'
REPAIR_SOURCES = {
    'protocol': (PROTOCOL,'a1ce08c7ca815c93ceec1c877530b959e9cb6ccd1659847cb8c941b89799166a'),
    'repair242': (ROOT/'242_容量函数接线修复_说明.txt','8b0a3d8c4819740a5775b7956b88c4c393773628bbf72371d70045aa7083cfd9'),
    'frozen230': (ROOT/'230_BASE24_cached_displacement_floor_probe.py','21905053354bfe2fef4189a454914e258c92e5cf341ff48142653fd87ed55c51'),
    'failed231': (ROOT/'231_BASE24_cached_displacement_floor_probe/FAILED.json','094ce4611ea2efcb15d88aae29b48783f456c41b4cd38cdd5eeea28b1546e1b1'),
}
REUSE237 = ROOT/'237_BASE24_saved_GO_repair_probe.py'
SHA237 = '20a98d07a57def68399fa71338e05388b6cf7f359117a44d89ceaaabf7ff348d'
if hashlib.sha256(REUSE237.read_bytes()).hexdigest() != SHA237: raise RuntimeError('frozen237 changed')
spec = importlib.util.spec_from_file_location('_probe230_cache_defs',REUSE237)
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
P = V.P
np, torch, H = P.np, P.torch, P.H
need, sha, save = P.require, P.sha, P.save
ARMS = ('N','F4','E4','C4','F8','E8','C8','T2')
SIGNS = np.array([-1.,-1.,1.,1.])
PARITY_ATOL = 1e-6
COUNTS = dict(prediction_cache_loads=0, native_matcher_calls=0, model_builds=0, checkpoint_loads=0,
              model_forwards=0, model_backwards=0, model_vjps=0, autograd_calls=0, optimizer_steps=0, heldout_reads=0)

def array(value):
    return value.detach().cpu().numpy().copy() if torch.is_tensor(value) else np.asarray(value)

def xyxy(boxes):
    return np.concatenate((boxes[...,:2]-boxes[...,2:]/2,boxes[...,:2]+boxes[...,2:]/2),axis=-1)

def cxcywh(edges):
    return np.concatenate(((edges[...,:2]+edges[...,2:])/2,edges[...,2:]-edges[...,:2]),axis=-1)

def statistics(values):
    values = np.asarray(values,dtype=np.float64).reshape(-1); finite = values[np.isfinite(values)]
    return dict(n=int(values.size),finite_n=int(finite.size),nonfinite_n=int(values.size-finite.size),
                sum=float(finite.sum()),mean=float(finite.mean()) if finite.size else None,
                quantiles=np.quantile(finite,[0,.25,.5,.75,1]).tolist() if finite.size else None)

def geometry(pred, gt):
    """Paired FP64 metrics. Invalid boxes: IoU0; GIoU and compound loss NA."""
    pred,gt = np.asarray(pred,dtype=np.float64),np.asarray(gt,dtype=np.float64)
    finite = np.isfinite(pred).all(-1); valid = finite & (pred[:,2:] > 0).all(-1)
    a,b = xyxy(pred),xyxy(gt); result = {}
    iou = np.zeros(len(pred)); giou = np.full(len(pred),np.nan)
    if valid.any():
        aa,bb = a[valid],b[valid]
        inter = np.maximum(np.minimum(aa[:,2:],bb[:,2:])-np.maximum(aa[:,:2],bb[:,:2]),0).prod(-1)
        union = (aa[:,2:]-aa[:,:2]).prod(-1)+(bb[:,2:]-bb[:,:2]).prod(-1)-inter
        enclosure = (np.maximum(aa[:,2:],bb[:,2:])-np.minimum(aa[:,:2],bb[:,:2])).prod(-1)
        need((union > 0).all() and (enclosure > 0).all(),'valid paired geometry denominator')
        iou[valid] = inter/union; giou[valid] = iou[valid]-(enclosure-union)/enclosure
    l1 = np.abs(pred-gt).sum(-1); corner_l1 = np.abs(a-b).sum(-1)
    result.update(valid=valid,finite=finite,iou=iou,giou=giou,l1=l1,corner_l1=corner_l1,
                  geometry_loss=5*l1+2*(1-giou))
    return result

def arm_layout(refs, nodes, hw, arm):
    """Outward offset nodes increase for every edge; physical L/T use SIGNS."""
    refs = np.asarray(refs,dtype=np.float64); n = len(refs); hh,ww = hw
    need(refs.shape == (n,4) and np.isfinite(refs).all() and (refs[:,2:] > 0).all(),'positive reference schema')
    base_ref = refs.copy(); axis = refs[:,2:].copy(); node = np.broadcast_to(nodes,(n,4,33)).copy()
    k = int(arm[1:]) if arm[0] in ('F','E','C') else 0
    floor = np.maximum(axis,np.array([k/ww,k/hh])) if k else axis.copy()
    if arm.startswith('C'): base_ref[:,2:] = floor
    if arm[0] in ('F','C'): axis = floor
    edge_unit = axis[:,[0,1,0,1]]/4.
    independent_endpoint_error = 0.
    if arm.startswith('E'):
        alpha = floor[:,[0,1,0,1]]/refs[:,[2,3,2,3]]
        node[:,:,0] *= alpha; node[:,:,-1] *= alpha
    if arm == 'T2': node[:,:,0] *= 2; node[:,:,-1] *= 2
    offset = node*edge_unit[:,:,None]
    if arm.startswith('E'):
        canonical = floor[:,[0,1,0,1]]/4.
        ends = np.stack((nodes[0]*canonical,nodes[-1]*canonical),axis=-1)
        independent_endpoint_error = float(np.abs(offset[:,:,[0,32]]-ends).max()) if n else 0.
        need(independent_endpoint_error <= 1e-12,'independent E/F physical endpoint identity')
        offset[:,:,0],offset[:,:,-1] = ends[:,:,0],ends[:,:,1]
    need((np.diff(offset,axis=-1) > 0).all(),'strictly ordered outward nodes')
    return dict(base=xyxy(base_ref),unit=edge_unit,node=node,offset=offset,k=k,
                active=(refs[:,2:] < np.array([k/ww,k/hh])).any(-1) if k else np.zeros(n,dtype=bool),
                endpoint_identity_max_error=independent_endpoint_error)

def decode(prob, layout):
    # Base is not multiplied by sum(p); matches the actual residual decoder.
    mean_offset = (prob*layout['offset']).sum(-1)
    return cxcywh(layout['base']+SIGNS*mean_offset)

def supports(layout, gt):
    delta = (xyxy(gt)[None,:,:]-layout['base'][:,None,:])*SIGNS
    lower,upper = layout['offset'][:,:,0][:,None,:],layout['offset'][:,:,-1][:,None,:]
    graph = ((delta >= lower)&(delta <= upper)).all(-1)
    eps = 1e-10*np.maximum(1.,np.maximum(np.abs(lower),np.abs(upper)))
    near = ((np.abs(delta-lower) <= eps)|(np.abs(delta-upper) <= eps)).any(-1)
    return graph,near

def encode(layout, q, gt):
    target = (xyxy(gt)-layout['base'][q])*SIGNS; grid = layout['offset'][q]
    raw = target/layout['unit'][q]
    idx = (grid <= target[:,:,None]).sum(-1)-1; inside = (target >= grid[:,:,0])&(target <= grid[:,:,-1])
    left = np.clip(idx,0,31).astype(np.int64)
    lo = np.take_along_axis(grid,left[:,:,None],axis=-1)[:,:,0]
    hi = np.take_along_axis(grid,(left+1)[:,:,None],axis=-1)[:,:,0]
    wr = np.clip((target-lo)/(hi-lo),0,1); wl = 1-wr
    dense = np.zeros_like(grid)
    np.put_along_axis(dense,left[:,:,None],wl[:,:,None],axis=-1)
    np.put_along_axis(dense,(left+1)[:,:,None],wr[:,:,None],axis=-1)
    reconstructed = (dense*grid).sum(-1)
    need(np.max(np.abs(dense.sum(-1)-1),initial=0) <= 1e-12,'target probability mass')
    return dict(raw_target=raw,physical_target=target,left=left,wr=wr,wl=wl,inside=inside,
                dense=dense,physical_reconstruction=reconstructed,physical_error=reconstructed-target)

def summarize_pair(metrics, baseline, enc, enc0, mask):
    mask = np.asarray(mask,dtype=bool); old,new = baseline['iou'][mask],metrics['iou'][mask]
    return dict(n=int(mask.sum()),invalid=int((~metrics['valid'][mask]).sum()),
        newly_invalid=int((baseline['valid'][mask]&~metrics['valid'][mask]).sum()),
        originally_invalid=int((~baseline['valid'][mask]).sum()),nonfinite=int((~metrics['finite'][mask]).sum()),
        negative_width_height=int((metrics['negative'][mask]).sum()),zero_width_height=int(metrics['zero'][mask].sum()),
        IoU=statistics(new),L1_normalized_cxcywh=statistics(metrics['l1'][mask]),corner_L1_normalized=statistics(metrics['corner_l1'][mask]),
        GIoU_valid_only=statistics(metrics['giou'][mask]),geometry5L1plus2GIoU_valid_only=statistics(metrics['geometry_loss'][mask]),
        IoU_delta=statistics(new-old),geometry_delta_valid_in_both=statistics((metrics['geometry_loss']-baseline['geometry_loss'])[mask]),
        zero_to_positive=int(((old == 0)&(new > 0)).sum()),positive_to_zero=int(((old > 0)&(new == 0)).sum()),
        up05=int(((old < .5)&(new >= .5)).sum()),down05=int(((old >= .5)&(new < .5)).sum()),
        up075=int(((old < .75)&(new >= .75)).sum()),down075=int(((old >= .75)&(new < .75)).sum()),
        all_four_targets_inside=int(enc['inside'][mask].all(-1).sum()),outside_edges=int((~enc['inside'][mask]).sum()),
        target_changed_queries=int((enc['dense'][mask] != enc0['dense'][mask]).any((1,2)).sum()),
        target_L2_change=statistics(np.linalg.norm((enc['dense']-enc0['dense']).reshape(-1,132),axis=-1)[mask]),
        physical_target_clipping_L1=statistics(np.abs(enc['physical_error'][mask]).sum(-1)))

def native_parity(stage, nodes32, native):
    refs,corners,boxes = stage['ref_points'],stage['pred_corners'],stage['pred_boxes']
    need(corners.shape[:-1] == refs.shape[:-1] == boxes.shape[:-1] and corners.shape[-1] == 132,'FDR stage shapes')
    need(all(t.dtype == torch.float32 and t.device.type == 'cpu' and not t.requires_grad and bool(torch.isfinite(t).all()) for t in (refs,corners,boxes)),'frozen FP32 stage')
    prob = corners.reshape(*corners.shape[:-1],4,33).softmax(-1)
    distance = torch.nn.functional.linear(prob.reshape(-1,33),nodes32).reshape(refs.shape)
    reconstructed = native.distance2bbox(refs,distance,torch.tensor([4.],dtype=torch.float32))
    error = float((reconstructed-boxes).abs().max())
    need(error <= PARITY_ATOL,'native FP32 decoder/cache parity: '+str(error))
    return prob,error

def synthetic(native, box):
    need(P.A is P.B.A and callable(P.A.matching_count),'159 matching_count module wiring')
    capacity_cases = {
        'empty_0x0': (np.zeros((0,0),dtype=bool),0),
        'empty_0x3': (np.zeros((0,3),dtype=bool),0),
        'empty_3x0': (np.zeros((3,0),dtype=bool),0),
        'zero_2x3': (np.zeros((2,3),dtype=bool),0),
        'augmenting_3x3': (np.array([[1,1,0],[1,0,0],[0,0,1]],dtype=bool),3),
        'deficient_3x2': (np.array([[1,0],[1,0],[0,1]],dtype=bool),2),
    }
    capacity_results = {}
    for name,(edge,expected) in capacity_cases.items():
        actual = P.A.matching_count(edge)
        need(actual == expected,'capacity interface synthetic: '+name)
        capacity_results[name] = dict(actual=actual,expected=expected)
    nodes32 = native.weighting_function(32,torch.tensor([.5]),torch.tensor([4.])); nodes = array(nodes32).astype(np.float64)
    need(nodes.shape == (33,) and nodes[0] == -4 and nodes[-1] == 4,'native node contract')
    ref = np.array([[.5,.4,2/640,3/640],[.3,.7,.1,.2]])
    prob = np.zeros((2,4,33)); prob[:,:,16] = 1
    layouts = {arm:arm_layout(ref,nodes,(640,640),arm) for arm in ARMS}
    for arm in ARMS:
        pred = decode(prob,layouts[arm]); expected = ref.copy()
        if arm.startswith('C'): expected[:,2:] = np.maximum(expected[:,2:],int(arm[1:])/640)
        need(np.max(np.abs(pred-expected)) <= 1e-15,'zero residual base identity')
    for k in (4,8):
        f,e = layouts[f'F{k}'],layouts[f'E{k}']
        need(np.array_equal(f['offset'][:,:,[0,32]],e['offset'][:,:,[0,32]]),'same physical endpoints')
        need(np.array_equal(f['offset'][1],layouts['N']['offset'][1]),'inactive reference unchanged')
        need((f['offset'][0,:,1:32] != e['offset'][0,:,1:32]).any(),'same support need not same interior')
    gt = np.array([[.501,.402,3/640,3/640],[.31,.69,.11,.19]])
    for arm in ARMS:
        enc = encode(layouts[arm],np.arange(2),gt)
        valid = enc['inside']
        need(np.max(np.abs(enc['physical_error'][valid]),initial=0) <= 1e-15,'inside encoding reconstruction')
        graph,_ = supports(layouts[arm],gt)
        need(np.array_equal(graph[np.arange(2),np.arange(2)],valid.all(-1)),'support/encoding correspondence')
    n,_ = supports(layouts['N'],gt)
    for k in (4,8):
        f,_ = supports(layouts[f'F{k}'],gt); e,_ = supports(layouts[f'E{k}'],gt)
        need(np.array_equal(f,e) and not (n&~f).any(),'support identity and nesting')
    need(np.array_equal(layouts['T2']['offset'][:,:,1:32],layouts['N']['offset'][:,:,1:32]),'T2 interior unchanged')
    negative = prob.copy(); negative[:,:,:] = 0; negative[:,:,0] = 1
    need((decode(negative,layouts['F4'])[:,2:] <= 0).any(),'invalid boxes must not be silently clamped')
    beyond = gt.copy(); beyond[:,0] = 10
    enc = encode(layouts['F4'],np.arange(2),beyond)
    need((~enc['inside']).any() and np.allclose(enc['dense'].sum(-1),1,atol=1e-15),'endpoint target saturation')
    m = geometry(ref,gt); native_iou = box.box_iou(torch.tensor(xyxy(ref)),torch.tensor(xyxy(gt)))[0].diag().numpy()
    native_giou = box.generalized_box_iou(torch.tensor(xyxy(ref)),torch.tensor(xyxy(gt))).diag().numpy()
    need(np.allclose(m['iou'],native_iou,atol=1e-15,rtol=0) and np.allclose(m['giou'],native_giou,atol=1e-15,rtol=0),'paired native metric formulas')
    return dict(zero_residual=True,inactive_reference=True,inverse_interpolation=True,
                F_equals_E_support=True,N_subset_F=True,T2_interior_unchanged=True,
                invalid_unclamped=True,endpoint_encoding=True,native_geometry=True,
                capacity_interface=True,capacity_cases=capacity_results)

def analyze_stage(name, stage, prob32, target, pairs, hw, index, nodes, native, box, arrays):
    refs = array(stage['ref_points'][index]).astype(np.float64)
    cached_boxes = array(stage['pred_boxes'][index]).astype(np.float64)
    prob = array(prob32[index]).astype(np.float64); gt = array(target['boxes']).astype(np.float64)
    labels = array(target['labels']); actual = (gt[:,2:]*np.array([hw[1],hw[0]])).min(-1) < 4
    prefix = f'i{index}_{name}'; layouts = {a:arm_layout(refs,nodes,hw,a) for a in ARMS}
    decoded = {a:decode(prob,layouts[a]) for a in ARMS}
    error = float(np.abs(decoded['N']-cached_boxes).max())
    need(error <= PARITY_ATOL,'FP64 diagnostic/native cache parity: '+str(error))
    ref_active = {k:(refs[:,2:]*np.array([hw[1],hw[0]]) < k).any(-1) for k in (4,8)}
    arrays[prefix+'_ref'] = refs; arrays[prefix+'_GT'] = gt; arrays[prefix+'_GT_labels'] = labels
    arrays[prefix+'_actual_input_sub4'] = actual; arrays[prefix+'_cached_boxes'] = cached_boxes
    result = dict(stage=name,queries=len(refs),GT=len(gt),actual_input_sub4=int(actual.sum()),
                  diagnostic_FP64_max_error=error,ref_activated4=int(ref_active[4].sum()),
                  ref_activated8=int(ref_active[8].sum()),pairs={},node_spacing={},support=None)
    for arm in ARMS:
        arrays[prefix+'_'+arm+'_decoded'] = decoded[arm]
        pixel_gap = np.diff(layouts[arm]['offset'],axis=-1)*np.array([hw[1],hw[0],hw[1],hw[0]])[None,:,None]
        result['node_spacing'][arm] = dict(all_gaps_pixels=statistics(pixel_gap),interior_gaps_pixels=statistics(pixel_gap[:,:,1:31]),
            tail_gaps_pixels=statistics(pixel_gap[:,:,[0,31]]),endpoint_identity_error=layouts[arm]['endpoint_identity_max_error'])
        arrays[prefix+'_'+arm+'_gap_pixels'] = pixel_gap
    for pair_name,indices in pairs.items():
        q,g = np.asarray(indices,dtype=np.int64).T
        need(len(set(q.tolist())) == len(q) and np.all((q >= 0)&(q < len(refs))) and np.all((g >= 0)&(g < len(gt))), 'fixed positive pair schema')
        pg = gt[g]; enc = {a:encode(layouts[a],q,pg) for a in ARMS}
        met = {a:geometry(decoded[a][q],pg) for a in ARMS}
        for a in ARMS:
            met[a]['negative'] = (decoded[a][q,2:] < 0).any(-1)
            met[a]['zero'] = (decoded[a][q,2:] == 0).any(-1)
        cached_t = stage['pred_boxes'][index,torch.as_tensor(q)].detach()
        target_t = target['boxes'][torch.as_tensor(g)]
        # Actual original IoU is native FP32, while all counterfactual comparisons share FP64.
        legal = (cached_t[:,2:] > 0).all(-1)
        cached_iou = np.zeros(len(q),dtype=np.float32)
        if bool(legal.any()):
            cached_iou[array(legal)] = array(box.box_iou(box.box_cxcywh_to_xyxy(cached_t[legal]),box.box_cxcywh_to_xyxy(target_t[legal]))[0].diag())
        nleft,nwr,nwl = native.bbox2distance(stage['ref_points'][index,torch.as_tensor(q)],box.box_cxcywh_to_xyxy(target_t),32,torch.tensor([4.]),torch.tensor([.5]))
        pair_prefix = prefix+'_'+pair_name
        arrays[pair_prefix+'_pairs'] = np.stack((q,g),axis=-1); arrays[pair_prefix+'_cached_native_IoU32'] = cached_iou
        arrays[pair_prefix+'_native_target_left_FP32'] = array(nleft).reshape(-1,4)
        arrays[pair_prefix+'_native_target_wr_FP32'] = array(nwr).reshape(-1,4)
        arrays[pair_prefix+'_native_target_wl_FP32'] = array(nwl).reshape(-1,4)
        native_dense = np.zeros_like(enc['N']['dense']); ni = array(nleft).reshape(-1,4).astype(np.int64)
        np.put_along_axis(native_dense,ni[:,:,None],array(nwl).reshape(-1,4,1),axis=-1)
        np.put_along_axis(native_dense,(ni+1)[:,:,None],array(nwr).reshape(-1,4,1),axis=-1)
        native_outside = ~enc['N']['inside'].all(-1)
        masks = dict(all=np.ones(len(q),dtype=bool),actual_sub4=actual[g],native_IoU0_outside=(cached_iou == 0)&native_outside,
                     main_actual_sub4_IoU0_outside=actual[g]&(cached_iou == 0)&native_outside,native_good05=cached_iou >= .5,
                     actual_sub4_refactive4=actual[g]&ref_active[4][q],non_sub4_refactive4=(~actual[g])&ref_active[4][q],
                     ref_inactive4=~ref_active[4][q])
        original = target.get('original_sub4')
        if original is not None:
            need(torch.is_tensor(original) and original.dtype == torch.bool and original.shape == (len(gt),),'explicit original_sub4')
            masks['original_sub4'] = array(original)[g]
        detail = dict(slots=len(q),unique_GT=len(set(g.tolist())),original_sub4_available=original is not None,
                      native_target_FP32_vs_common_FP64=statistics(np.linalg.norm((native_dense-enc['N']['dense']).reshape(-1,132),axis=-1)),
                      native_IoU32_vs_N_common_FP64=statistics(met['N']['iou']-cached_iou),arms={})
        for arm in ARMS:
            r = {key:summarize_pair(met[arm],met['N'],enc[arm],enc['N'],mask) for key,mask in masks.items()}
            mainmask = masks['main_actual_sub4_IoU0_outside']; inside = enc[arm]['inside'].all(-1)
            r['main_new_support_slots'] = int((mainmask&inside).sum())
            r['main_new_support_unique_GT'] = len(set(g[mainmask&inside].tolist()))
            r['support_gained'] = int((native_outside&inside).sum()); r['support_lost'] = int(((~native_outside)&~inside).sum())
            r['paired_target_cohort'] = {key:int(mask.sum()) for key,mask in masks.items()}
            detail['arms'][arm] = r
            stem = pair_prefix+'_'+arm
            for key in ('raw_target','left','wr','wl','inside','physical_reconstruction','physical_error'): arrays[stem+'_'+key] = enc[arm][key]
            for key in ('iou','giou','l1','corner_l1','geometry_loss','valid'): arrays[stem+'_'+key] = met[arm][key]
            arrays[stem+'_new_weight_IoU64'] = met[arm]['iou']
        for k in (4,8):
            # Pair support equality is a required identity, not a method win.
            need(np.array_equal(enc[f'F{k}']['inside'],enc[f'E{k}']['inside']),'F/E pair encoding support equality')
            detail[f'F{k}_vs_E{k}'] = dict(decoded_box_L1=statistics(np.abs(decoded[f'F{k}'][q]-decoded[f'E{k}'][q]).sum(-1)),
                target_L2=statistics(np.linalg.norm((enc[f'F{k}']['dense']-enc[f'E{k}']['dense']).reshape(-1,132),axis=-1)),
                exact_same_boxes=bool(np.array_equal(decoded[f'F{k}'][q],decoded[f'E{k}'][q])),
                exact_same_targets=bool(np.array_equal(enc[f'F{k}']['dense'],enc[f'E{k}']['dense'])))
        result['pairs'][pair_name] = detail
    if name == 'final':
        graphs,near = {},{}
        for arm in ARMS:
            graphs[arm],near[arm] = supports(layouts[arm],gt)
            arrays[prefix+'_'+arm+'_exact_GT_support'] = graphs[arm]
        for k in (4,8):
            need(np.array_equal(graphs[f'F{k}'],graphs[f'E{k}']) and not (graphs['N']&~graphs[f'F{k}']).any(), 'all300 support identity/nesting')
        qclass = array(stage['pred_logits'][index]).argmax(-1); arrays[prefix+'_argmax_class'] = qclass
        result['support'] = {}
        for mode in ('unrestricted','argmax_class_sensitivity'):
            classmask = np.ones_like(graphs['N']) if mode == 'unrestricted' else qclass[:,None] == labels[None,:]
            result['support'][mode] = {}
            for cohort,mask in (('all',np.ones(len(gt),dtype=bool)),('actual_sub4',actual)):
                selected = {a:(graphs[a]&classmask)[:,mask] for a in ARMS}; old_any = selected['N'].any(0)
                table = {}
                for arm in ARMS:
                    edge = selected[arm]; anyq = edge.any(0)
                    table[arm] = dict(GT=int(mask.sum()),any_query_exact_representable=int(anyq.sum()),
                        native_any_unreachable_to_reachable=int((~old_any&anyq).sum()),native_reachable_lost=int((old_any&~anyq).sum()),
                        exact_GT_query_capacity1=P.A.matching_count(edge),near_boundary_pairs=int(near[arm][:,mask].sum()))
                for k in (4,8):
                    ff,ee,cc = selected[f'F{k}'].any(0),selected[f'E{k}'].any(0),selected[f'C{k}'].any(0)
                    need(np.array_equal(ff,ee),'any-query support identity')
                    table[f'F{k}_vs_C{k}'] = dict(both=int((ff&cc).sum()),floor_only=int((ff&~cc).sum()),clamp_only=int((cc&~ff).sum()))
                    tt = selected['T2'].any(0)
                    table[f'F{k}_vs_T2'] = dict(both=int((ff&tt).sum()),floor_only=int((ff&~tt).sum()),T2_only=int((tt&~ff).sum()))
                result['support'][mode][cohort] = table
    return result

def pool_pair_summaries(rows):
    scalar = ('n','invalid','newly_invalid','originally_invalid','nonfinite','negative_width_height','zero_width_height',
              'zero_to_positive','positive_to_zero','up05','down05','up075','down075','all_four_targets_inside','outside_edges','target_changed_queries')
    metrics = ('IoU','L1_normalized_cxcywh','corner_L1_normalized','GIoU_valid_only','geometry5L1plus2GIoU_valid_only',
               'IoU_delta','geometry_delta_valid_in_both','target_L2_change','physical_target_clipping_L1')
    out = {k:sum(r[k] for r in rows) for k in scalar}
    for key in metrics:
        nn = sum(r[key]['n'] for r in rows); fn = sum(r[key]['finite_n'] for r in rows); total = sum(r[key]['sum'] for r in rows)
        out[key] = dict(n=nn,finite_n=fn,nonfinite_n=nn-fn,sum=total,mean=total/fn if fn else None,pooled_quantiles_not_computed=True)
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--synthetic-only',action='store_true')
    parser.add_argument('--out',type=pathlib.Path,default=ROOT/'244_BASE24_cached_displacement_floor_probe_fixed'); args = parser.parse_args()
    started = time.monotonic(); resources = H.resource_limits()
    need(resources['priority'] == 'below_normal' and torch.version.cuda is None,'CPU resource contract')
    for key,(path,digest) in REPAIR_SOURCES.items():
        need(sha(path) == digest,'frozen repair provenance before synthetic: '+key)
    for key,filename in (('matcher','matcher.py'),('criterion','deim_criterion.py'),('box_ops','box_ops.py'),('dfine_utils','dfine_utils.py')):
        need(sha(P.A.NATIVE/filename) == P.A.EXPECTED[key],'native source before synthetic')
    native = H.load_native_math(); box = sys.modules[native.__package__+'.box_ops']
    tests = dict(geometry_encoding=synthetic(native,box),saved_GO_portability=V.compatibility_self_checks())
    if args.synthetic_only: print(json.dumps(dict(tests=tests,counts=COUNTS),ensure_ascii=False)); return
    outdir = args.out.resolve(); need(outdir.parent == ROOT and outdir.name.startswith('244_') and not outdir.exists(),'new244 only'); outdir.mkdir()
    stage = 'fixed_inputs'; files,expected = V.manifest(); before = {}; batches=[]; artifacts={}
    files.update(reuse237=files['script'],protocol236=files['protocol'],script=pathlib.Path(__file__),protocol=PROTOCOL,
                 design226=ROOT/'226_A_FDR固定物理步长_设计与原文边界.txt')
    expected.update(reuse237=SHA237,design226='7da468a66ec744a008a9b832d6b608fdf22f94a45000b64dbc68daf653ad4ca7')
    files.update({key:path for key,(path,digest) in REPAIR_SOURCES.items()})
    expected.update({key:digest for key,(path,digest) in REPAIR_SOURCES.items()})
    try:
        before = {key:sha(path) for key,path in files.items()}
        need(all(before[key] == digest for key,digest in expected.items()),'frozen input/source identity')
        save(outdir/'00_inputs_and_synthetic.json',dict(paths={k:str(p.resolve()) for k,p in files.items()},sha256=before,tests=tests,resources=resources))
        recipes,package = P.read_json(files['recipe']),P.read_json(files['package']); shapes=P.input_shapes(); audit=V.load_union_audit()
        package_hashes={x['relative']:x['sha256'] for x in package['files']}
        for key in ('matcher','criterion','box_ops','dfine_utils'): need(package_hashes['native/engine/deim/'+files[key].name] == before[key],'native package source identity')
        for letter,seed in (('A',1),('B',2)):
            predrows,analysisrows,provenance=P.metadata(letter,seed); recipe=recipes[str(seed)]
            need(recipe['DEIMCriterion']['weight_dict']['loss_bbox'] == 5 and recipe['DEIMCriterion']['weight_dict']['loss_giou'] == 2,'geometry recipe weights')
            compute,union=P.native_readers(box,recipe)
            for batch in range(8):
                P.M.guard(); stage=f'{letter}_development_batch{batch:02d}'; shape=shapes[batch]['shape']; hw=(shape[2],shape[3])
                need(predrows[batch]['input_sha256'] == shapes[batch]['sha256'],'input shape seal identity')
                predpath=P.DATA/letter/'frozen_predictions/development'/predrows[batch]['relative']
                analysispath=P.DATA/letter/'development'/analysisrows[batch]['relative']
                need(sha(predpath)==predrows[batch]['sha256'] and sha(analysispath)==analysisrows[batch]['sha256'],'batch hashes before content')
                prior=P.read_json(analysispath); value=torch.load(predpath,map_location='cpu',weights_only=False); COUNTS['prediction_cache_loads']+=1
                COUNTS['native_matcher_calls']+=1
                output,targets,original,costs,denom=V.validate_batch(value,prior,batch,seed,predrows[batch],recipe,compute,union,audit)
                need(H.scalar(output['up'],'up') == .5 and abs(H.scalar(output['reg_scale'],'reg_scale')) == 4 and recipe['DFINETransformer']['reg_max']==32,'cached FDR constants')
                nodes32=native.weighting_function(32,output['up'],output['reg_scale']); nodes=array(nodes32).astype(np.float64)
                need(nodes.shape==(33,) and nodes[0]==-4 and nodes[-1]==4,'native W33 endpoints')
                stages=[('final',output)]+[(f'aux_{i}',s) for i,s in enumerate(output['aux_outputs'])]
                dn_pairs=None
                if 'dn_outputs' in output:
                    need('dn_meta' in output and 'dn_pre_outputs' in output,'DN metadata/reference availability')
                    groups=int(output['dn_meta']['dn_num_group']); need(groups>0,'DN positive groups')
                    dn_pairs=[]
                    for i,target in enumerate(targets):
                        q=array(output['dn_meta']['dn_positive_idx'][i]); g=np.tile(np.arange(len(target['labels']),dtype=np.int64),groups)
                        need(q.dtype==np.int64 and q.shape==g.shape,'native DN positive index identity')
                        dn_pairs.append(np.stack((q,g),axis=1))
                    for i,s in enumerate(output['dn_outputs']):
                        need(torch.equal(s['ref_points'],output['dn_pre_outputs']['pred_boxes']),'DN actual shared pre reference')
                        stages.append((f'dn_{i}',s))
                arrays={}; images=[dict(image_index=i,GT=len(t['labels']),GT_over300=len(t['labels'])>300,stages={}) for i,t in enumerate(targets)]
                parity={}
                for name,s in stages:
                    P.M.guard(); prob,error=native_parity(s,nodes32,native); parity[name]=error
                    for i,target in enumerate(targets):
                        if name.startswith('dn_'): pairsets={'DN_positive':dn_pairs[i]}
                        else:
                            own=np.stack(prior['branch_matches'][i][name],axis=1)
                            pairsets={'GO':original[i],'own':own}
                        report=analyze_stage(name,s,prob,target,pairsets,hw,i,nodes,native,box,arrays)
                        report['native_FP32_max_error']=error; images[i]['stages'][name]=report
                # Direct pre-clamp effect is separate from FDR; no fake pre PMF.
                for i,target in enumerate(targets):
                    q,g=original[i].T; pre=array(output['pre_outputs']['pred_boxes'][i]).astype(np.float64); gt=array(target['boxes']).astype(np.float64)[g]
                    pretable={}
                    for k in (0,4,8):
                        pred=pre.copy()
                        if k: pred[:,2:]=np.maximum(pred[:,2:],np.array([k/hw[1],k/hw[0]]))
                        mm=geometry(pred[q],gt); pretable['N' if not k else f'C{k}']={key:statistics(mm[key]) for key in ('iou','l1','giou','geometry_loss')}
                    images[i]['pre_clamp_fixed_GO']=pretable
                batchreport=dict(letter=letter,seed=seed,batch=batch,shape=shape,provenance=provenance,denominators=denom,
                    prediction_sha256=predrows[batch]['sha256'],input_sha256=predrows[batch]['input_sha256'],source_ids=predrows[batch]['source_ids'],
                    stages=list(parity),native_FP32_parity=parity,DN_available=dn_pairs is not None,DN_num_group=int(output['dn_meta']['dn_num_group']) if dn_pairs is not None else None,
                    images=images,scope='fixed saved targets/corners/ref; no new decoder/criterion/parameter graph')
                stem=f'{letter}_batch{batch:02d}'; save(outdir/(stem+'.json'),batchreport)
                np.savez(outdir/(stem+'_arrays.npz'),**arrays)
                artifacts[stem+'.json']=sha(outdir/(stem+'.json')); artifacts[stem+'_arrays.npz']=sha(outdir/(stem+'_arrays.npz'))
                batches.append(batchreport); del value,output,targets,costs,arrays,prob,dn_pairs; gc.collect(); print(stage,'complete',flush=True)
        need(COUNTS['prediction_cache_loads']==COUNTS['native_matcher_calls']==16 and len(batches)==16,'exact development cache budget')
        summaries={}
        for letter in ('A','B'):
            images=[im for b in batches if b['letter']==letter for im in b['images']]
            stages=[im['stages']['final'] for im in images]; summary=dict(inputs=len(images),GT=sum(im['GT'] for im in images),arms={})
            for arm in ARMS:
                rr=[s['pairs']['GO']['arms'][arm] for s in stages]
                summary['arms'][arm]=dict(main_new_support_slots=sum(r['main_new_support_slots'] for r in rr),
                    main_new_support_unique_GT_exposures=sum(r['main_new_support_unique_GT'] for r in rr),
                    main_cohort=sum(r['paired_target_cohort']['main_actual_sub4_IoU0_outside'] for r in rr),
                    support_gained=sum(r['support_gained'] for r in rr),support_lost=sum(r['support_lost'] for r in rr),
                    all_GO=pool_pair_summaries([r['all'] for r in rr]),actual_sub4_GO=pool_pair_summaries([r['actual_sub4'] for r in rr]),
                    all300_unrestricted_new_actual_sub4=sum(s['support']['unrestricted']['actual_sub4'][arm]['native_any_unreachable_to_reachable'] for s in stages))
            summary['necessary_screen_k4']=summary['arms']['F4']['main_new_support_slots']>0
            summaries[letter]=summary
        passed=[summaries[a]['necessary_screen_k4'] for a in ('A','B')]
        unchanged={key:sha(path)==before[key] for key,path in files.items()}; need(all(unchanged.values()),'source/input changed')
        need(all(COUNTS[k]==0 for k in ('model_builds','checkpoint_loads','model_forwards','model_backwards','model_vjps','autograd_calls','optimizer_steps','heldout_reads')),'zero model/update budget')
        save(outdir/'01_results.json',dict(state='complete',summaries=summaries,primary_scale_pixels=4,sensitivity_scale_pixels=8,
            necessary_screen_both=all(passed),screen_status='both' if all(passed) else ('mixed' if any(passed) else 'neither'),
            training_authorized=False,AP=False,continuous_joint_IoU_oracle=False,full_criterion_replay=False,shared_parameter_gradient=False,
            counts=COUNTS,resources=resources,sources_unchanged=unchanged,artifact_sha256=artifacts,seconds=time.monotonic()-started))
        save(outdir/'02_output_sha256.json',{p.name:sha(p) for p in outdir.iterdir() if p.is_file()})
    except BaseException:
        save(outdir/'FAILED.json',dict(state='error_no_method_conclusion',stage=stage,counts=COUNTS,traceback=traceback.format_exc(),seconds=time.monotonic()-started)); raise

if __name__ == '__main__': main()
