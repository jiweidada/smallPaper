"""258: fixed-wh center-translation/observation-window necessary geometry only.

Written for root review, NOT executed by the author. Saved GO + frozen237
validation. No image, checkpoint, detector, gradient, training or heldout read.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import gc
import hashlib
import importlib.util
import math
import pathlib
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent
PROTOCOL = ROOT / '258_A_新增观测中心移动_必要面协议.txt'
REUSE = ROOT / '237_BASE24_saved_GO_repair_probe.py'
SHA237 = '20a98d07a57def68399fa71338e05388b6cf7f359117a44d89ceaaabf7ff348d'
if hashlib.sha256(REUSE.read_bytes()).hexdigest() != SHA237:
    raise RuntimeError('frozen237 changed')
_spec = importlib.util.spec_from_file_location('_probe259_frozen237', REUSE)
if _spec is None or _spec.loader is None:
    raise RuntimeError('237 import unavailable')
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)  # Definitions only; no historical main/experiment.
P, H, np, torch = V.P, V.H, V.np, V.torch
need, sha = V.require, V.sha
FLOOR_RESULT = ROOT / '244_BASE24_cached_displacement_floor_probe_fixed/01_results.json'
SHA244 = '9dfbf71fc5ea60820b529c4e1e0fb0c0412fd96af034a40f439a3e98618160b9'
RHO = 4.0
NEAR_PX = 1e-9  # Audit only. Never used to classify support/visibility/screen.
COUNTS = {k: 0 for k in ('prediction_cache_loads', 'native_matcher_calls', 'image_tensor_loads',
    'checkpoint_loads', 'detector_builds', 'model_forwards', 'model_backwards', 'autograd_calls',
    'optimizer_steps', 'heldout_reads')}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    return None if isinstance(value, float) and not math.isfinite(value) else value


def save(path, value):
    V.save(path, clean(value))


def array(value):
    return value.detach().cpu().numpy().copy()


def stats(value):
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    finite = value[np.isfinite(value)]
    return dict(n=len(value), finite_n=len(finite), NA_n=len(value)-len(finite),
        mean=float(finite.mean()) if len(finite) else None,
        quantiles=dict(zip(('min', 'p25', 'median', 'p75', 'max'),
            np.quantile(finite, [0, .25, .5, .75, 1]).tolist())) if len(finite) else None)


def xyxy(boxes):
    return np.concatenate((boxes[:, :2]-boxes[:, 2:]/2, boxes[:, :2]+boxes[:, 2:]/2), axis=1)


def visibility(gt_edges, windows):
    full = ((gt_edges[:, :2] >= windows[:, :2]) & (gt_edges[:, 2:] <= windows[:, 2:])).all(1)
    inter = np.maximum(np.minimum(gt_edges[:, 2:], windows[:, 2:])-
                       np.maximum(gt_edges[:, :2], windows[:, :2]), 0)
    area = np.prod(gt_edges[:, 2:]-gt_edges[:, :2], axis=1)
    ratio = np.prod(inter, axis=1)/area
    need(np.isfinite(ratio).all() and (ratio >= 0).all() and (ratio <= 1+1e-12).all(), 'visibility ratio')
    return full, ratio


def geometry(ref_px, gt_px, hw):
    """Analytic minimum translation in true pixels; no IoU optimization/search."""
    r, g = np.asarray(ref_px, dtype=np.float64), np.asarray(gt_px, dtype=np.float64)
    need(r.shape == g.shape and r.ndim == 2 and r.shape[1] == 4, 'pair geometry shape')
    need(np.isfinite(r).all() and np.isfinite(g).all() and (r[:, 2:] > 0).all()
         and (g[:, 2:] > 0).all(), 'positive finite physical boxes')
    dc = g[:, :2]-r[:, :2]
    slack = r[:, 2:]-np.abs(g[:, 2:]-r[:, 2:])/2
    feasible = (slack >= 0).all(1)
    support_margin = slack-np.abs(dc)
    inside = (support_margin >= 0).all(1)
    re, ge = xyxy(r), xyxy(g)
    corner_margin = np.tile(r[:, 2:], (1, 2))-np.abs(ge-re)
    corner_inside = (corner_margin >= 0).all(1)
    # Arithmetic identity is independently checked; no tolerance changes labels.
    need(np.array_equal(inside, corner_inside), 'center/corner native support identity')
    delta = np.full((len(r), 2), np.nan)
    delta[feasible] = np.sign(dc[feasible])*np.maximum(np.abs(dc[feasible])-slack[feasible], 0)
    minimum = np.full(len(r), np.nan)
    minimum[feasible] = np.max(np.abs(delta[feasible]), axis=1)
    repairable = feasible & (minimum <= RHO)
    projected = r.copy()
    projected[:, :2] += delta
    if feasible.any():
        projected_margin = np.tile(r[feasible, 2:], (1, 2))-np.abs(ge[feasible]-xyxy(projected[feasible]))
        need((projected_margin >= -NEAR_PX).all(), 'analytic projection does not reach closed support')
        need(np.array_equal(projected[feasible, 2:], r[feasible, 2:]), 'width/height changed')
        need(np.allclose(np.abs(delta[feasible]), np.maximum(-support_margin[feasible], 0),
                         rtol=0, atol=NEAR_PX), 'axis minimum distance certificate')
    windows = re + np.array([-RHO, -RHO, RHO, RHO])
    full, ratio = visibility(ge, windows)
    h, w = hw
    clipped = np.clip(windows, [0, 0, 0, 0], [w, h, w, h])
    full_input, ratio_input = visibility(ge, clipped)
    # Full nominal visibility + feasible size implies repairability analytically.
    need(not np.any(full & feasible & ~repairable), 'window/center containment identity')
    need(not np.any(full_input & ~full), 'clipped visibility cannot add full observations')
    window_margin = np.concatenate((ge[:, :2]-windows[:, :2], windows[:, 2:]-ge[:, 2:]), axis=1)
    return dict(ref_px=r, GT_px=g, dc_px=dc, size_slack_px=slack, size_feasible=feasible,
        native_support_margin_px=support_margin, native_inside=inside,
        minimum_delta_px=delta, minimum_Linf_px=minimum, repairable_rho4=repairable,
        original_observation_window_px=windows, clipped_observation_window_px=clipped,
        full_visible=full, visible_ratio=ratio, full_visible_input=full_input, visible_ratio_input=ratio_input,
        near_size_boundary=(np.abs(slack) <= NEAR_PX).any(1),
        near_support_boundary=(np.abs(support_margin) <= NEAR_PX).any(1),
        near_rho_boundary=feasible & (np.abs(minimum-RHO) <= NEAR_PX),
        exactly_rho_boundary=feasible & (minimum == RHO),
        near_window_boundary=(np.abs(window_margin) <= NEAR_PX).any(1))


def synthetic(native):
    # All cases use pixels, with a sufficiently large valid input canvas.
    ref = np.array([[32., 32., 2., 2.]]*8)
    gt = np.array([[32,32,7,2], [32,32,2,2], [38,32,2,2], [38.25,32,2,2],
                   [36,32,2,2], [32,38,2,2], [32,32,6,2], [32,32,2,7]], dtype=np.float64)
    z = geometry(ref, gt, (96,128))
    need(z['size_feasible'].tolist() == [False,True,True,True,True,True,True,False], 'synthetic size')
    need(z['native_inside'].tolist() == [False,True,False,False,False,False,True,False], 'synthetic inside')
    need(z['repairable_rho4'].tolist() == [False,True,True,False,True,True,True,False], 'synthetic rho')
    need(z['full_visible'].tolist() == [True,True,False,False,True,False,True,True], 'synthetic visibility')
    need(np.isnan(z['minimum_Linf_px'][0]) and np.isnan(z['minimum_delta_px'][0]).all(), 'infeasible is NA')
    need(z['minimum_Linf_px'][1] == 0 and z['minimum_Linf_px'][2] == RHO
         and z['minimum_Linf_px'][3] == 4.25 and z['minimum_Linf_px'][4] == 2, 'exact distance cases')
    need(z['minimum_delta_px'][5].tolist() == [0.,4.], 'per-axis y movement')
    cut = geometry(np.array([[1.,1.,2.,2.]]), np.array([[-1.,1.,2.,2.]]), (64,96))
    need(cut['full_visible'][0] and not cut['full_visible_input'][0]
         and cut['visible_ratio_input'][0] == 0, 'input boundary observation')
    # One normalized center separation is 3, 3.1875, 3.75 input pixels at these W.
    for h,w in ((512,512),(544,544),(640,640),(512,640)):
        norm_r = np.array([[.5,.5,2/w,2/h]])
        norm_g = norm_r.copy(); norm_g[:,0] += 6/1024
        zz = geometry(norm_r*np.array([w,h,w,h]), norm_g*np.array([w,h,w,h]), (h,w))
        need(abs(zz['minimum_Linf_px'][0]-(w*6/1024-2)) <= NEAR_PX, 'actual pixel conversion')
    nodes = array(native.weighting_function(32, torch.tensor([.5]), torch.tensor([4.])))
    need(nodes.shape == (33,) and nodes[0] == -4 and nodes[-1] == 4, 'native FDR endpoints')
    return dict(size_infeasible_NA=True, zero_shift=True, exact_rho_boundary=True,
        above_rho=True, visible_but_unrepairable=True, repairable_but_partial=True,
        repairable_and_full=True, y_axis=True, actual_shapes_and_nonsquare=True,
        input_clipping=True, independent_corner_certificate=True, native_endpoints=True)


def summarize(z, mask, gt_indices):
    outside, repair, full = ~z['native_inside'], z['repairable_rho4'], z['full_visible']
    joint = mask & outside & repair & full
    count = lambda condition: int(np.count_nonzero(mask & condition))
    result = dict(slots=int(mask.sum()), unique_GT_exposures=len(set(gt_indices[mask].tolist())),
        size_infeasible=count(~z['size_feasible']), native_inside=count(~outside), native_outside=count(outside),
        repairable_rho4=count(repair), outside_repairable=count(outside & repair),
        outside_repairable_full=count(outside & repair & full),
        outside_repairable_full_unique_GT_exposures=len(set(gt_indices[joint].tolist())),
        outside_repairable_full_in_input=count(outside & repair & z['full_visible_input']),
        outside_repairable_not_full=count(outside & repair & ~full),
        outside_not_repairable_full=count(outside & ~repair & full),
        outside_not_repairable_not_full=count(outside & ~repair & ~full),
        full=count(full), partial=count((z['visible_ratio'] > 0) & ~full), none=count(z['visible_ratio'] == 0),
        full_input=count(z['full_visible_input']),
        partial_input=count((z['visible_ratio_input'] > 0) & ~z['full_visible_input']),
        none_input=count(z['visible_ratio_input'] == 0),
        nominal_full_but_not_input_full=count(full & ~z['full_visible_input']))
    for name in ('near_size_boundary','near_support_boundary','near_rho_boundary','exactly_rho_boundary','near_window_boundary'):
        result[name] = count(z[name])
    for name in ('minimum_Linf_px','visible_ratio','visible_ratio_input'):
        result[name] = stats(z[name][mask])
    return result


def analyze(output, target, pairs, image_index, hw, box, arrays, prefix):
    pairs = np.asarray(pairs, dtype=np.int64)
    need(pairs.ndim == 2 and pairs.shape[1] == 2, 'fixed pair schema')
    q,g = pairs.T
    need(len(set(q.tolist())) == len(q), 'query uniqueness')
    scale = np.array([hw[1],hw[0],hw[1],hw[0]], dtype=np.float64)
    ref = array(output['ref_points'][image_index])[q].astype(np.float64)*scale
    gt = array(target['boxes'])[g].astype(np.float64)*scale
    z = geometry(ref, gt, hw)
    pred_t = output['pred_boxes'][image_index, torch.as_tensor(q)]
    gt_t = target['boxes'][torch.as_tensor(g)]
    need(bool((pred_t[:,2:] > 0).all()), 'invalid paired prediction is not native IoU0')
    iou = array(box.box_iou(box.box_cxcywh_to_xyxy(pred_t), box.box_cxcywh_to_xyxy(gt_t))[0].diag())
    need(iou.dtype == np.float32 and np.isfinite(iou).all(), 'native FP32 paired IoU')
    actual = np.min(gt[:,2:], axis=1) < 4
    main = actual & (iou == 0) & ~z['native_inside']
    masks = dict(all=np.ones(len(q),dtype=bool), actual_sub4=actual,
        native_IoU0_outside=(iou == 0) & ~z['native_inside'], main_actual_sub4_IoU0_outside=main)
    original = target.get('original_sub4')
    if original is not None:
        need(torch.is_tensor(original) and original.dtype == torch.bool and original.shape == (len(target['labels']),), 'original_sub4 explicit label')
        masks['original_sub4'] = array(original)[g]
    labels = array(target['labels'])[g]
    z.update(pairs=pairs, GT_class=labels, native_IoU_FP32=iou, actual_sub4=actual, main_mask=main)
    for key,value in z.items():
        arrays[prefix+'_'+key] = value
    if original is not None:
        arrays[prefix+'_original_sub4'] = masks['original_sub4']
    qualified = np.flatnonzero(main & z['repairable_rho4'] & z['full_visible'])
    witnesses = [dict(query=int(q[i]),GT=int(g[i]),GT_class=int(labels[i]),ref_px=ref[i],GT_px=gt[i],
        minimum_delta_px=z['minimum_delta_px'][i],minimum_Linf_px=z['minimum_Linf_px'][i],native_IoU_FP32=float(iou[i]),
        full_visible=True,full_visible_input=bool(z['full_visible_input'][i]),visible_ratio_input=z['visible_ratio_input'][i],
        near_rho_boundary=bool(z['near_rho_boundary'][i])) for i in qualified]
    return dict(cohorts={key:summarize(z,mask,g) for key,mask in masks.items()},
        original_sub4_available=original is not None, original_sub4_status='explicit_label' if original is not None else 'NA',
        main_witnesses=witnesses)


def pooled(rows):
    keys = [key for key,value in rows[0].items() if isinstance(value, int)]
    result = {key:sum(row[key] for row in rows) for key in keys}
    for key in ('minimum_Linf_px','visible_ratio','visible_ratio_input'):
        total = sum(row[key]['finite_n'] for row in rows)
        result[key] = dict(n=sum(row[key]['n'] for row in rows),finite_n=total,
            NA_n=sum(row[key]['NA_n'] for row in rows),
            mean=sum((row[key]['mean'] or 0)*row[key]['finite_n'] for row in rows)/total if total else None,
            pooled_quantiles_not_computed=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic-only', action='store_true')
    parser.add_argument('--out', type=pathlib.Path, default=ROOT/'260_BASE24_pre_center_observation_probe')
    args = parser.parse_args()
    started,stage = time.monotonic(),'resources_and_frozen_sources'
    resources = H.resource_limits()
    need(resources['priority'] == 'below_normal' and torch.version.cuda is None
         and not torch.is_grad_enabled() and P.M.nx.__version__ == '3.3'
         and P.A.scipy.__version__ == '1.13.1', 'frozen CPU runtime')
    files,expected = V.manifest()
    files.update(reuse237=files['script'],protocol236=files['protocol'],script=pathlib.Path(__file__),
                 protocol=PROTOCOL,floor244=FLOOR_RESULT)
    expected.update(reuse237=SHA237,floor244=SHA244)
    before = {key:sha(path) for key,path in files.items()}
    need(all(before[key] == digest for key,digest in expected.items()), 'frozen input/source identity')
    native = H.load_native_math()
    box = sys.modules[native.__package__+'.box_ops']
    tests = dict(center_geometry=synthetic(native),saved_GO=V.compatibility_self_checks())
    if args.synthetic_only:
        need(all(sha(path) == before[key] for key,path in files.items()), 'source changed during synthetic')
        print(H.json.dumps(clean(dict(state='synthetic_only',tests=tests,resources=resources,counts=COUNTS,
              source_sha256=before)),ensure_ascii=False),flush=True)
        return
    outdir = args.out.resolve()
    need(outdir.parent == ROOT and outdir.name.startswith('260_') and not outdir.exists(), 'new260 directory only')
    outdir.mkdir(exist_ok=False)
    batches,artifacts = [],{}
    try:
        save(outdir/'00_inputs_and_synthetic.json',dict(paths={k:str(p.resolve()) for k,p in files.items()},
            sha256=before,tests=tests,resources=resources,rho_actual_input_pixels=RHO,near_boundary_audit_pixels=NEAR_PX))
        stage='frozen_metadata'
        recipes,package = P.read_json(files['recipe']),P.read_json(files['package'])
        shapes,audit = P.input_shapes(),V.load_union_audit()
        oldfloor = P.read_json(FLOOR_RESULT)
        need(oldfloor['state'] == 'complete' and oldfloor['screen_status'] == 'mixed'
             and oldfloor['necessary_screen_both'] is False, '244 historical negative preserved')
        package_hashes={row['relative']:row['sha256'] for row in package['files']}
        for key in ('matcher','criterion','box_ops','dfine_utils'):
            need(package_hashes['native/engine/deim/'+files[key].name] == before[key], 'native package identity')
        for letter,seed in (('A',1),('B',2)):
            predrows,analysisrows,provenance = P.metadata(letter,seed)
            recipe=recipes[str(seed)]; compute,union=P.native_readers(box,recipe)
            for batch in range(8):
                P.M.guard(); stage=f'{letter}_development_batch{batch:02d}'
                shape=shapes[batch]['shape']; hw=(shape[2],shape[3])
                need(predrows[batch]['input_sha256'] == shapes[batch]['sha256'], 'actual shape/input identity')
                predpath=P.DATA/letter/'frozen_predictions/development'/predrows[batch]['relative']
                analysispath=P.DATA/letter/'development'/analysisrows[batch]['relative']
                need(sha(predpath)==predrows[batch]['sha256'] and sha(analysispath)==analysisrows[batch]['sha256'], 'batch before content')
                prior=P.read_json(analysispath)
                value=torch.load(predpath,map_location='cpu',weights_only=False); COUNTS['prediction_cache_loads']+=1
                COUNTS['native_matcher_calls']+=1
                output,targets,original,costs,denom=V.validate_batch(value,prior,batch,seed,predrows[batch],recipe,compute,union,audit)
                need(H.scalar(output['up'],'up') == .5 and abs(H.scalar(output['reg_scale'],'reg_scale')) == 4
                     and recipe['DFINETransformer']['reg_max'] == 32, 'native fixed support constants')
                arrays,images={},[]
                for i,target in enumerate(targets):
                    pairsets=dict(GO=original[i],own=np.stack(prior['branch_matches'][i]['final'],axis=1))
                    images.append(dict(image_index=i,GT=len(target['labels']),GT_over300=len(target['labels'])>300,
                        pairs={name:analyze(output,target,pairs,i,hw,box,arrays,f'image{i}_{name}') for name,pairs in pairsets.items()}))
                report=dict(letter=letter,seed=seed,batch=batch,shape=shape,source_ids=predrows[batch]['source_ids'],
                    prediction_sha256=predrows[batch]['sha256'],input_sha256=predrows[batch]['input_sha256'],
                    provenance=provenance,denominators=denom,images=images,
                    missing_metadata=['natural_original_image_HL','complete_Mosaic_Mixup_provenance_and_inverse_transform','original_pixel_content'])
                stem=f'{letter}_batch{batch:02d}'
                save(outdir/(stem+'.json'),report); np.savez(outdir/(stem+'_arrays.npz'),**arrays)
                for suffix in ('.json','_arrays.npz'):
                    artifacts[stem+suffix]=sha(outdir/(stem+suffix))
                batches.append(report); del value,output,targets,costs,arrays; gc.collect()
                print(stage,'complete',flush=True)
        need(COUNTS['prediction_cache_loads']==COUNTS['native_matcher_calls']==16 and len(batches)==16, 'exact cache budget')
        summaries={}
        for letter in ('A','B'):
            images=[image for batch in batches if batch['letter']==letter for image in batch['images']]
            summary=dict(inputs=len(images),GT=sum(image['GT'] for image in images),pairs={})
            for pair_name in ('GO','own'):
                rr=[image['pairs'][pair_name] for image in images]
                summary['pairs'][pair_name]=dict(cohorts={cohort:pooled([row['cohorts'][cohort] for row in rr])
                    for cohort in ('all','actual_sub4','native_IoU0_outside','main_actual_sub4_IoU0_outside')},
                    original_sub4_available_inputs=sum(row['original_sub4_available'] for row in rr),
                    original_sub4_all_inputs_status='available' if all(row['original_sub4_available'] for row in rr) else 'NA_or_incomplete')
            mainrow=summary['pairs']['GO']['cohorts']['main_actual_sub4_IoU0_outside']
            need(mainrow['slots'] == oldfloor['summaries'][letter]['arms']['N']['main_cohort']
                 == {'A':1170,'B':1066}[letter], 'main cohort identity differs from244')
            summary['necessary_screen_rho4']=mainrow['outside_repairable_full'] >= 1
            summary['nominal_qualifiers_full_in_actual_input']=mainrow['outside_repairable_full_in_input']
            summaries[letter]=summary
        passed=[summaries[letter]['necessary_screen_rho4'] for letter in ('A','B')]
        after={key:sha(path) for key,path in files.items()}
        need(before == after, 'source/input changed')
        need(all(COUNTS[key]==0 for key in COUNTS if key not in ('prediction_cache_loads','native_matcher_calls'))
             and not torch.is_grad_enabled(), 'zero image/model/gradient/update budget')
        save(outdir/'01_results.json',dict(state='complete',summaries=summaries,
            necessary_screen_both=all(passed),screen_status='both' if all(passed) else ('mixed' if any(passed) else 'neither'),
            rho_actual_input_pixels=RHO,screen_window='original_pre_box_expanded_by4_each_side_nominal',
            clipped_input_visibility_reported_separately=True,closed_support_only=True,
            deployable_center_rule=False,new_pixel_information_evidence=False,training_authorized=False,AP=False,
            saved_GO_used=True,current_Windows_GO_used=False,counts=COUNTS,resources=resources,
            sources_unchanged=True,sha256_before=before,sha256_after=after,synthetic_self_checks=tests,
            artifact_sha256=artifacts,seconds=time.monotonic()-started,
            RSS_observed_peak_bytes=P.M.RSS_PEAK,
            scope='necessary center/support/window geometry only; no center-head, image, PMF, loss or gradient evaluation'))
        save(outdir/'02_output_sha256.json',{path.name:sha(path) for path in outdir.iterdir() if path.is_file()})
    except BaseException:
        after={key:sha(path) if path.exists() else None for key,path in files.items()}
        save(outdir/'FAILED.json',dict(state='error_no_method_conclusion',stage=stage,counts=COUNTS,
            traceback=traceback.format_exc(),sha256_before=before,sha256_after=after,sources_unchanged=before==after,
            completed_batches=[dict(letter=b['letter'],batch=b['batch']) for b in batches],seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
