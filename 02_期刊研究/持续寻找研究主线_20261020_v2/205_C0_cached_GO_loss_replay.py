"""204 frozen native GO loss replay. Written only; root reviews and executes."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
os.environ['MPLBACKEND'] = 'Agg'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[key] = '2'
sys.dont_write_bytecode = True
import argparse, collections, copy, csv, gc, hashlib, json, math, pathlib, time, traceback
ROOT = pathlib.Path(__file__).resolve().parent
COMMON = ROOT.parent / 'code/dn_pre_frozen_diagnostic_20261020_v1'
NATIVE = COMMON / 'native/engine/deim'
RECIPE = ROOT.parent / '相位机制论文_20261018/data/source_snapshot/common.yml'
GO = ROOT / '188_C0_true_GO_support_cache'
ASSIGN = ROOT / '193_C0_support_assignment_exact'
PROTOCOL = ROOT / '204_真实GO损失回放_事前协议.txt'
ARMS = {'G0': 'G0', 'G': 'G', 'I': 'final_IoU_face', 'R': 'ref_IoU_face', 'C': 'native_cost_face'}
ATOL, RTOL = 5e-9, 5e-5
COUNTS = dict(model_forwards=0, model_vjps=0, checkpoint_reads=0, optimizer_steps=0, criterion_calls=0, cached_output_autograd_calls=0, synthetic_autograd_calls=0)
def need(ok, message):
    if not bool(ok): raise RuntimeError(message)
def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()
EXPECTED = {NATIVE/'deim_criterion.py': '16cdd80b2197b8def8ee3f5f19f6311f5221c3da08abacb99f50d346610a2c30',
            NATIVE/'matcher.py': 'b20d4501bbaf1430daf63f1c82394762695a5b793a472bbba152383841983200',
            NATIVE/'box_ops.py': '398fc107fc003b0fabaf36154594d50a8e0aa54195056b343031960c0aff0975',
            NATIVE/'dfine_utils.py': 'ad87ec5a50b1970e76b06f7242298ffa332beaf78c828ac3e8d41ac656ec5f6c',
            RECIPE: '25c158ef1cae63595512e2a57f5a60507cc06978b68724edb705923d44ea3157'}
need(all(sha(p) == digest for p, digest in EXPECTED.items()), 'native source/recipe changed')
sys.path[:0] = [str(COMMON/'runtime'), str(COMMON/'native')]
import numpy as np
import torch, psutil, yaml
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.matcher import HungarianMatcher
from engine.deim.box_ops import box_iou, box_cxcywh_to_xyxy
def cpu(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().clone()
    if isinstance(x, dict): return {k: cpu(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return type(x)(cpu(v) for v in x)
    return x
def plain(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, pathlib.Path): return str(x)
    if isinstance(x, dict): return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [plain(v) for v in x]
    return x
def save(path, value):
    with pathlib.Path(path).open('x', encoding='utf-8') as f: json.dump(plain(value), f, ensure_ascii=False, indent=2, allow_nan=False)
def eq(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor): return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a, b))
    if isinstance(a, dict) and isinstance(b, dict): return set(a) == set(b) and all(eq(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)): return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
    return a == b
def guard(): need(psutil.virtual_memory().available / 2**30 > 8, 'RAM guard')
def resources():
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    p = psutil.Process(); p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    need({13, 15}.issubset(p.cpu_affinity()), 'affinity'); p.cpu_affinity([13, 15]); guard()
    need(torch.version.cuda is None and sorted(p.cpu_affinity()) == [13, 15], 'CPU only')
    return dict(threads=torch.get_num_threads(), interop=torch.get_num_interop_threads(), affinity=p.cpu_affinity(), priority='BelowNormal', available_RAM_GiB=psutil.virtual_memory().available/2**30, torch=torch.__version__)
def stages(out):
    result = {'final': out}
    for key in ('aux_outputs', 'enc_aux_outputs', 'dn_outputs'):
        result.update({f'{key}.{i}': v for i, v in enumerate(out.get(key, []))})
    result.update({key: out[key] for key in ('pre_outputs', 'dn_pre_outputs') if key in out})
    return result
def leaves_from(record):
    out = cpu(record['native_outputs']); leaves = {}
    for stage, value in stages(out).items():
        for key in ('pred_boxes', 'pred_logits', 'pred_corners'):
            if key in value:
                value[key] = value[key].detach().clone().requires_grad_(True); leaves[stage+'.'+key] = value[key]
    return out, leaves
def tensor_sha(t): return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()
def grad(loss, leaves, synthetic=False, retain=True):
    key = 'synthetic_autograd_calls' if synthetic else 'cached_output_autograd_calls'
    limit = 6 if synthetic else 80
    need(COUNTS[key] < limit, 'autograd call budget'); COUNTS[key] += 1
    values = torch.autograd.grad(loss, tuple(leaves.values()), retain_graph=retain, allow_unused=True)
    result = {k: torch.zeros_like(t) if g is None else g.detach().clone() for (k, t), g in zip(leaves.items(), values)}
    need(all(bool(torch.isfinite(g).all()) for g in result.values()), 'nonfinite output gradient')
    return result
def norm(t): return float(torch.linalg.vector_norm(t.detach().double()))
def gradient_summary(values): return {k: dict(norm=norm(v), max_abs=float(v.abs().max()), nonzero=int(torch.count_nonzero(v)), sha256=tensor_sha(v)) for k, v in values.items()}
def close_report(actual, expected):
    error = actual.double() - expected.double()
    return dict(allclose=bool(torch.allclose(actual, expected, atol=ATOL, rtol=RTOL)), max_abs=float(error.abs().max()), L2_error=norm(error), actual_L2=norm(actual), expected_L2=norm(expected))
def dense_target(left, wr, wl, bins=33):
    t = torch.zeros((left.numel(), bins), dtype=wr.dtype)
    t.scatter_add_(1, left.long().reshape(-1, 1), wl.reshape(-1, 1))
    t.scatter_add_(1, (left.long()+1).reshape(-1, 1), wr.reshape(-1, 1))
    need(float((t.sum(-1)-1).abs().max()) <= 1e-7, 'native FGL target mass')
    return t
def terms(p, t0, w0, t1, w1, coef):
    reweight = coef * (w1-w0)[:, None, None] * (p-t0)
    transfer = coef * w1[:, None, None] * (t0-t1)
    return reweight, transfer
def synthetic():
    reports = []
    z = torch.linspace(-2, 2, 3*4*33, dtype=torch.float32).reshape(3, 4, 33).requires_grad_(True)
    left0 = torch.arange(12).remainder(31).float(); left1 = (left0+3).remainder(31)
    wr0 = torch.full((12,), .25); wr1 = torch.full((12,), .625)
    w0 = torch.tensor([.2, .3, 0.]); w1 = torch.tensor([.7, 0., .4]); coef = .15/3
    for case in ('both', 'same_target', 'same_weight'):
        l1, r1 = (left0, wr0) if case == 'same_target' else (left1, wr1)
        wa = w0 if case == 'same_weight' else w1
        t0 = dense_target(left0, wr0, 1-wr0).reshape(3,4,33); t1 = dense_target(l1, r1, 1-r1).reshape(3,4,33)
        def loss(left, wr, weight):
            return .15 * DEIMCriterion.unimodal_distribution_focal_loss(None, z.reshape(-1,33), left, wr, 1-wr, weight[:,None].expand(-1,4).reshape(-1), avg_factor=3)
        g0 = grad(loss(left0,wr0,w0), {'z':z}, True, False)['z']; g1 = grad(loss(l1,r1,wa), {'z':z}, True, False)['z']
        p = z.detach().softmax(-1); r, t = terms(p,t0,w0,t1,wa,coef)
        check = close_report(g1-g0,r+t); need(check['allclose'] and close_report(g0,coef*w0[:,None,None]*(p-t0))['allclose'], 'synthetic native FGL chain rule')
        need(not bool(torch.count_nonzero(g1[wa == 0])), 'zero-IoU FGL must be zero')
        if case == 'same_target': need(not bool(torch.count_nonzero(t)), 'pure reweight control')
        if case == 'same_weight': need(not bool(torch.count_nonzero(r)), 'pure target control')
        reports.append(dict(case=case, check=check))
    return dict(native_coefficient='.15/num_go, no /4', cases=reports, zero_IoU_zero_gradient=True)
class ReplayCriterion(DEIMCriterion):
    def configure(self, out, record, arm_indices):
        self.record, self.arm_indices = record, arm_indices
        self.pointer = {id(v['pred_boxes']): k for k,v in stages(out).items()}
        self.calls, self.own = [], {}
        self.handle = self.matcher.register_forward_hook(self.observe_matcher)
    def observe_matcher(self, module, inputs, result):
        name = self.pointer[id(inputs[0]['pred_boxes'])]; actual = result['indices']
        need(eq(actual,self.record['matching']['own_by_stage'][name]), 'own matcher changed: '+name)
        self.own[name] = cpu(actual)
    def _get_go_indices(self, indices, auxiliary):
        native = super()._get_go_indices(indices,auxiliary)
        need(eq(native,self.record['matching']['indices_go']), 'native GO no longer reproduces188')
        return cpu(self.arm_indices)
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        values = super().get_loss(loss,outputs,targets,indices,num_boxes,**kwargs)
        need(all(bool(torch.isfinite(v).all()) for v in values.values()), 'native nonfinite loss before nan_to_num')
        self.calls.append(dict(stage=self.pointer[id(outputs['pred_boxes'])], loss=loss, indices=cpu(indices), denominator=float(num_boxes), raw_values={k:float(v.detach()) for k,v in values.items()}))
        return values
def build(recipe):
    cc = copy.deepcopy(recipe['DEIMCriterion']); mc = cc.pop('matcher'); need(mc.pop('type') == 'HungarianMatcher','matcher recipe')
    c = ReplayCriterion(matcher=HungarianMatcher(**mc,use_focal_loss=True),num_classes=10,**cc)
    need(not c.state_dict() and str(c.load_state_dict({},strict=True)) == '<All keys matched successfully>', 'empty native criterion state')
    need(c.use_uni_set and c.boxes_weight_format is None and c.weight_dict['loss_fgl'] == .15 and c.losses == ['mal','boxes','local'], 'criterion configuration')
    return c
def comparison(actual, prior):
    need(set(actual) == set(prior), 'loss keys differ')
    return {k:dict(actual=actual[k],prior=prior[k],difference=actual[k]-prior[k],exact=actual[k] == prior[k]) for k in actual}
def assignment_maps(record, image, arrays):
    q0,g0 = record['matching']['indices_go'][0]; old = dict(zip(q0.tolist(),g0.tolist()))
    need(image['input_sha256'] == record['input_sha256'] and np.array_equal(np.asarray(image['original_pairs']),np.stack((q0.numpy(),g0.numpy()),1)), '193/188 original identity')
    mappings, edge = {}, {}
    for arm,name in ARMS.items():
        pairs = [pair for group in image['groups'] for pair in group['domains']['physical64']['pairs'][name]]
        mapping = dict(pairs)
        need(len(mapping) == len(pairs) == len(old) and set(mapping) == set(old) and sorted(mapping.values()) == sorted(old.values()), 'GO budgets changed')
        need(all(int(record['target']['labels'][mapping[q]]) == int(record['target']['labels'][old[q]]) for q in old), 'per-query target class changed')
        mappings[arm] = [(q0.clone(),torch.tensor([mapping[q] for q in q0.tolist()],dtype=torch.long))]
    need(eq(mappings['G0'],record['matching']['indices_go']), 'G0 mapping differs')
    for group in image['groups']:
        prefix = f"true_GO_image{record['image_index']}_class{group['class_id']}_"
        qs,gs = arrays[prefix+'q_ids'], arrays[prefix+'GT_ids']; support = arrays[prefix+'physical64']; iou = arrays[prefix+'final_IoU32']
        for i,q in enumerate(qs):
            for j,g in enumerate(gs): edge[(int(q),int(g))] = (bool(support[i,j]),float(iou[i,j]))
    return mappings,edge
def fgl_data(c, out, record, indices):
    q,g = indices[0]; left,wr,wl = c.fgl_targets
    t = torch.zeros((300,4,33),dtype=torch.float32); t[q] = dense_target(left,wr,wl).reshape(-1,4,33)
    w = torch.zeros(300,dtype=torch.float32)
    w[q] = box_iou(box_cxcywh_to_xyxy(out['pred_boxes'][0,q].detach()),box_cxcywh_to_xyxy(record['target']['boxes'][g]))[0].diag()
    return t,w
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--synthetic-only',action='store_true'); ap.add_argument('--out',type=pathlib.Path,default=ROOT/'206_C0_cached_GO_loss_replay'); args = ap.parse_args()
    started = time.monotonic(); resource = resources(); tests = synthetic()
    if args.synthetic_only: print(json.dumps(plain(dict(synthetic=tests,counts=COUNTS)))); return
    outdir = args.out.resolve(); need(outdir.parent == ROOT and outdir.name.startswith('206_') and not outdir.exists(),'new206 only'); outdir.mkdir()
    stage = 'input_validation'; hashes = {}; reports, tensor_records, pair_rows = [], [], []
    try:
        def lock(path, expected=None):
            path = pathlib.Path(path).resolve(); digest = sha(path); need(expected is None or digest == expected,'SHA mismatch '+str(path)); hashes[str(path)] = digest
        for path,digest in EXPECTED.items(): lock(path,digest)
        lock(PROTOCOL); lock(__file__)
        lock(GO/'05_output_sha256.json','b655c054be540e0332256db51027433e00b876182b2e36bea6b55bff3d1bb6ba')
        manifest = json.loads((GO/'05_output_sha256.json').read_text(encoding='utf-8'))
        for name,digest in manifest.items(): lock(GO/name,digest)
        need(manifest['01_cache.pt'] == '8a100f920dff9c961cc13ac5814e338dd37fc0bf5f390cc453a607f546b0ced9','188 cache identity')
        lock(ASSIGN/'00_results.json','660884b5412275dccdf16e91ddb979346ad869b1153c11a5642f2ed680c9b4bc')
        r193 = json.loads((ASSIGN/'00_results.json').read_text(encoding='utf-8')); need(r193['state'] == 'complete' and r193['sources_unchanged'] is True,'193 incomplete/source change')
        for name,digest in r193['artifact_sha256'].items(): lock(ASSIGN/name,digest)
        need(r193['artifact_sha256']['02_true_GO_assignments_and_certificates.json'] == 'cfc41cbb0409c46041b003a0b7b805e3a41b26ebb3672e364a2b3eb17934509e' and r193['artifact_sha256']['03_pairwise_graphs.npz'] == 'f21a2a9fed59153e55b147c535a01c8832091ca07dd8e5fa1b6dcf9a7c9673ad','193 fixed assignments')
        for key in ('script','protocol','collector187','recipe','matcher','criterion','box_ops','dfine_utils'): lock(r193['source_paths'][key],r193['sha256_before'][key])
        for module in list(sys.modules.values()):
            path = getattr(module,'__file__',None)
            if path and pathlib.Path(path).suffix == '.py' and (COMMON/'native').resolve() in pathlib.Path(path).resolve().parents: lock(path)
        save(outdir/'00_sources_and_synthetic.json',dict(sources=hashes,synthetic=tests,resources=resource))
        data = torch.load(GO/'01_cache.pt',map_location='cpu',weights_only=False)
        need(data['schema'] == 'true-GO-train-ref-v1' and data['weight_key'] == 'model_online' and data['GT'] == 902 and data['original_sub4'] == 453,'188 schema')
        need([r['image_index'] for r in data['records']] == [0,1,2,3] and [r['matching']['num_boxes_go'] for r in data['records']] == [172,290,300,70],'four actual GO budgets')
        recipe = yaml.safe_load(RECIPE.read_text(encoding='utf-8-sig'))
        assignment = json.loads((ASSIGN/'02_true_GO_assignments_and_certificates.json').read_text(encoding='utf-8'))
        arrays = np.load(ASSIGN/'03_pairwise_graphs.npz',allow_pickle=False)
        for record in data['records']:
            guard(); index = record['image_index']; image = next(x for x in assignment['images'] if x['image_index'] == index)
            mappings,edge = assignment_maps(record,image,arrays); base = None
            for arm in ARMS:
                stage = f'image{index}_{arm}'; guard(); output,leaves = leaves_from(record); c = build(recipe)
                c._clear_cache(); c.configure(output,record,mappings[arm])
                try:
                    need(COUNTS['criterion_calls'] < 20,'criterion budget'); COUNTS['criterion_calls'] += 1
                    losses = c(output,[cpu(record['target'])],epoch=94,step=0,epoch_step=1,global_step=None)
                    need(set(c.own) == set(record['matching']['own_by_stage']),'all own stages required')
                    values = {k:float(v.detach()) for k,v in losses.items()}
                    old_calls = {(v['stage'],v['loss']):v for v in record['loss_calls']}
                    need(len(c.calls) == len(old_calls) and {(v['stage'],v['loss']) for v in c.calls} == set(old_calls),'all native loss calls required')
                    for call in c.calls:
                        previous = old_calls[(call['stage'],call['loss'])]
                        ordinary = call['stage'] in ('final','pre_outputs') or call['stage'].startswith('aux_outputs.')
                        uses_go = (ordinary and call['loss'] in ('boxes','local')) or (call['stage'].startswith('enc_aux_outputs.') and call['loss'] == 'boxes')
                        expected = mappings[arm] if uses_go else previous['indices']
                        need(eq(call['indices'],expected) and call['denominator'] == previous['denominator'],'native loss routing/denominator differs')
                    if arm == 'G0':
                        parity = comparison(values,record['loss_values'])
                        raw_parity = {v['stage']+'/'+v['loss']:comparison(v['raw_values'],old_calls[(v['stage'],v['loss'])]['raw_loss_values_before_weight_dict']) for v in c.calls}
                        save(outdir/f'01_baseline_image_{index}.json',dict(loss_comparison=parity,raw_loss_comparison=raw_parity,own_exact=True,GO_exact=True))
                        need(all(v['exact'] for v in parity.values()),'baseline_precision_review_no_method_conclusion')
                        need(all(v['exact'] for entries in raw_parity.values() for v in entries.values()),'baseline raw loss precision review')
                    t,w = fgl_data(c,output,record,mappings[arm]); p = output['pred_corners'][0].detach().reshape(300,4,33).softmax(-1)
                    coefficient = c.weight_dict['loss_fgl']/record['matching']['num_boxes_go']
                    fgl = grad(losses['loss_fgl'],{'corners':output['pred_corners']})['corners'][0].reshape(300,4,33)
                    manual = coefficient*w[:,None,None]*(p-t); check = close_report(fgl,manual); need(check['allclose'],'native FGL/manual mismatch')
                    need(not bool(torch.count_nonzero(fgl[w == 0])),'zero IoU has nonzero final FGL gradient')
                    mal_keys = [k for k in losses if k.startswith('loss_mal') and '_dn_' not in k]; dn_keys = [k for k in losses if '_dn_' in k]
                    mal_grad = grad(sum(losses[k] for k in mal_keys),leaves)
                    dn_grad = grad(sum(losses[k] for k in dn_keys),leaves)
                    total_grad = grad(sum(losses.values()),leaves,retain=False)
                    need(all(v.grad is None for v in leaves.values()),'autograd must not accumulate leaf.grad')
                    if arm == 'G0': base = dict(values=values,t=t.clone(),w=w.clone(),fgl=fgl.clone(),mal_grad=mal_grad,dn_grad=dn_grad)
                    invariant_values = all(values[k] == base['values'][k] for k in mal_keys+dn_keys)
                    mal_exact = all(torch.equal(mal_grad[k],base['mal_grad'][k]) for k in leaves)
                    dn_exact = all(torch.equal(dn_grad[k],base['dn_grad'][k]) for k in leaves)
                    need(invariant_values and mal_exact and dn_exact,'MAL/DN value or gradient invariance failed')
                    rw,tr = terms(p,base['t'],base['w'],t,w,coefficient); delta = fgl-base['fgl']; dcheck = close_report(delta,rw+tr); need(dcheck['allclose'],'FGL delta decomposition mismatch')
                    q,g = mappings[arm][0]; oldg = dict(zip(*[x.tolist() for x in mappings['G0'][0]])); rows = []
                    for query,gt in zip(q.tolist(),g.tolist()):
                        prior = oldg[query]; old_support,_ = edge[(query,prior)]; support,iou193 = edge[(query,gt)]
                        need(float(w[query]) == iou193,'193/native FGL IoU disagreement')
                        row = dict(image_index=index,arm=arm,query=query,old_GT=prior,new_GT=gt,GT_class=int(record['target']['labels'][gt]),original_sub4=bool(record['original_sub4'][gt]),owner_changed=gt != prior,old_supported=old_support,new_supported=support,gained_support=not old_support and support,old_IoU=float(base['w'][query]),new_IoU=float(w[query]),new_IoU_zero=bool(w[query] == 0),target_L2_change=norm(t[query]-base['t'][query]),reweight_L2=norm(rw[query]),target_transfer_L2=norm(tr[query]),FGL_delta_L2=norm(delta[query]))
                        rows.append(row)
                    pair_rows.extend(rows); dot = float((rw.double()*tr.double()).sum()); nr,nt = norm(rw),norm(tr)
                    report = dict(image_index=index,arm=arm,GO_indices=mappings[arm],GO_denominator=record['matching']['num_boxes_go'],calls=c.calls,weighted_loss=values,loss_delta_from_G0={k:values[k]-base['values'][k] for k in values},total_loss=float(sum(losses.values()).detach()),MAL_values_and_gradients_exact=bool(invariant_values and mal_exact),DN_values_and_gradients_exact=bool(invariant_values and dn_exact),final_FGL_manual_check=check,decomposition_check=dcheck,reweight_L2=nr,target_transfer_L2=nt,delta_L2=norm(delta),term_dot=dot,term_cosine=dot/(nr*nt) if nr and nt else None,total_output_gradient=gradient_summary(total_grad),ordinary_MAL_gradient=gradient_summary(mal_grad),DN_gradient=gradient_summary(dn_grad),changed_supported_zero_IoU=[r for r in rows if r['owner_changed'] and r['new_supported'] and r['new_IoU_zero']],gained_support_zero_IoU=[r for r in rows if r['gained_support'] and r['new_IoU_zero']],changed_owner_count=sum(r['owner_changed'] for r in rows),new_supported_count=sum(r['new_supported'] for r in rows),nonzero_target_transfer_queries=sum(r['target_transfer_L2'] > 0 for r in rows))
                    save(outdir/f'02_image_{index}_{arm}.json',report); reports.append(report)
                    tensor_records.append(dict(image_index=index,arm=arm,final_FGL_gradient=fgl,targets=t,IoU_weight=w,reweight=rw,target_transfer=tr,FGL_delta=delta,total_output_gradients=total_grad))
                finally:
                    c.handle.remove(); c._clear_cache(); need(not c.state_dict(),'criterion state changed')
                del output,leaves,c,losses,mal_grad,dn_grad,total_grad; gc.collect()
                print(stage,'complete',flush=True)
            del base
        arrays.close(); need(COUNTS['criterion_calls'] == 20 and COUNTS['cached_output_autograd_calls'] == 80 and COUNTS['synthetic_autograd_calls'] == 6,'exact replay budget')
        torch.save(dict(records=tensor_records,scope='independent output leaves; no native shared parameter graph'),outdir/'03_output_gradients.pt')
        with (outdir/'04_pair_FGL_decomposition.csv').open('x',encoding='utf-8-sig',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(pair_rows[0])); writer.writeheader(); writer.writerows(pair_rows)
        aggregate = {}
        for arm in ARMS:
            rr = [r for r in reports if r['arm'] == arm]; weighted = {k:sum(r['weighted_loss'][k] for r in rr)/4 for k in rr[0]['weighted_loss']}
            aggregate[arm] = dict(mean_image_weighted_losses=weighted,mean_image_total_loss=sum(r['total_loss'] for r in rr)/4,changed_owner_count=sum(r['changed_owner_count'] for r in rr),new_supported_count=sum(r['new_supported_count'] for r in rr),changed_supported_zero_IoU=sum(len(r['changed_supported_zero_IoU']) for r in rr),gained_support_zero_IoU=sum(len(r['gained_support_zero_IoU']) for r in rr),nonzero_target_transfer_queries=sum(r['nonzero_target_transfer_queries'] for r in rr))
        save(outdir/'05_summary.json',dict(arms=aggregate,GT=902,original_sub4=453,GO_slots=832,method_gate=None,AP_prediction=False,shared_parameter_gradient=False,baseline_exact=True,MAL_DN_invariant=True))
        unchanged = {path:sha(path) == digest for path,digest in hashes.items()}; need(all(unchanged.values()),'input/source changed')
        save(outdir/'06_execution.json',dict(state='complete',resources=resource,counts=COUNTS,source_unchanged=unchanged,criterion_state_sha256=hashlib.sha256(b'').hexdigest(),seconds=time.monotonic()-started,heldout_reads=False,optimizer_constructed=False))
        save(outdir/'07_output_sha256.json',{p.name:sha(p) for p in outdir.iterdir() if p.is_file()})
    except BaseException:
        save(outdir/'FAILED.json',dict(state='failed_no_method_conclusion',stage=stage,counts=COUNTS,traceback=traceback.format_exc(),seconds=time.monotonic()-started)); raise
if __name__ == '__main__': main()
