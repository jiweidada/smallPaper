"""213 adapter: frozen 210 assignments, reuse 205 helpers, never call 205.main."""
import os, sys, importlib.util, pathlib, hashlib, json, argparse, time, traceback, csv, gc, collections
os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
os.environ['MPLBACKEND'] = 'Agg'; sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[key] = '2'
ROOT = pathlib.Path(__file__).resolve().parent
HELPER = ROOT/'205_C0_cached_GO_loss_replay.py'
HELPER_SHA = 'de0223e9c375dcd9f2d18f6bb537af6d9a6d7544d284c801f9830cf5d5d3fb33'
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA: raise RuntimeError('frozen205 changed')
spec = importlib.util.spec_from_file_location('frozen205_repair_helpers', HELPER)
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
torch = H.torch
ARMS = ('G0', 'U_cost', 'U_change', 'G_repair')
ASSIGN = ROOT/'210_C0_monotone_GO_repair'
PROTOCOL = ROOT/'213_单调修复损失回放_事前协议.txt'
FIXED210 = {'00_results.json':'0c79441e97621b06a7b1544361e4bf31a31fd2a019f38692693d253ca902a178',
            '01_assignments_and_certificates.json':'569c8ca4fae3c5c46041fb84743f9295ff2211b660519973152e927694e2f8d4',
            '02_per_query_transitions.json':'a5afbd78fabd173f5fec31185044faf304267b80e88a39d7c8a5f1496c35dd7f'}

def cache_grad(loss, leaves, retain=True):
    H.need(H.COUNTS['cached_output_autograd_calls'] < 64, '214 independent cache-grad budget')
    return H.grad(loss, leaves, retain=retain)

def mappings_for(record, image, query_rows):
    q0,g0 = record['matching']['indices_go'][0]; old = dict(zip(q0.tolist(),g0.tolist()))
    H.need(image['input_sha256'] == record['input_sha256'] and image['event'] == str(record['event']), '210/188 input identity')
    H.need(image['GT'] == len(record['target']['labels']) and image['slots'] == len(old) and image['unique_GT'] == len(set(old.values())), '210/188 budgets')
    H.need(set(query_rows) == set(old) and len(old) == len(q0), 'query capacity1 or row identity')
    maps = {}
    for arm in ARMS:
        pairs = [pair for group in image['groups'] for pair in group['pairs'][arm]]; mapping = dict(pairs)
        H.need(len(mapping) == len(pairs) == len(old) and set(mapping) == set(old) and sorted(mapping.values()) == sorted(old.values()), 'GO query/GT multiset changed')
        for group in image['groups']:
            H.need(set(q for q,g in group['pairs'][arm]) == set(group['q_ids']), 'class group query budget')
            H.need(collections.Counter(g for q,g in group['pairs'][arm]) == dict(zip(group['GT_ids'],group['m_g'])), 'class group GT capacities')
        for q,g in mapping.items():
            row = query_rows[q]; info = row['arms'][arm]; prior = row['arms']['G0']
            H.need(row['original_GT'] == old[q] and prior['GT'] == old[q] and info['GT'] == g, '210 pair/transition discrepancy')
            H.need(row['class_id'] == int(record['target']['labels'][g]) == int(record['target']['labels'][old[q]]), 'query class changed')
            H.need(info['original_sub4'] == bool(record['original_sub4'][g]) and info['changed'] == (g != old[q]), '210 cohort/owner discrepancy')
            H.need(info['old_in'] == prior['new_in'] and info['forbidden_edge'] == (g != old[q] and not info['new_in']), '210 support transition discrepancy')
        maps[arm] = [(q0.clone(),torch.tensor([mapping[q] for q in q0.tolist()],dtype=torch.long))]
    H.need(H.eq(maps['G0'],record['matching']['indices_go']), 'G0 order differs')
    return maps

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--synthetic-only',action='store_true')
    ap.add_argument('--out',type=pathlib.Path,default=ROOT/'215_C0_cached_repair_loss_replay'); args = ap.parse_args()
    started = time.monotonic(); H.need(not any(H.COUNTS.values()), 'fresh independent214 counts required')
    resource = H.resources(); tests = H.synthetic()
    if args.synthetic_only: print(json.dumps(H.plain(dict(synthetic=tests,counts=H.COUNTS)))); return
    outdir = args.out.resolve(); H.need(outdir.parent == ROOT and outdir.name.startswith('215_') and not outdir.exists(), 'new215 only'); outdir.mkdir()
    stage = 'input_validation'; hashes = {}; reports, tensor_records, pair_rows, repairs, protected, gt_checks = [], [], [], [], [], []
    try:
        def lock(path, expected=None):
            path = pathlib.Path(path).resolve(); digest = H.sha(path); H.need(expected is None or digest == expected,'SHA mismatch '+str(path)); hashes[str(path)] = digest
        lock(HELPER,HELPER_SHA); lock(PROTOCOL); lock(__file__)
        for path,digest in H.EXPECTED.items(): lock(path,digest)
        lock(H.GO/'05_output_sha256.json','b655c054be540e0332256db51027433e00b876182b2e36bea6b55bff3d1bb6ba')
        manifest = json.loads((H.GO/'05_output_sha256.json').read_text(encoding='utf-8'))
        for name,digest in manifest.items(): lock(H.GO/name,digest)
        H.need(manifest['01_cache.pt'] == '8a100f920dff9c961cc13ac5814e338dd37fc0bf5f390cc453a607f546b0ced9', '188 identity')
        for name,digest in FIXED210.items(): lock(ASSIGN/name,digest)
        r210 = json.loads((ASSIGN/'00_results.json').read_text(encoding='utf-8'))
        H.need(r210['state'] == 'complete' and r210['sources_unchanged'] is True, '210 incomplete/source change')
        for name,digest in r210['artifact_sha256'].items(): H.need(FIXED210[name] == digest,'210 artifact identity')
        for key,path in r210['source_paths'].items(): lock(path,r210['sha256_before'][key])
        for module in list(sys.modules.values()):
            path = getattr(module,'__file__',None)
            if path and pathlib.Path(path).suffix == '.py' and (H.COMMON/'native').resolve() in pathlib.Path(path).resolve().parents: lock(path)
        H.save(outdir/'00_sources_and_synthetic.json',dict(sources=hashes,synthetic=tests,resources=resource))
        data = torch.load(H.GO/'01_cache.pt',map_location='cpu',weights_only=False)
        H.need(data['schema'] == 'true-GO-train-ref-v1' and data['weight_key'] == 'model_online' and data['GT'] == 902 and data['original_sub4'] == 453,'188 schema')
        H.need([r['image_index'] for r in data['records']] == [0,1,2,3] and [r['matching']['num_boxes_go'] for r in data['records']] == [172,290,300,70], 'actual four GO budgets')
        recipe = H.yaml.safe_load(H.RECIPE.read_text(encoding='utf-8-sig'))
        assignment = json.loads((ASSIGN/'01_assignments_and_certificates.json').read_text(encoding='utf-8'))
        transitions = json.loads((ASSIGN/'02_per_query_transitions.json').read_text(encoding='utf-8'))['queries']
        qr = {(row['image_index'],row['query']):row for row in transitions}
        H.need(assignment['state'] == 'complete' and len(qr) == len(transitions) == 832 and sorted(x['image_index'] for x in assignment['images']) == [0,1,2,3], '210 row/image identities')
        for record in data['records']:
            H.guard(); index = record['image_index']; image = next(x for x in assignment['images'] if x['image_index'] == index)
            qrows = {q:row for (i,q),row in qr.items() if i == index}; maps = mappings_for(record,image,qrows); base = None
            for arm in ARMS:
                stage = f'image{index}_{arm}'; H.guard(); output,leaves = H.leaves_from(record); c = H.build(recipe)
                c._clear_cache(); c.configure(output,record,maps[arm])
                try:
                    H.need(H.COUNTS['criterion_calls'] < 16,'214 criterion budget'); H.COUNTS['criterion_calls'] += 1
                    losses = c(output,[H.cpu(record['target'])],epoch=94,step=0,epoch_step=1,global_step=None)
                    H.need(set(c.own) == set(record['matching']['own_by_stage']), 'all own stages required')
                    values = {k:float(v.detach()) for k,v in losses.items()}; old_calls = {(v['stage'],v['loss']):v for v in record['loss_calls']}
                    H.need(len(c.calls) == len(old_calls) and {(v['stage'],v['loss']) for v in c.calls} == set(old_calls), 'native call identities')
                    for call in c.calls:
                        previous = old_calls[(call['stage'],call['loss'])]
                        ordinary = call['stage'] in ('final','pre_outputs') or call['stage'].startswith('aux_outputs.')
                        go = (ordinary and call['loss'] in ('boxes','local')) or (call['stage'].startswith('enc_aux_outputs.') and call['loss'] == 'boxes')
                        H.need(H.eq(call['indices'],maps[arm] if go else previous['indices']) and call['denominator'] == previous['denominator'], 'native routing/denominator changed')
                    if arm == 'G0':
                        parity = H.comparison(values,record['loss_values'])
                        raw = {v['stage']+'/'+v['loss']:H.comparison(v['raw_values'],old_calls[(v['stage'],v['loss'])]['raw_loss_values_before_weight_dict']) for v in c.calls}
                        H.save(outdir/f'01_baseline_image_{index}.json',dict(loss_comparison=parity,raw_loss_comparison=raw,own_exact=True,GO_exact=True))
                        H.need(all(v['exact'] for v in parity.values()) and all(v['exact'] for entries in raw.values() for v in entries.values()), 'baseline_precision_review_no_method_conclusion')
                    t,w = H.fgl_data(c,output,record,maps[arm]); p = output['pred_corners'][0].detach().reshape(300,4,33).softmax(-1)
                    native_targets = H.cpu(c.fgl_targets); coefficient = c.weight_dict['loss_fgl']/record['matching']['num_boxes_go']
                    fgl = cache_grad(losses['loss_fgl'],{'corners':output['pred_corners']})['corners'][0].reshape(300,4,33)
                    check = H.close_report(fgl,coefficient*w[:,None,None]*(p-t)); H.need(check['allclose'], 'FGL/manual mismatch')
                    H.need(not bool(torch.count_nonzero(fgl[w == 0])), 'zero-IoU FGL nonzero')
                    mal_keys = [k for k in losses if k.startswith('loss_mal') and '_dn_' not in k]; dn_keys = [k for k in losses if '_dn_' in k]
                    mal_grad = cache_grad(sum(losses[k] for k in mal_keys),leaves)
                    dn_grad = cache_grad(sum(losses[k] for k in dn_keys),leaves)
                    total_grad = cache_grad(sum(losses.values()),leaves,retain=False)
                    H.need(all(v.grad is None for v in leaves.values()), 'leaf.grad accumulation')
                    if arm == 'G0': base = dict(values=values,t=t.clone(),w=w.clone(),fgl=fgl.clone(),mal_grad=mal_grad,dn_grad=dn_grad)
                    invariant = all(values[k] == base['values'][k] for k in mal_keys+dn_keys)
                    mal_exact = all(torch.equal(mal_grad[k],base['mal_grad'][k]) for k in leaves); dn_exact = all(torch.equal(dn_grad[k],base['dn_grad'][k]) for k in leaves)
                    H.need(invariant and mal_exact and dn_exact, 'MAL/DN value or output-gradient invariance failed')
                    rw,tr = H.terms(p,base['t'],base['w'],t,w,coefficient); delta = fgl-base['fgl']; dcheck = H.close_report(delta,rw+tr)
                    H.need(dcheck['allclose'], 'FGL delta decomposition mismatch')
                    q,g = maps[arm][0]; rows = []
                    for query,gt in zip(q.tolist(),g.tolist()):
                        prior = qrows[query]['original_GT']; info = qrows[query]['arms'][arm]
                        H.need(float(w[query]) == info['final_IoU'], '210/native FP32 FGL IoU differs')
                        row = dict(image_index=index,arm=arm,query=query,old_GT=prior,new_GT=gt,GT_class=int(record['target']['labels'][gt]),original_sub4=bool(record['original_sub4'][gt]),owner_changed=gt != prior,old_supported=info['old_in'],new_supported=info['new_in'],gained_support=not info['old_in'] and info['new_in'],lost_support=info['old_in'] and not info['new_in'],forbidden_edge=info['forbidden_edge'],old_IoU=float(base['w'][query]),new_IoU=float(w[query]),new_IoU_zero=bool(w[query] == 0),target_L2_change=H.norm(t[query]-base['t'][query]),reweight_L2=H.norm(rw[query]),target_transfer_L2=H.norm(tr[query]),FGL_delta_L2=H.norm(delta[query]))
                        rows.append(row)
                        if arm == 'G_repair':
                            H.need(not row['lost_support'] and not row['forbidden_edge'], 'repair monotonicity/allowed-edge violated')
                            if row['old_supported']: protected.append(row.copy())
                            if row['gained_support']:
                                repairs.append(dict(**row,coefficient=coefficient,t0=base['t'][query],t1=t[query],reweight=rw[query],target_transfer=tr[query],FGL_delta=delta[query],pred_box=record['pred_boxes'][query],ref_point=record['ref_points_initial'][query],old_GT_box=record['target']['boxes'][prior],new_GT_box=record['target']['boxes'][gt]))
                    transition = dict(repaired_old_outside=sum(r['gained_support'] for r in rows),lost_old_inside=sum(r['lost_support'] for r in rows),changed_still_outside=sum(r['owner_changed'] and not r['old_supported'] and not r['new_supported'] for r in rows),forbidden_edges=sum(r['forbidden_edge'] for r in rows),changed_owners=sum(r['owner_changed'] for r in rows))
                    H.need(transition == image['transitions'][arm], '210 transition aggregate differs')
                    old_support = collections.Counter(r['old_GT'] for r in rows if r['old_supported']); new_support = collections.Counter(r['new_GT'] for r in rows if r['new_supported'])
                    for group in image['groups']:
                        for item in group['GT_support'][arm]: H.need(new_support[item['GT']] == item['support_slots'], '210 per-GT support differs')
                    if arm == 'G_repair':
                        for gt in sorted(set(g.tolist())):
                            H.need(new_support[gt] >= old_support[gt], 'per-GT supported-slot count decreases')
                            gt_checks.append(dict(image_index=index,GT=gt,old_support_slots=old_support[gt],new_support_slots=new_support[gt],nondecreasing=True))
                    pair_rows.extend(rows); nr,nt = H.norm(rw),H.norm(tr); dot = float((rw.double()*tr.double()).sum())
                    report = dict(image_index=index,arm=arm,GO_indices=maps[arm],GO_denominator=record['matching']['num_boxes_go'],calls=c.calls,weighted_loss=values,loss_delta_from_G0={k:values[k]-base['values'][k] for k in values},total_loss=float(sum(losses.values()).detach()),MAL_values_and_gradients_exact=bool(invariant and mal_exact),DN_values_and_gradients_exact=bool(invariant and dn_exact),coefficient=coefficient,final_FGL_manual_check=check,decomposition_check=dcheck,reweight_L2=nr,target_transfer_L2=nt,delta_L2=H.norm(delta),term_dot=dot,term_cosine=dot/(nr*nt) if nr and nt else None,total_output_gradient=H.gradient_summary(total_grad),ordinary_MAL_gradient=H.gradient_summary(mal_grad),DN_gradient=H.gradient_summary(dn_grad),transitions=transition,new_supported_count=sum(r['new_supported'] for r in rows),gained_support_zero_IoU=[r for r in rows if r['gained_support'] and r['new_IoU_zero']],changed_supported_zero_IoU=[r for r in rows if r['owner_changed'] and r['new_supported'] and r['new_IoU_zero']],nonzero_target_transfer_queries=sum(r['target_transfer_L2'] > 0 for r in rows))
                    H.save(outdir/f'02_image_{index}_{arm}.json',report); reports.append(report)
                    tensor_records.append(dict(image_index=index,arm=arm,GO_indices=maps[arm],final_FGL_gradient=fgl,native_fgl_left_wr_wl=native_targets,targets=t,IoU_weight=w,reweight=rw,target_transfer=tr,FGL_delta=delta,total_output_gradients=total_grad))
                finally:
                    c.handle.remove(); c._clear_cache(); H.need(not c.state_dict(), 'criterion state changed')
                del output,leaves,c,losses,mal_grad,dn_grad,total_grad; gc.collect(); print(stage,'complete',flush=True)
            del base
        H.need(H.COUNTS['criterion_calls'] == 16 and H.COUNTS['cached_output_autograd_calls'] == 64 and H.COUNTS['synthetic_autograd_calls'] == 6, '214 exact independent budget')
        H.need(all(H.COUNTS[k] == 0 for k in ('model_forwards','model_vjps','checkpoint_reads','optimizer_steps')), 'zero model/CP/update budget')
        H.need(len(repairs) == 5 and len(protected) == 558 and sum(r['owner_changed'] for r in pair_rows if r['arm'] == 'G_repair') == 12, 'frozen210 repair identities/counts')
        torch.save(dict(records=tensor_records,scope='independent output leaves; no shared parameter/model graph'),outdir/'03_output_gradients.pt')
        with (outdir/'04_pair_FGL_decomposition.csv').open('x',encoding='utf-8-sig',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(pair_rows[0])); writer.writeheader(); writer.writerows(pair_rows)
        H.save(outdir/'05_repair_details_and_monotonicity.json',dict(repaired_old_outside=repairs,old_inside_query_preservation=protected,per_GT_support=gt_checks,scope='support monotonicity only; owner may change; not IoU/loss/AP monotonicity'))
        aggregate = {}
        for arm in ARMS:
            rr = [r for r in reports if r['arm'] == arm]; weighted = {k:sum(r['weighted_loss'][k] for r in rr)/4 for k in rr[0]['weighted_loss']}
            aggregate[arm] = dict(mean_image_weighted_losses=weighted,mean_image_total_loss=sum(r['total_loss'] for r in rr)/4,transitions={k:sum(r['transitions'][k] for r in rr) for k in rr[0]['transitions']},new_supported_count=sum(r['new_supported_count'] for r in rr),gained_support_zero_IoU=sum(len(r['gained_support_zero_IoU']) for r in rr),changed_supported_zero_IoU=sum(len(r['changed_supported_zero_IoU']) for r in rr),nonzero_target_transfer_queries=sum(r['nonzero_target_transfer_queries'] for r in rr))
            H.need(aggregate[arm]['transitions'] == r210['transitions'][arm], '210 global transitions differ')
        H.save(outdir/'06_summary.json',dict(arms=aggregate,GT=902,original_sub4=453,GO_slots=832,method_gate=None,AP_prediction=False,shared_parameter_gradient=False,baseline_exact=True,MAL_DN_invariant=True,repair_oldinside_preserved=True,repair_per_GT_support_nondecreasing=True))
        unchanged = {path:H.sha(path) == digest for path,digest in hashes.items()}; H.need(all(unchanged.values()), 'input/source changed')
        H.save(outdir/'07_execution.json',dict(state='complete',resources=resource,counts=H.COUNTS,source_unchanged=unchanged,criterion_state_sha256=hashlib.sha256(b'').hexdigest(),seconds=time.monotonic()-started,heldout_reads=False,optimizer_constructed=False))
        H.save(outdir/'08_output_sha256.json',{p.name:H.sha(p) for p in outdir.iterdir() if p.is_file()})
    except BaseException:
        H.save(outdir/'FAILED.json',dict(state='failed_no_method_conclusion',stage=stage,counts=H.COUNTS,traceback=traceback.format_exc(),seconds=time.monotonic()-started)); raise
if __name__ == '__main__': main()
