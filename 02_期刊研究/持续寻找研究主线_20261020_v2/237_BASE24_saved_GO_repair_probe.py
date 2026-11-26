"""236: preserve saved GO while auditing the exact frozen CPU tie differences.

Definitions reused from frozen223. No old main, no old validation bypass hidden
inside the math. This file explicitly validates every compatibility condition.
Written only; root reviews/runs. No checkpoint/model/gradient/heldout.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import copy
import gc
import hashlib
import importlib.util
import pathlib
import sys
import time
import traceback
from collections import Counter

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent
REUSE = ROOT/'223_BASE24_cached_GO_repair_probe.py'
REUSE_SHA = '86ba26409d7a596cd085b7364e6588b25121a016e103803a613befc3a481c533'
if hashlib.sha256(REUSE.read_bytes()).hexdigest() != REUSE_SHA:
    raise RuntimeError('223 changed')
_spec = importlib.util.spec_from_file_location('_audit237_reuse223',REUSE)
if _spec is None or _spec.loader is None:
    raise RuntimeError('223 import unavailable')
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)  # Definitions only. Never P.main or P.validate_batch.
H, M, A, B, J = P.H, P.M, P.A, P.B, P.J
np, torch, psutil = P.np, P.torch, P.psutil
require, sha, save = P.require, P.sha, P.save
PROTOCOL = ROOT/'236_保存GO与跨平台并列兼容_事前修订.txt'
EXPLANATION = ROOT/'233_修复前置复现审计.txt'
AUDIT = ROOT/'235_GO前置CPU并列复现_只读JSON审计.json'
AUDIT_SHA = '4f2b4f4ef60e3540ab259ac347675395cb293a4788f4711e1974c32b3463b8e9'
FAILED = ROOT/'224_BASE24_cached_GO_repair_probe/01_results.json'
FAILED_SHA = '065159ea235227d0950df26ed48cacb09645f546d99ef7c9e74eed8a37867eb7'
OLD_PROTOCOL_SHA = '0aab24f4538d468733761166cdbb659907fc4412b8560143435391f122ff00fb'


def manifest():
    files, expected = P.manifest()
    files.update(reuse223=files['script'],protocol222=files['protocol'],script=pathlib.Path(__file__),
                 protocol=PROTOCOL,explanation233=EXPLANATION,audit235=AUDIT,failure224=FAILED)
    expected.update(reuse223=REUSE_SHA,protocol222=OLD_PROTOCOL_SHA,audit235=AUDIT_SHA,failure224=FAILED_SHA)
    return files, expected


def load_union_audit():
    require(sha(AUDIT) == AUDIT_SHA, '235 fixed SHA')
    audit = P.read_json(AUDIT)
    require(audit['scope'] == 'saved development JSON branch matches only; no prediction tensors/checkpoints/models'
        and audit['torch'] == torch.__version__ == '2.5.1+cpu'
        and audit['criterion_source_sha256'] == A.EXPECTED['criterion']
        and audit['heldout_reads'] is False and audit['model_forwards'] == audit['tensor_cache_loads'] == audit['optimizer_steps'] == 0
        and audit['threads'] == 2 and audit['affinity'] == [13,15], '235 scope/runtime')
    wanted = {str((P.DATA/relative).resolve()):digest for relative,digest in P.INPUT_HASHES.items() if relative.endswith('_analysis.json')}
    observed = {str(pathlib.Path(path).resolve()):digest for path,digest in audit['sources_sha256'].items()}
    require(observed == wanted and len(wanted) == 16, '235 exact JSON sources')
    rows = {(row['account'],int(row['batch']),int(row['image'])):row for row in audit['rows']}
    require(len(audit['rows']) == len(rows) == 64 and set(rows) == {(a,b,i) for a in ('A','B') for b in range(8) for i in range(4)}, '235 complete identities')
    require(audit['total_differences'] == sum(len(x['differences']) for x in rows.values()) == 566
        and audit['affected_images'] == sum(bool(x['differences']) for x in rows.values()) == 50
        and audit['all_differences_top_ties'] is True, '235 frozen summary')
    for row in rows.values():
        require(row['diff_count'] == len(row['differences']) and row['all_difference_top_ties'] is True
            and all(d['both_top_tied'] is True for d in row['differences']), '235 row witnesses')
    return rows


def audit_saved_union(prior, index, rebuilt, expected):
    """Strict votes + original tie metadata + exact pre-recorded mismatch list."""
    saved = prior['inputs'][index]
    names = prior['branch_names']
    require(names == ['final','aux_0','aux_1','pre','enc_0'], 'original branch order')
    branch = prior['branch_matches'][index]
    require(set(branch) == set(names), 'branch keys')
    votes = Counter()
    for name in names:
        qs, gs = branch[name]
        require(len(qs) == len(gs) and len(set(qs)) == len(qs) and len(set(gs)) == len(gs)
            and all(type(q) is int and 0 <= q < 300 for q in qs)
            and all(type(g) is int and 0 <= g < saved['GT'] for g in gs), 'branch matching identity')
        votes.update(zip(qs,gs))
    vote_rows = [[int(q),int(g),int(n)] for (q,g),n in sorted(votes.items())]
    require(vote_rows == saved['votes'], 'original votes exactly reconstructed')
    byq = {}
    for (q,g),count in votes.items():
        byq.setdefault(q,{})[g] = count
    record = saved['branches']['final']['native_union']
    old = dict(zip(record['pair_query'],record['pair_GT']))
    fresh = dict(zip(rebuilt[0].tolist(),rebuilt[1].tolist()))
    require(len(old) == len(record['pair_query']) == len(fresh) == len(rebuilt[0])
        and set(old) == set(fresh) == set(byq), 'union query set')
    ties, differences = {}, []
    for q in sorted(byq):
        maximum = max(byq[q].values())
        best = sorted(g for g,c in byq[q].items() if c == maximum)
        require(old[q] in best and fresh[q] in best, 'saved/replayed owner is not a highest-vote GT')
        if len(best) > 1:
            ties[str(q)] = {'GT':best,'votes':maximum,'production_U0_GT':old[q]}
        if old[q] != fresh[q]:
            differences.append({'q':q,'saved_GT':old[q],'rebuilt_GT':fresh[q],
                'saved_votes':byq[q][old[q]],'rebuilt_votes':byq[q][fresh[q]],'max_votes':maximum,'both_top_tied':True})
    sensitivity = saved['tie_sensitivity']
    require(sensitivity['top_vote_ties'] == ties and sensitivity['top_vote_tied_queries'] == len(ties)
        and sensitivity['query_count'] == len(byq), 'original tie metadata exactly reconstructed')
    minimum = [[q,min(g for g,c in byq[q].items() if c == max(byq[q].values()))] for q in sorted(byq)]
    require(sensitivity['deterministic_min_GT_union'] == minimum
        and sensitivity['native_differs_min_GT_queries'] == sum(old[q] != g for q,g in minimum), 'original smallest-GT sensitivity')
    require(expected['slots'] == saved['K'] == len(old) and expected['saved_tie_count'] == len(ties)
        and expected['differences'] == differences and expected['diff_count'] == len(differences)
        and expected['all_difference_top_ties'] is True, 'CPU difference list differs from frozen235')
    return {'query_count':len(old),'top_vote_tied_queries':len(ties),'differences':differences,
        'exact_map_reproduction':not differences,'differences_match_frozen235':True,
        'votes_reconstructed_exact':True,'original_tie_metadata_exact':True,
        'saved_G0_used':True,'Windows_replayed_G0_used':False}


def compatibility_self_checks():
    names = ['final','aux_0','aux_1','pre','enc_0']
    branch = {name:[[0,1],[g,2]] for name,g in zip(names,[0,1,0,1,3])}
    record = {'GT':4,'K':2,'votes':[[0,0,2],[0,1,2],[0,3,1],[1,2,5]],
        'branches':{'final':{'native_union':{'pair_query':[0,1],'pair_GT':[1,2]}}},
        'tie_sensitivity':{'top_vote_ties':{'0':{'GT':[0,1],'votes':2,'production_U0_GT':1}},
            'top_vote_tied_queries':1,'query_count':2,'deterministic_min_GT_union':[[0,0],[1,2]],'native_differs_min_GT_queries':1}}
    prior = {'inputs':[record],'branch_names':names,'branch_matches':[branch]}
    fresh = (torch.tensor([0,1]),torch.tensor([0,2]))
    expected = {'slots':2,'saved_tie_count':1,'diff_count':1,'all_difference_top_ties':True,
        'differences':[{'q':0,'saved_GT':1,'rebuilt_GT':0,'saved_votes':2,'rebuilt_votes':2,'max_votes':2,'both_top_tied':True}]}
    result = audit_saved_union(prior,0,fresh,expected)
    require(not result['exact_map_reproduction'] and result['saved_G0_used'], 'synthetic original baseline preserved')
    cases = []
    wrong = copy.deepcopy(prior)
    wrong['inputs'][0]['branches']['final']['native_union']['pair_GT'][0] = 3
    cases.append((wrong,expected))
    wrong_expected = copy.deepcopy(expected)
    wrong_expected['differences'][0]['rebuilt_GT'] = 1
    cases.append((prior,wrong_expected))
    wrong = copy.deepcopy(prior)
    wrong['inputs'][0]['votes'][0][2] = 3
    cases.append((wrong,expected))
    for wrong, witness in cases:
        rejected = False
        try:
            audit_saved_union(wrong,0,fresh,witness)
        except RuntimeError:
            rejected = True
        require(rejected, 'compatibility accepted non-top owner/wrong frozen list/wrong votes')
    return {'saved_baseline_preserved':True,'reject_nonmax_votes':True,'reject_new_difference_list':True,'reject_changed_votes':True}


def validate_batch(value, prior, batch, seed, predrow, recipe, compute, union, union_audit):
    """223 validations retained, except the explicitly audited CPU tie equality."""
    require(value['group'] == prior['group'] == 'development' and value['tag'] == prior['tag'] == f'batch{batch:02d}'
            and prior['seed'] == seed and prior['prediction_sha256'] == predrow['sha256']
            and prior['source_ids'] == predrow['source_ids'], 'batch content identity')
    output, targets = value['outputs'], value['targets']
    require(len(targets) == len(prior['inputs']) == len(prior['branch_matches']) == 4, 'batch4 schema')
    require(output['pred_boxes'].shape == output['ref_points'].shape == (4,300,4)
            and output['pred_logits'].shape == (4,300,10), 'prediction schema')
    for tensor in (output['pred_boxes'],output['ref_points'],output['pred_logits']):
        require(tensor.device.type == 'cpu' and tensor.dtype == torch.float32 and not tensor.requires_grad
            and bool(torch.isfinite(tensor).all()), 'CPU frozen FP32')
    require(torch.equal(output['ref_points'],output['pre_outputs']['pred_boxes']), 'own pre/FDR refs')
    require(all(torch.equal(output['ref_points'],aux['ref_points']) for aux in output['aux_outputs']), 'shared FDR refs')
    names = ['final']+[f'aux_{i}' for i in range(len(output['aux_outputs']))]+['pre']+[f'enc_{i}' for i in range(len(output['enc_aux_outputs']))]
    require(prior['branch_names'] == names == ['final','aux_0','aux_1','pre','enc_0'], 'native GO branches')
    matches = [[tuple(torch.tensor(x,dtype=torch.int64) for x in row[name]) for row in prior['branch_matches']] for name in names]
    rebuilt = union(None,matches[0],matches[1:])
    costs, rematched = compute(output,targets)
    original, l1_terms, portability = [], [], []
    letter = {1:'A',2:'B'}[seed]
    for i,(target,saved,own,again,oldunion) in enumerate(zip(targets,prior['inputs'],matches[0],rematched,rebuilt)):
        gt, labels = target['boxes'],target['labels']
        require(gt.dtype == torch.float32 and gt.device.type == 'cpu' and gt.shape == (len(labels),4)
            and labels.dtype == torch.int64 and labels.device.type == 'cpu' and bool(torch.isfinite(gt).all())
            and bool((gt[:,2:] > 0).all()) and bool(((labels >= 0) & (labels < 10)).all()), 'GT schema')
        require(all(torch.equal(a,b) for a,b in zip(own,again)), 'same-cache own final Hungarian reproduction')
        saved_u = saved['branches']['final']['native_union']
        qs, gs = np.asarray(saved_u['pair_query'],dtype=np.int64),np.asarray(saved_u['pair_GT'],dtype=np.int64)
        require(qs.ndim == gs.ndim == 1 and qs.shape == gs.shape and len(set(qs.tolist())) == len(qs)
            and np.array_equal(qs,np.sort(qs)) and np.all((qs >= 0) & (qs < 300)) and np.all((gs >= 0) & (gs < len(gt))), 'saved G0 indices')
        require(saved['input_within_batch'] == i and saved['GT'] == saved_u['GT'] == len(gt)
            and saved['K'] == saved_u['pairs'] == len(qs)
            and saved_u['covered'] == saved['native_covered'] == len(set(gs.tolist()))
            and saved_u['GT_repetitions'] == np.bincount(gs,minlength=len(gt)).tolist(), 'saved G0 budgets')
        evidence = audit_saved_union(prior,i,oldunion,union_audit[(letter,batch,i)])
        portability.append({'account':letter,'batch':batch,'image':i,**evidence})
        # Deliberately append saved qs/gs, never oldunion/current Windows owner.
        original.append(np.stack((qs,gs),axis=1))
        l1_terms.append((output['pred_boxes'][i,qs]-gt[gs]).abs().reshape(-1))
    go_denominator = max(sum(len(p) for p in original),1)
    ordinary_denominator = max(sum(len(t['labels']) for t in targets),1)
    l1 = float(torch.cat(l1_terms).sum()/go_denominator*recipe['DEIMCriterion']['weight_dict']['loss_bbox'])
    expected = float(prior['batch_losses']['native_union']['losses']['loss_bbox'])
    require(abs(l1-expected) <= 1e-5+1e-5*abs(expected), 'native final L1/GO denominator replay')
    denominator_audit = {'GO':go_denominator,'ordinary_GT':ordinary_denominator,'all_arms_fixed':True,
        'GO_rule':'max(sum image K,1), single CPU process','ordinary_rule':'max(sum image GT,1)',
        'native_weighted_final_L1_replay':l1,'saved_native_weighted_final_L1':expected,
        'absolute_difference':abs(l1-expected),'fixed_FP32_reduction_atol_rtol':[1e-5,1e-5],
        'GO_portability':{'audit235_sha256':AUDIT_SHA,'saved_G0_used':True,'Windows_replayed_G0_used':False,'images':portability}}
    return output,targets,original,costs,denominator_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=pathlib.Path,default=ROOT/'238_BASE24_saved_GO_repair_probe')
    args = parser.parse_args()
    args.out = args.out.resolve()
    require(args.out.parent == ROOT and args.out.name.startswith('238_') and not args.out.exists(), 'new local238 output only')
    args.out.mkdir(exist_ok=False)
    files, expected = manifest()
    before, artifacts, batches, provenance = {}, {}, [], {}
    resources, checks, loaded = None, None, 0
    started, stage = time.monotonic(),'resources'
    result = {'state':'error'}
    try:
        resources = H.resource_limits()
        require(resources['priority'] == 'below_normal' and torch.version.cuda is None
            and M.nx.__version__ == '3.3' and A.scipy.__version__ == '1.13.1', 'CPU/package contract')
        stage = 'fixed_path_hash_manifest_before_content'
        before = {key:sha(path) for key,path in files.items()}
        require(all(before[key] == digest for key,digest in expected.items()), 'fixed source/input hashes')
        save(args.out/'00_input_manifest.json',{'paths':{k:str(p.resolve()) for k,p in files.items()},'sha256':before,
            'tensor_or_result_content_read':False,'heldout_file_reads':0})
        stage = 'synthetic_before_real_content'
        native = H.load_native_math()
        box = sys.modules[native.__package__+'.box_ops']
        checks = {'physical_support':B.self_checks(native),'integer_repair':J.self_checks(),'GO_compatibility':compatibility_self_checks()}
        stage = 'metadata_after_synthetic'
        failure = P.read_json(FAILED)
        require(failure['state'] == 'error' and failure['error'] == 'saved native union reproduction'
            and failure['completed_batches'] == [] and failure['sources_unchanged'] is True
            and failure['sha256_before']['script'] == REUSE_SHA, 'original224 failure retained')
        union_audit = load_union_audit()
        recipes, package = P.read_json(files['recipe']),P.read_json(files['package'])
        shapes = P.input_shapes()
        package_hashes = {x['relative']:x['sha256'] for x in package['files']}
        for key in ('matcher','criterion','box_ops','dfine_utils'):
            require(package_hashes['native/engine/deim/'+files[key].name] == before[key], 'original package/native math identity')
        for letter,seed in (('A',1),('B',2)):
            predrows, analysisrows, provenance[letter] = P.metadata(letter,seed)
            recipe = recipes[str(seed)]
            provenance[letter]['recipe'] = {k:recipe[k] for k in ('DEIMCriterion','DFINETransformer','use_focal_loss','eval_spatial_size')}
            provenance[letter]['GO_portability_audit235_sha256'] = AUDIT_SHA
            compute, union = P.native_readers(box,recipe)
            for batch in range(8):
                M.guard()
                require(predrows[batch]['input_sha256'] == shapes[batch]['sha256'], 'actual shape/input seal identity')
                stage = f'{letter}_development_batch{batch:02d}'
                predpath = P.DATA/letter/'frozen_predictions/development'/predrows[batch]['relative']
                analysispath = P.DATA/letter/'development'/analysisrows[batch]['relative']
                require(sha(predpath) == predrows[batch]['sha256'] and sha(analysispath) == analysisrows[batch]['sha256'], 'batch hashes before read')
                prior = P.read_json(analysispath)
                value = torch.load(predpath,map_location='cpu',weights_only=False)
                loaded += 1
                output,targets,original,costs,denom = validate_batch(value,prior,batch,seed,predrows[batch],recipe,compute,union,union_audit)
                images, query_rows, matrices = [], [], {}
                for index in range(4):
                    image,rows = P.analyze_image(index,output,targets[index],original[index],costs[index],recipe,native,box,prior['inputs'][index],matrices,shapes[batch]['shape'])
                    images.append(image)
                    query_rows.extend(rows)
                require(sum(x['slots'] for x in images) == sum(len(p) for p in original) == len(query_rows), 'batch slot count')
                summaries = {d:P.aggregate([x['domains'][d] for x in images]) for d in P.DOMAINS}
                compact = {'seed':seed,'letter':letter,'batch':batch,'input_archive_sha256':predrows[batch]['input_sha256'],
                    'prediction_sha256':predrows[batch]['sha256'],'source_ids':predrows[batch]['source_ids'],
                    'forward_DN_seed':int(value['forward_DN_seed']),'denominators':denom,'actual_input_shape':shapes[batch]['shape'],
                    'GT':sum(x['GT'] for x in images),'slots':sum(x['slots'] for x in images),'domains':summaries,
                    'per_image':[{k:x[k] for k in ('image_index','GT','slots','unique_GT','cohort','domains')} for x in images]}
                stem = f'{letter}_batch{batch:02d}'
                save(args.out/(stem+'_assignments.json'),{**compact,'images':images,'queries':query_rows})
                np.savez_compressed(args.out/(stem+'_matrices.npz'),**matrices)
                for suffix in ('_assignments.json','_matrices.npz'):
                    artifacts[stem+suffix] = sha(args.out/(stem+suffix))
                batches.append(compact)
                print(stage,'complete',flush=True)
                del value,prior,output,targets,original,costs,images,query_rows,matrices,image,rows
                gc.collect()
                M.guard()
        require(loaded == len(batches) == 16 and sum(len(x['per_image']) for x in batches) == 64, 'all fixed development exposures')
        portability = [x for batch in batches for x in batch['denominators']['GO_portability']['images']]
        require(sum(len(x['differences']) for x in portability) == 566 and sum(bool(x['differences']) for x in portability) == 50, 'all235 differences reproduced')
        seeds = {}
        for seed in (1,2):
            chosen = [x for x in batches if x['seed'] == seed]
            summaries = {d:P.aggregate([x['domains'][d] for x in chosen]) for d in P.DOMAINS}
            seeds[str(seed)] = {'batches':8,'input_exposures':32,'GT':sum(x['GT'] for x in chosen),
                'slots':sum(x['slots'] for x in chosen),'domains':summaries,'screens':{d:P.screen(summaries[d]) for d in P.DOMAINS}}
        flags = [seeds[str(seed)]['screens']['physical64']['necessary_noncopy'] for seed in (1,2)]
        result = {'state':'complete','primary':{'per_seed_necessary_noncopy':dict(zip(('1','2'),flags)),
            'status':'both' if all(flags) else ('mixed' if any(flags) else 'neither'),'training_authorized':False,
            'scope':'existing development cached premise only; 210 not inherited'},
            'seeds':seeds,'batches':batches,'provenance':provenance,'artifact_sha256':artifacts,
            'GO_portability':{'audit235_sha256':AUDIT_SHA,'differences':566,'affected_input_exposures':50,
                'saved_G0_used':True,'Windows_replayed_G0_used':False,'all_differences_explicitly_reported':True,'original224_failure_retained':True}}
        stage = 'complete'
    except Exception as exc:
        result = {'state':'error','error':str(exc),'traceback':traceback.format_exc(),
            'scope':'implementation/provenance/precision/certificate failure; not a method negative',
            'completed_batches':batches,'artifact_sha256':artifacts}
    finally:
        after = {key:sha(files[key]) for key in before if files[key].exists()}
        unchanged = bool(before and before == after and set(before) == set(files))
        if not unchanged:
            result['state'],result['hash_error'] = 'error','source/input missing or changed'
        result.update(stage=stage,sources_unchanged=unchanged,sha256_before=before,sha256_after=after,
            resources=resources,synthetic_self_checks=checks,loaded_prediction_batches=loaded,heldout_file_reads=0,
            checkpoint_loads=0,model_forwards=0,model_backwards=0,model_vjps=0,optimizer_steps=0,new_training=False,
            flow_statistics=M.FLOW_STATS,RSS_observed_peak_bytes=M.RSS_PEAK,
            OS_peak_working_set_bytes=getattr(psutil.Process().memory_info(),'peak_wset',None),
            free_RAM_GiB_at_end=psutil.virtual_memory().available/2**30,seconds=time.monotonic()-started)
        save(args.out/'01_results.json',result)
        print(H.json.dumps(H.jsonable({'state':result['state'],'primary':result.get('primary'),'seconds':result['seconds']}),ensure_ascii=False),flush=True)
    if result['state'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
