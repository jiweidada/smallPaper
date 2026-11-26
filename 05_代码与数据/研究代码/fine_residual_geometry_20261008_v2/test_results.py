"""Engineering tests on existing archives only; no model, matcher or evaluator execution."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import results

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'fine_residual_geometry_20261008_v1'
EVIDENCE = HERE.parents[1]/'data/00_实验计划与证据索引'
ARCHIVE = EVIDENCE/'中段常见错误多图原生核查_20261008_v1'
PER_GT = EVIDENCE/'正式132轮基线整图双问题诊断_20261008_v1/native_transform_v2'


def main():
    output = HERE/'engineering_results'
    output.mkdir(exist_ok=False)
    checks, sources = [], {}
    def record(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    def rejected(name, callback, error=ValueError):
        try:
            callback()
        except error:
            checks.append(name)
        else:
            raise AssertionError(name+' was accepted')
    def remember(path):
        sources[str(path)] = results.sha(path)
        return path
    bound = results.read(OLD/'03_交付回执与SHA.json')['files_sha256']
    record('v1 bound files unchanged before testing', all(results.sha(OLD/k)==v for k,v in bound.items()))
    receipt = results.read(OLD/'prepared_local/baseline_seed1/receipt.json')
    annotation_path = remember(Path(receipt['paths']['val_ann']))
    ann = results.read(annotation_path)
    summaries, flags, endpoints = [], [], []
    for seed in (1,2):
        cached = results.read(remember(ARCHIVE/f'seed{seed}_COCO逐类缓存.json'))
        records = results.read(remember(ARCHIVE/f'seed{seed}_COCO_allarea_image_matching.json'))
        with np.load(remember(ARCHIVE/f'seed{seed}_COCO_precision_recall.npz'),allow_pickle=False) as values:
            tensors = {k:values[k] for k in values.files}
        params = SimpleNamespace(iouType='bbox',maxDets=[1,10,100],
            areaRng=[[0,1e10],[0,1024],[1024,9216],[9216,1e10]],
            areaRngLbl=['all','small','medium','large'],catIds=tensors['catIds'],
            imgIds=[x['id'] for x in ann['images']],useCats=1,
            iouThrs=tensors['iouThrs'],recThrs=tensors['recThrs'])
        evaluator = SimpleNamespace(params=params,eval=tensors,evalImgs=records,
            stats=np.asarray(cached['aggregate_stats_percent'])/100)
        identity = deepcopy(receipt['identity'])
        identity['seed'] = seed
        # This is a baseline archive, not a v2 training result or candidate output.
        destination = output/f'archived_baseline_seed{seed}'
        kwargs = dict(evaluator=evaluator,test_stats={'coco_eval_bbox':evaluator.stats},
            identity=identity,annotation_path=annotation_path,output=destination,
            epoch=131,mode='archived_fixture')
        summary = results.capture(**kwargs)
        loaded, flag, _ = results.load_endpoint(destination)
        record(f'seed{seed} all six aggregate AP metrics agree with archive',
            np.allclose(list(summary['overall_percent'].values()),cached['aggregate_stats_percent'][:6],rtol=0,atol=1e-9))
        record(f'seed{seed} all 60 category metrics agree with archive',
            all(abs(c[k]-cached['categories'][i][k])<1e-9 for i,c in enumerate(summary['categories_percent']) for k in results.METRICS))
        with np.load(remember(PER_GT/f'seed{seed}/analysis/per_gt.npz'),allow_pickle=False) as old:
            order = np.argsort(old['annotation_id'])
            record(f'seed{seed} all 38759 archived TP50 identities agree',
                np.array_equal(flag['annotation_id'],old['annotation_id'][order]) and
                np.array_equal(flag['hits'][:,0],old['official_match_row'][order]>=0))
        record(f'seed{seed} annotation zero retains native unmatched sentinel behavior',
            not flag['hits'][np.where(flag['annotation_id']==0)[0][0]].any())
        identical = results.pair(loaded,loaded,flag,flag)
        record(f'seed{seed} identity comparison has no gains, losses or metric changes',
            not any(identical['GT_new_TP_by_IoU']+identical['GT_lost_TP_by_IoU']+list(identical['delta_AP_metrics_pp'].values())))
        rejected(f'seed{seed} export cannot overwrite prior results',lambda:results.capture(**kwargs),FileExistsError)
        bad = {**kwargs,'output':output/'must_not_exist'}
        rejected(f'seed{seed} wrong final epoch rejected',lambda:results.capture(**{**bad,'epoch':130}))
        wrong = deepcopy(params); wrong.maxDets=[1,10,300]
        rejected(f'seed{seed} wrong maxDets rejected',lambda:results.capture(**{**bad,'evaluator':SimpleNamespace(**{**vars(evaluator),'params':wrong})}))
        rejected(f'seed{seed} missing GT coverage rejected',lambda:results.capture(**{**bad,'evaluator':SimpleNamespace(**{**vars(evaluator),'evalImgs':[]})}))
        rejected(f'seed{seed} duplicate matching cell rejected',lambda:results.capture(**{**bad,'evaluator':SimpleNamespace(**{**vars(evaluator),'evalImgs':records+[records[0]]})}))
        invalid = dict(tensors); invalid['precision']=tensors['precision'].copy(); invalid['precision'].flat[0]=np.nan
        rejected(f'seed{seed} nonfinite AP tensor rejected',lambda:results.capture(**{**bad,'evaluator':SimpleNamespace(**{**vars(evaluator),'eval':invalid})}))
        record(f'seed{seed} invalid captures created no output',not bad['output'].exists())
        summaries.append(loaded); flags.append(flag); endpoints.append(destination)
    delta = results.pair(*summaries,*flags)
    record('cross-seed archive arithmetic obeys gain-minus-loss conservation at all ten IoUs',
        np.array_equal(np.asarray(delta['GT_new_TP_by_IoU'])-delta['GT_lost_TP_by_IoU'],flags[1]['hits'].sum(0)-flags[0]['hits'].sum(0)))
    record('per-class gains/losses sum to overall gains/losses',
        np.array_equal(np.sum([x['new_TP_by_IoU'] for x in delta['class_GT_changes']],axis=0),delta['GT_new_TP_by_IoU']) and
        np.array_equal(np.sum([x['lost_TP_by_IoU'] for x in delta['class_GT_changes']],axis=0),delta['GT_lost_TP_by_IoU']))
    rejected('old baseline fixtures cannot become candidate experimental evidence',lambda:results.compare(endpoints*3))
    swapped=deepcopy(summaries[1]); swapped['categories_percent'].reverse()
    rejected('misordered class comparison rejected',lambda:results.pair(summaries[0],swapped,*flags))

    # Extract only the export hook; StubBase has no training, model, torch or CUDA.
    tree=ast.parse((HERE/'experiment.py').read_text(encoding='utf-8'))
    cls=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='ExperimentSolver')
    hook=deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='save_best_model'))
    cls.body=[hook]
    class StubBase:
        def save_best_model(self,*args):
            self.delegated=getattr(self,'delegated',0)+1
            return 'native_return'
    context={'DetSolver':StubBase,'receipt':receipt,'paths':{'val_ann':annotation_path},
        'torch':SimpleNamespace(__version__='stub',version=SimpleNamespace(cuda=None),
            cuda=SimpleNamespace(get_device_name=lambda _: 'stub'))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'isolated_export_hook','exec'),context)
    instance=context['ExperimentSolver']()
    instance.ema=object(); instance.device='stub'; instance.output_dir=output
    instance.evaluator=SimpleNamespace(coco_eval={'bbox':object()})
    with patch.dict(sys.modules,{'torchvision':SimpleNamespace(__version__='stub')}):
        with patch.object(results,'capture',return_value={}) as captured:
            record('nonfinal hook delegates without capture',instance.save_best_model({}, {},130)=='native_return' and not captured.called)
            record('final hook delegates and captures once',instance.save_best_model({}, {},131)=='native_return' and captured.call_count==1 and instance.endpoint_capture_complete)
        del instance.endpoint_capture_complete
        with patch.object(results,'capture',side_effect=RuntimeError('deliberate export failure')):
            record('export failure returns to native fit for checkpoint saving',instance.save_best_model({}, {},131)=='native_return' and 'deliberate export failure' in instance.endpoint_capture_error and not getattr(instance,'endpoint_capture_complete',False))
    record('v1 bound files unchanged after testing',all(results.sha(OLD/k)==v for k,v in bound.items()))
    record('all archive inputs unchanged',all(results.sha(k)==v for k,v in sources.items()))
    record('no torch or native evaluator imported',not any(k in sys.modules for k in ('torch','pycocoeval','pycocotools')))
    report=dict(status='PASS_ARCHIVED_DATA_ENGINEERING_ONLY',checks=checks,check_count=len(checks),
        inputs_sha256=sources,original_v1_bound_files=len(bound),
        test_code_sha256=results.sha(__file__),capture_code_sha256=results.sha(HERE/'results.py'),
        model_forwards=0,matcher_calls=0,evaluator_calls=0,optimizer_constructions=0,
        training_started=False,new_candidate_results=False,
        native_training_hook_integration_verified=False,
        artifact_note='Archived baseline fixtures cannot serve as v2 or fine_residual experimental results.')
    results.write(output/'report.json',report)
    print(json.dumps({'status':report['status'],'checks':len(checks)},ensure_ascii=False))


if __name__=='__main__':
    main()
