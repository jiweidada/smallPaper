"""Compare summaries of existing evaluation arrays with complete historical log rows.

This performs no model inference, evaluator execution, or optimizer update. The
only computation is arithmetic on already-saved precision and recall tensors.
"""
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
from inspect_saved_arrays import InertArray, read_saved

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
OUTPUT = HERE/'saved_array_log_matching.json'
TOLERANCE = 1e-12  # Fixed before comparing complete 12-value vectors.
METRICS = ['AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl']

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

def as_array(x):
    return x.to_numpy() if isinstance(x,InertArray) else np.asarray(x)

def summarize(precision, recall):
    assert precision.shape==(10,101,10,4,3) and recall.shape==(10,10,4,3)
    assert np.isfinite(precision).all() and np.isfinite(recall).all()
    def avg(x):
        valid = x[x>-1]
        return float(np.mean(valid)) if valid.size else -1.0
    return np.array([avg(precision[:,:,:,0,2]), avg(precision[0,:,:,0,2]),
                     avg(precision[5,:,:,0,2]), avg(precision[:,:,:,1,2]),
                     avg(precision[:,:,:,2,2]), avg(precision[:,:,:,3,2]),
                     avg(recall[:,:,0,0]), avg(recall[:,:,0,1]), avg(recall[:,:,0,2]),
                     avg(recall[:,:,1,2]), avg(recall[:,:,2,2]), avg(recall[:,:,3,2])])

def check_params(obj):
    p=obj['params']
    assert np.allclose(as_array(p.iouThrs),np.linspace(.50,.95,10),rtol=0,atol=1e-14)
    assert np.allclose(as_array(p.recThrs),np.linspace(0,1,101),rtol=0,atol=1e-14)
    assert list(p.maxDets)==[1,10,100] and list(p.areaRngLbl)==['all','small','medium','large']
    assert list(p.catIds)==list(range(10)) and len(p.imgIds)==548 and p.useCats==1

def compare_arrays(a,b):
    assert a.shape==b.shape
    diff=np.abs(a-b)
    return {'elements':int(a.size), 'exact_equal_elements':int(np.count_nonzero(a==b)),
            'different_elements':int(np.count_nonzero(a!=b)),
            'max_absolute_difference':float(diff.max()), 'mean_absolute_difference':float(diff.mean())}

def main():
    assert not OUTPUT.exists()
    manifest=json.loads((REC/'source_manifest.json').read_text(encoding='utf-8-sig'))
    runs={}
    for run in ['N0','N1','N2','N3','N4','S1']:
        matched=[x for x in manifest if Path(x['snapshot']).parent.name==run and Path(x['snapshot']).name=='log.txt']
        assert len(matched)==1
        snapshot=Path(matched[0]['snapshot'])
        original=Path(matched[0]['source'])
        assert sha(snapshot)==sha(original)==matched[0]['sha256']
        logs=[json.loads(line) for line in snapshot.read_text(encoding='utf-8').splitlines() if line.strip()]
        assert all(len(x['test_coco_eval_bbox'])==12 for x in logs)
        runs[run]={'log':snapshot,'log_sha256':sha(snapshot),'output_dir':original.parent,'rows':logs}
    cpu={}
    for run in ['N0','N4','S1']:
        path=PAPER/f'本地CPU核验_20261103/results/{run}/evaluation_arrays.npz'
        stats_path=PAPER/f'本地CPU核验_20261103/results/{run}/metrics.json'
        with np.load(path,allow_pickle=False) as f:
            arrays={k:f[k].copy() for k in ['precision','recall','scores']}
        stats=json.loads(stats_path.read_text(encoding='utf-8'))['metrics']
        values=np.array([stats[k] for k in METRICS])
        computed=summarize(arrays['precision'],arrays['recall'])
        assert np.max(np.abs(computed-values))<=TOLERANCE
        cpu[run]={'path':path,'sha256':sha(path),'metrics_path':stats_path,'metrics_sha256':sha(stats_path),
                  'arrays':arrays,'values':values,'stored_array_summary_max_error':float(np.max(np.abs(computed-values)))}
    items=[]
    for run,info in runs.items():
        eval_dir=info['output_dir']/'eval'
        files=sorted(eval_dir.glob('*.pth')) if eval_dir.is_dir() else []
        for path in files:
            try:
                stored=read_saved(path)
                check_params(stored)
                arrays={k:stored[k].to_numpy() for k in ['precision','recall','scores']}
                values=summarize(arrays['precision'],arrays['recall'])
                distances=[float(np.max(np.abs(values-np.asarray(row['test_coco_eval_bbox'],dtype=float)))) for row in info['rows']]
                same=[{'epoch':row['epoch'],'line':i+1,'max_abs_difference':distances[i]}
                      for i,row in enumerate(info['rows']) if distances[i]<=TOLERANCE]
                row={'run':run,'path':str(path),'sha256':sha(path),'stored_date':stored.get('date'),
                     'summary':dict(zip(METRICS,values.tolist())),'complete_log_vector_matches':same,
                     'nearest_log_max_abs_difference':min(distances),
                     'expected_by_filename':int(path.stem) if path.stem.isdigit() else None}
                if run in cpu:
                    row['cpu_comparison']={k:compare_arrays(arrays[k],cpu[run]['arrays'][k]) for k in arrays}
                items.append(row)
            except Exception as exc:
                items.append({'run':run,'path':str(path),'sha256':sha(path),'error':type(exc).__name__+': '+str(exc)})
    s1_dir=runs['S1']['output_dir'].parent/'deim_hgnetv2_s_MSCA_EnhancedStem_SiLU_320_eval'
    path=s1_dir/'eval.pth'
    if path.exists():
        try:
            stored=read_saved(path)
            check_params(stored)
            arrays={k:stored[k].to_numpy() for k in ['precision','recall','scores']}
            values=summarize(arrays['precision'],arrays['recall'])
            info=runs['S1']
            diffs=[float(np.max(np.abs(values-np.asarray(row['test_coco_eval_bbox'],dtype=float)))) for row in info['rows']]
            items.append({'run':'S1_standalone','path':str(path),'sha256':sha(path),'stored_date':stored.get('date'),
                          'summary':dict(zip(METRICS,values.tolist())),
                          'complete_log_vector_matches':[{'epoch':row['epoch'],'line':i+1,'max_abs_difference':diffs[i]}
                                  for i,row in enumerate(info['rows']) if diffs[i]<=TOLERANCE],
                          'nearest_log_max_abs_difference':min(diffs),
                          'cpu_comparison':{k:compare_arrays(arrays[k],cpu['S1']['arrays'][k]) for k in arrays}})
        except Exception as exc:
            items.append({'run':'S1_standalone','path':str(path),'sha256':sha(path),'error':type(exc).__name__+': '+str(exc)})
    result={'status':'EXISTING_EVALUATION_ARRAYS_SUMMARIZED_AND_MATCHED',
            'matching_tolerance_full_12_vector':TOLERANCE,
            'source_logs':{run:{'path':str(info['log']),'sha256':info['log_sha256'],'rows':len(info['rows'])} for run,info in runs.items()},
            'cpu':{run:{'path':str(v['path']),'sha256':v['sha256'],'metrics_path':str(v['metrics_path']),
                        'metrics_sha256':v['metrics_sha256'],'saved_array_summary_max_error':v['stored_array_summary_max_error']}
                   for run,v in cpu.items()},
            'saved_evaluations':items,
            'limits':['Arithmetic on saved precision and recall, not a model or evaluator run.',
                      'The complete 12-value match is a consistency check; it does not cryptographically identify a checkpoint, source version or historical dataset.',
                      'File naming and current solver behavior do not certify the historical solver.',
                      'No new AP estimate, data filtering, training, or weight evaluation was performed.'],
            'models_run':0,'forwards':0,'evaluator_runs':0,'optimizer_updates':0}
    with OUTPUT.open('x',encoding='utf-8',newline='\n') as out:
        json.dump(result,out,ensure_ascii=False,indent=2)
        out.write('\n')
    print(json.dumps({'files':len(items),'errors':[x for x in items if 'error' in x],
                      'exact_vector_matches':sum(bool(x.get('complete_log_vector_matches')) for x in items),
                      'cpu_validation_max_error':max(v['stored_array_summary_max_error'] for v in cpu.values())},ensure_ascii=False))
    for row in items:
        if row['run']=='S1_standalone' or row['path'].endswith('latest.pth'):
            print(json.dumps({'run':row['run'],'file':Path(row['path']).name,'matches':row.get('complete_log_vector_matches'),
                              'nearest':row.get('nearest_log_max_abs_difference'),
                              'cpu_max_abs':{k:v['max_absolute_difference'] for k,v in row.get('cpu_comparison',{}).items()},
                              'error':row.get('error')},ensure_ascii=False))

if __name__=='__main__':
    main()
