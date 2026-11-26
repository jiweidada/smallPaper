"""Independently check log matching coverage and selected array summaries."""
from hashlib import sha256
from pathlib import Path
import json
import statistics
import numpy as np
from inspect_saved_arrays import read_saved

HERE=Path(__file__).resolve().parent
INPUT=HERE/'saved_array_log_matching.json'
PRED=HERE/'s1_saved_prediction_rank_comparison.json'
OUTPUT=HERE/'independent_saved_array_verification.json'
TOLERANCE=1e-12

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

def mean_nonnegative(a):
    values=(float(v) for v in a.flat if v > -1)
    return statistics.fmean(values)

def independent_summary(p,r):
    return [mean_nonnegative(x) for x in [p[:,:,:,0,2],p[0,:,:,0,2],p[5,:,:,0,2],
        p[:,:,:,1,2],p[:,:,:,2,2],p[:,:,:,3,2],r[:,:,0,0],r[:,:,0,1],r[:,:,0,2],
        r[:,:,1,2],r[:,:,2,2],r[:,:,3,2]]]

def main():
    assert not OUTPUT.exists()
    audit=json.loads(INPUT.read_text(encoding='utf-8'))
    predictions=json.loads(PRED.read_text(encoding='utf-8'))
    assert audit['status']=='EXISTING_EVALUATION_ARRAYS_SUMMARIZED_AND_MATCHED'
    assert audit['matching_tolerance_full_12_vector']==TOLERANCE
    saved=audit['saved_evaluations']
    assert len(saved)==49 and not any('error' in row for row in saved)
    runs=['N0','N1','N2','N3','N4','S1']
    expected={'000.pth','050.pth','100.pth','150.pth','200.pth','250.pth','300.pth','latest.pth'}
    maxima=[]
    best_epochs={}
    selected_evaluations=[]
    for run in runs:
        records=[x for x in saved if x['run']==run]
        assert len(records)==8 and {Path(x['path']).name for x in records}==expected
        source=audit['source_logs'][run]
        assert sha(Path(source['path']))==source['sha256'] and source['rows']==320
        logs=[json.loads(line) for line in Path(source['path']).read_text(encoding='utf-8').splitlines() if line.strip()]
        best=max(logs,key=lambda x:x['test_coco_eval_bbox'][0])
        best_epochs[run]=best['epoch']
        for row in records:
            assert sha(Path(row['path']))==row['sha256']
            matches=row['complete_log_vector_matches']
            assert len(matches)==1
            expected_epoch=319 if Path(row['path']).name=='latest.pth' else int(Path(row['path']).stem)
            assert matches[0]['epoch']==expected_epoch
            maxima.append(matches[0]['max_abs_difference'])
            if matches[0]['epoch']==best['epoch']:
                selected_evaluations.append({'run':run,'epoch':best['epoch'],'path':row['path'],'sha256':row['sha256']})
    assert best_epochs=={'N0':315,'N1':306,'N2':300,'N3':301,'N4':306,'S1':302}
    assert [(x['run'],x['epoch']) for x in selected_evaluations]==[('N2',300)]
    n2=read_saved(Path(selected_evaluations[0]['path']))
    calc=independent_summary(n2['precision'].to_numpy(),n2['recall'].to_numpy())
    n2_logs=[json.loads(line) for line in Path(audit['source_logs']['N2']['path']).read_text(encoding='utf-8').splitlines() if line.strip()]
    target=next(x for x in n2_logs if x['epoch']==300)['test_coco_eval_bbox']
    err=max(abs(a-b) for a,b in zip(calc,target))
    assert err<TOLERANCE
    standalone=next(x for x in saved if x['run']=='S1_standalone')
    assert sha(Path(standalone['path']))==standalone['sha256'] and not standalone['complete_log_vector_matches']
    assert any(v['different_elements']>0 for v in standalone['cpu_comparison'].values())
    assert predictions['status']=='EXISTING_PREDICTIONS_COMPARED_BY_SAVED_ORDER'
    assert predictions['historical']['records']==predictions['cpu']['records']==164400
    assert predictions['same_image_count_map'] and predictions['rank_aligned']['different_category_count']>0
    report={'status':'ALL_48_CORRESPONDING_ROWS_VERIFIED',
            'source_audit_sha256':sha(INPUT),'prediction_audit_sha256':sha(PRED),
            'runs':runs,'arrays_checked':48,'numbered_files_checked':42,'latest_files_checked':6,
            'each_full_vector_unique_match':True,'maximum_abs_difference':max(maxima),
            'best_ap_epochs':best_epochs,'selected_best_log_arrays':selected_evaluations,
            'n2_independent_fmean_max_error':err,
            'standalone_s1_complete_log_vector_matches':len(standalone['complete_log_vector_matches']),
            'standalone_s1_cpu_arrays_not_identical':True,
            'scope':'Consistency of stored evaluation summaries and logged values; no model, evaluator, or optimizer was executed.',
            'limits':['Matched summaries do not record or prove weight hashes or historical source/data identity.',
                      'The separately saved S1 prediction and eval files are not cryptographically bound to each other or a checkpoint.']}
    with OUTPUT.open('x',encoding='utf-8',newline='\n') as out:
        json.dump(report,out,ensure_ascii=False,indent=2)
        out.write('\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
