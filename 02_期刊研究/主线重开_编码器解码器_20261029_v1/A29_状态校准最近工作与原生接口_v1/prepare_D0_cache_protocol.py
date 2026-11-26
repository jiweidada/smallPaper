"""Freeze a descriptive reanalysis of existing source-image predictions, no inference."""
from pathlib import Path
import datetime,hashlib,json,sys
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parent
OLD=R.parent.parent/'前期调研_码率受限航拍小目标_20261028_v1'/'远端验证_20261029_v1'/'完成结果_20261029_v1'
paths={'predictions':OLD/'raw/evaluation/D0/source_predictions_input.json','metrics':OLD/'raw/evaluation/D0/source_metrics.json','GT':OLD/'raw/dataset/diagnostic_gt_orig.json','metadata':OLD/'raw/receipts/diagnostic_observations.json','checkpoint':OLD/'raw/results/D0/weights/last.pt','protocol':OLD/'raw/experiment_protocol.json','archive':OLD/'completed_baseline_results.tar.gz'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def main():
    dest=R/'D0_cache_audit_protocol_FROZEN.json';assert not dest.exists()
    expected=json.loads((OLD/'local_backup_verification.json').read_text(encoding='utf-8'))
    files={k:{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for k,p in paths.items()}
    assert files['checkpoint']['sha256']==expected['checkpoints']['D0']['sha256']
    assert files['archive']['sha256']==expected['archive_sha256']
    p={'phase':'A29','kind':'Exploratory descriptive reanalysis, not a new independent test or preregistered hypothesis confirmation','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'source':'D0 fixed100 EMA; unmodified dataset source JPEGs, no artificial recompression; diagnostic100 already used for prior development',
       'host':{'model':'YOLO26s','ultralytics':'8.4.191','source_commit':'2004cba021c68a46429be460e26954ab171b2862','branch':'unfused one2many + multi-label NMS','max_det':500,'imgsz':640},
       'files':files,'missed_set':'Noncrowd GT with input area<1024 absent from stored matched_small_uids at confidence0.25/IoU0.5; reuse historical evaluator matching, do not relabel availability as TP.',
       'coordinates':'Scale original xywh by metadata ratio and add left/top padding exactly as archived ground_truth_input; not effective rounded resize scale.',
       'exclusive_priority_bins':[
        {'id':'A_high_score_same_class_overlap_unmatched','condition':'some retained same-class box has IoU>=0.5 and score>=0.25'},
        {'id':'B_same_class_overlap_score_005_to_025','condition':'otherwise same-class box IoU>=0.5 and max covering score>=0.05'},
        {'id':'C_same_class_overlap_score_below_005','condition':'otherwise any retained same-class box IoU>=0.5'},
        {'id':'D_other_class_overlap_score_ge_005','condition':'otherwise any-class box IoU>=0.5 and score>=0.05'},
        {'id':'E_same_class_nearby_box_score_ge_005','condition':'otherwise same-class box IoU>=0.1 and score>=0.05'},
        {'id':'F_other_class_overlap_low_score','condition':'otherwise any retained box IoU>=0.5'},
        {'id':'G_no_above_retained_candidate','condition':'all remaining missed GT'}],
       'reading_rules':['These bins describe post-NMS saved candidates; absence does not locate the failure in an encoder.','A box may cover multiple GT; availability is not a recoverable recall bound.','Low confidence alone does not prove classification calibration or a novel algorithm opportunity.','No threshold tuning, AP re-estimation, statistical significance, held-out claim or new algorithm is authorized by this protocol.','Record size split strict<256 vs256<=area<1024; exclude all crowd annotations.'],
       'allocations':{'new_image_forwards':0,'new_native_scan_calls':0,'new_decoder_calls':0,'optimizer_updates':0,'GPU_calls':0,'cached_images':100}}
    dest.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'FROZEN_EXISTING_SOURCE_CACHE_DESCRIPTIVE_AUDIT','input_files':len(files),'D0_checkpoint_and_archive_reverified':True,'model_calls_allocated':0}))
if __name__=='__main__':main()
