"""Check the author's overall DET protocol against its released FRCNN precision/recall curve."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from scipy.io import loadmat

HERE=Path(__file__).resolve().parent;DATA=HERE.parents[1]/'data/07_第二数据集_20261017'
PACKAGE=DATA/'原始下载/UAVDT/UAV-benchmark-MOTD_v1.0.zip'
SEQUENCES=['M0203','M0205','M0208','M0209','M0403','M0601','M0602','M0606','M0701','M0801','M0802','M1001','M1004','M1007','M1009','M1101','M1301','M1302','M1303','M1401']
LENGTHS=[1007,646,265,1576,514,372,480,1374,1308,298,1101,1859,269,659,604,864,1182,719,445,1050]
IGNORE_PASS=set(SEQUENCES)-{'M0209','M0801'}

def voc_ap(recall,precision):
    r=np.r_[0.,recall,1.];p=np.r_[0.,precision,0.]
    p=np.maximum.accumulate(p[::-1])[::-1];ids=np.flatnonzero(r[1:]!=r[:-1])+1
    return float(np.sum((r[ids]-r[ids-1])*p[ids]))

def ignore_contained(detections,regions):
    if len(regions)==0 or len(detections)==0:return detections
    d=detections;r=regions
    inside=(d[:,None,0]>r[None,:,0])&(d[:,None,1]>r[None,:,1])&(
        d[:,None,0]+d[:,None,2]<r[None,:,0]+r[None,:,2])&(
        d[:,None,1]+d[:,None,3]<r[None,:,1]+r[None,:,3])
    return d[~inside.any(axis=1)]

def match_frame(gt,dt):
    dt=dt[np.argsort(-dt[:,4],kind='stable')];matched=np.zeros(len(gt),dtype=bool)
    flags=np.zeros(len(dt),dtype=np.int8)
    if len(gt)==0 or len(dt)==0:return dt[:,4],flags
    ends_d=dt[:,:2]+dt[:,2:4];ends_g=gt[:,:2]+gt[:,2:4]
    wh=np.maximum(0.,np.minimum(ends_d[:,None,:],ends_g[None,:,:])-np.maximum(dt[:,None,:2],gt[None,:,:2]))
    intersection=wh[:,:,0]*wh[:,:,1]
    overlap=intersection/(dt[:,2,None]*dt[:,3,None]+gt[None,:,2]*gt[None,:,3]-intersection)
    for index,row in enumerate(overlap):
        candidates=np.flatnonzero((~matched)&(row>=.7))
        if len(candidates):
            maximum=row[candidates].max();chosen=candidates[row[candidates]==maximum][-1]
            matched[chosen]=True;flags[index]=1
    return dt[:,4],flags

def group_frames(rows):
    if not len(rows):return {}
    rows=rows[np.argsort(rows[:,0],kind='stable')]
    return {int(block[0,0]):block for block in np.split(rows,np.flatnonzero(np.diff(rows[:,0]))+1)}

def main():
    # Cases which would expose COCO/IoD substitution, inclusive pixels, duplicate TPs, or .5 overlap.
    gt=np.array([[0.,0.,10.,10.]])
    scores,flags=match_frame(gt,np.array([[0.,0.,7.,10.,.9],[0.,0.,10.,10.,.8]]))
    assert flags.tolist()==[1,0]
    d=np.array([[1.,1.,8.,8.,.9],[0.,0.,10.,10.,.8],[0.,0.,7.,10.,.7]])
    assert ignore_contained(d,gt).shape[0]==2
    assert voc_ap(np.array([.5,1.]),np.array([1.,.5]))==.75
    collected_scores=[];collected_flags=[];gt_count=0;source_hashes={};per_sequence=[]
    with zipfile.ZipFile(PACKAGE) as archive:
        def read_array(name):
            raw=archive.read('UAV-benchmark-MOTD_v1.0/'+name)
            return np.loadtxt(io.BytesIO(raw),delimiter=',',ndmin=2) if raw.strip() else np.empty((0,9))
        for name in ['utils/CalculateDetectionPR_overall.m','utils/evalRes.m','compOas.m','utils/VOCap.m']:
            source_hashes[name]=hashlib.sha256(archive.read('UAV-benchmark-MOTD_v1.0/'+name)).hexdigest()
        for sequence,length in zip(SEQUENCES,LENGTHS):
            gt=group_frames(read_array('GT/'+sequence+'_gt_whole.txt'))
            dt=group_frames(read_array('RES_DET/det_FRCNN/'+sequence+'.txt'))
            ign=group_frames(read_array('GT/'+sequence+'_gt_ignore.txt'))
            kept=0;removed=0;positives=0
            for frame in range(1,length+1):
                g=gt.get(frame,np.empty((0,9)))[:,2:6]
                d=dt.get(frame,np.empty((0,9)))[:,2:7]
                original=len(d)
                if sequence in IGNORE_PASS:d=ignore_contained(d,ign.get(frame,np.empty((0,9)))[:,2:6])
                score,flag=match_frame(g,d);collected_scores.append(score);collected_flags.append(flag)
                gt_count+=len(g);positives+=len(g);kept+=len(d);removed+=original-len(d)
            per_sequence.append({'sequence':sequence,'frames':length,'normal_gt':positives,'detections_kept':kept,'detections_ignored':removed})
            print(json.dumps({'event':'reference_sequence_scored','sequence':sequence,'detections':kept}),flush=True)
        scores=np.concatenate(collected_scores);flags=np.concatenate(collected_flags)
        flags=flags[np.argsort(-scores,kind='stable')]
        tp=np.cumsum(flags==1);fp=np.cumsum(flags==0);recall=tp/max(1,gt_count);precision=tp/np.maximum(1,tp+fp)
        reference=loadmat(io.BytesIO(archive.read('UAV-benchmark-MOTD_v1.0/det_EVA/det_FRCNN_overall.mat')),simplify_cells=True)['attribute']
        rr=np.asarray(reference['rec']).reshape(-1);rp=np.asarray(reference['prec']).reshape(-1)
        same_length=len(rr)==len(recall)
        max_rec=float(np.max(np.abs(rr-recall))) if same_length else None
        max_prec=float(np.max(np.abs(rp-precision))) if same_length else None
        passed=bool(same_length and max_rec<1e-12 and max_prec<1e-12)
        record={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'author_source_sha256':source_hashes,
                'native_reference':'det_EVA/det_FRCNN_overall.mat','frames':sum(LENGTHS),'ground_truth':gt_count,
                'detections_scored':len(scores),'reference_detections_scored':len(rr),'AP70_raw':voc_ap(recall,precision),
                'reference_AP70_raw':voc_ap(rr,rp),'precision_curve_max_absolute_difference':max_prec,
                'recall_curve_max_absolute_difference':max_rec,'reference_curve_agreement_passed':passed,
                'protocol':{'overlap_threshold':.7,'ignore_rule':'Remove detections strictly fully contained in same-frame ignored rectangles; exact author pass list.',
                    'matching':'Greedy descending score, one GT once, last GT on equal best overlap, continuous xywh IoU >= .7.',
                    'overall_class_agnostic':True,'category_field_not_used_for_overall_match':True,'max_detections':'No native per-image cap',
                    'ap_integral':'VOC continuous precision envelope, not COCO AP50 or AP50:95.'},
                'synthetic_protocol_checks_passed':True,'per_sequence':per_sequence,'GPU_used':False,'training_started':False}
    target=DATA/'协议核查_v1/UAVDT官方DET参考曲线核验_v1.json'
    with target.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in record.items() if k not in ['per_sequence','author_source_sha256']},ensure_ascii=False),flush=True)
    assert passed,'Official reference curve mismatch, do not claim evaluator validated'

if __name__=='__main__':main()
