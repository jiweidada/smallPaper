"""Post-hoc cached geometry audit;no network,readout fitting or new predictions."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import numpy as np
from verify_A8_results import maxflow_count

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'A13_既有轨迹与坐标支持范围审计_v1'
SOURCE=ROOT/'A12_原生边界分布诊断_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def xy(b):
    return np.concatenate((b[...,:2]-b[...,2:]*.5,b[...,:2]+b[...,2:]*.5),axis=-1)*800.


def iou(a,b):
    extent=np.maximum(np.minimum(a[...,2:],b[...,2:])-np.maximum(a[...,:2],b[...,:2]),0.)
    intersection=extent.prod(-1)
    aa=np.maximum(a[...,2:]-a[...,:2],0.).prod(-1)
    ab=np.maximum(b[...,2:]-b[...,:2],0.).prod(-1)
    union=aa+ab-intersection
    return np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)


def maximum_count(matrix):
    edges=[np.flatnonzero(row).tolist() for row in matrix]
    owner=np.full(matrix.shape[1],-1,np.int64)
    def visit(g,seen):
        for q in edges[g]:
            if q in seen:continue
            seen.add(q)
            if owner[q]<0 or visit(int(owner[q]),seen):
                owner[q]=g;return True
        return False
    for g in range(len(edges)):visit(g,set())
    count=int((owner>=0).sum())
    assert maxflow_count(edges)==count
    return count


def ci(rows,key):
    kept=[r for r in rows if r[key] is not None]
    groups=sorted({r['prefix'] for r in kept})
    values=[[r[key] for r in kept if r['prefix']==g] for g in groups]
    rng=np.random.Generator(np.random.PCG64(2026103013))
    boot=[np.mean([x for i in rng.integers(len(groups),size=len(groups)) for x in values[i]]) for _ in range(2000)]
    return {'mean':float(np.mean([r[key] for r in kept])),'low':float(np.quantile(boot,.025)),'high':float(np.quantile(boot,.975)),'images':len(kept),'prefixes':len(groups)}


def main():
    assert not OUT.exists(),'Do not silently rerun or overwrite a post-hoc screen.'
    OUT.mkdir()
    verified=json.loads((SOURCE/'independent_verification.json').read_text(encoding='utf-8'))
    assert verified['status']=='PASS_RAW_ARCHIVE_SOURCE_MOMENTS_AND_ALL_PRIMARY_PAIR_FLAGS'
    completed=json.loads((SOURCE/'completed.json').read_text(encoding='utf-8'))
    assert completed['cumulative_full_model_forwards']==213
    source_files=[]
    for index in range(12):
        path=SOURCE/'predictions'/('case%02d.npz'%index)
        record=next(r for r in completed['files'] if r['path']=='predictions/'+path.name)
        assert sha(path)==record['sha256'];source_files.append({'path':str(path),'sha256':sha(path)})
    p={'utc':datetime.now(timezone.utc).isoformat(),'status':'FROZEN_BEFORE_A13_TRAJECTORY_AND_PROJECTED_SUPPORT_COUNTS','design':'Exploratory post-hoc use of12 A12 official-val cases. A12phase-correction failure and29/1052 out-of-support edges already known;not independent confirmation.','question':'Can the fixed regression-coordinate support express accurate tiny geometry without changing A12query assignments,and when does geometric degradation occur across reconstructed stages?','primary':'For all263tinyGT,keepfrozenA12GT-to-query assignment. ProjectGTedges into the allowed marginal edge intervals and measureIoU>=.75 coverage. Missingassignments count asuncovered.','construction':'Actual initialref cxcywh,reg_scale4 and savedbin min/max. Fouredge intervals follow source distance2bbox. Clip eachGTedge to its interval;this is a constructive closed-support free-logit oracle,not a trainedprediction or a general optimum claim.','gate':'Atleast120tinyGT and8prefixes. Image-mean projected assigned coverage>=.95 andbootstrap lower>.90 -> deprioritize support-range insufficiency alone. Lower coverage is inconclusive,not proof that the support ceiling is low. No new module approved in either case.','stage_control':'Fixed finalA12matching;pre_bboxes plus3source-mean-reconstructed FDR boxes. Previousdecoderstage pass/final fail is geometryonly,notSQRTP-fading,APoravailable earlier-stage classprediction. Generic phenomenon already covered bySQR/TPRD.','secondary':'All-raw-query class-agnostic maxmatching on tinyGT for actualboxes,FP64reconstructednativeboxes,and per-GT projected boxes atIoU.5/.75;onephysicalqueryperGTcapacity. Other sizebins havefixedassignmentgeometrystatistics only.','source_numeric_scope':'Previouslayer positions are reconstructed from capturedsource means;onlyfinalroundtrip was directly checkedinA12. Theseare referenceboxes,not actualMSDAsamplinglocationsorfeatures.','numerics':'FP64 projection andprobability reconstruction. Closurevalidity tolerance1e-8 inputpixels andnative-FP64 oracle-domination tolerance1e-8IoU. RawFP32prediction remains actual-native comparator;FP64sensitivity reported separately.','source_files':source_files,'source_code_sha256':sha(Path(__file__)),'source_matching_helper_sha256':sha(ROOT/'verify_A8_results.py'),'model_forwards_unchanged':213,'new_fits':0,'main_model_updates':0,'not_claimed':['Featureinformation already exists','Actualimage sampling covers a target','A causal encoder/decoder fault','New algorithm or training gain','Representability under shared learned head weights','APorblindtestgeneralization']}
    write(OUT/'protocol.json',p)
    write(OUT/'started.json',{'utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(OUT/'protocol.json')})
    assets=json.loads((SOURCE/'asset_selection.json').read_text(encoding='utf-8'))
    image_rows=[];gt_rows=[];global_rows=[];graph_counts=0
    for index,case in enumerate(assets['cases']):
        with np.load(SOURCE/'predictions'/('case%02d.npz'%index),allow_pickle=False) as z:
            gt=z['gt_boxes800'].astype(float);sizes=z['gt_sizes800'].astype(float)
            ref=z['ref_initial'].astype(float);raw=z['raw_boxes'].astype(float)
            decoded=z['decoded_boxes'].astype(float)
            mq=z['match_q'];mg=z['match_gt'];grid=z['bin_grid'].astype(float)
            logits=z['corner_logits'][-1].astype(float)
        q=len(ref);r=4.;small=np.tile(ref[:,2:]/r,(1,2))
        center=np.tile(ref[:,:2],(1,2))
        low=np.stack((ref[:,0]-(r*.5+grid[-1])*ref[:,2]/r,ref[:,1]-(r*.5+grid[-1])*ref[:,3]/r,ref[:,0]+(r*.5+grid[0])*ref[:,2]/r,ref[:,1]+(r*.5+grid[0])*ref[:,3]/r),axis=-1)*800
        high=np.stack((ref[:,0]-(r*.5+grid[0])*ref[:,2]/r,ref[:,1]-(r*.5+grid[0])*ref[:,3]/r,ref[:,0]+(r*.5+grid[-1])*ref[:,2]/r,ref[:,1]+(r*.5+grid[-1])*ref[:,3]/r),axis=-1)*800
        projected=np.minimum(np.maximum(gt[:,None,:],low[None,:,:]),high[None,:,:])
        assert np.min(projected[...,2:]-projected[...,:2])>=-1e-8
        oracle_iou=iou(gt[:,None,:],projected)
        actual_iou=iou(gt[:,None,:],xy(raw)[None,:,:])
        probs=np.exp(logits-logits.max(-1,keepdims=True));probs/=probs.sum(-1,keepdims=True)
        distances=(probs*grid).sum(-1)
        signs=np.asarray([-1.,-1.,1.,1.])
        native64=(center+signs*(r*.5+distances)*small)*800.
        native64_iou=iou(gt[:,None,:],native64[None,:,:])
        assert np.max(native64_iou-oracle_iou)<=1e-8
        allstages=np.concatenate((ref[None],decoded),axis=0)
        stage_iou=np.stack([iou(gt[mg],xy(b[mq])) for b in allstages],axis=1)
        # Final actual source result is the stage3 reference;CPU replay sensitivity is separate.
        native_assigned=iou(gt[mg],xy(raw[mq]))
        final_class_changes=int(np.sum((stage_iou[:,-1]>=.75)!=(native_assigned>=.75)))
        stage_iou[:,-1]=native_assigned
        projected_assigned=oracle_iou[mg,mq]
        row={'case':index,'file':case['file'],'prefix':case['prefix'],'matched_GT':len(mg),'final_FP32_replay_threshold_disagreements':final_class_changes}
        masks={'all':np.ones(len(mg),bool),'lt8':sizes[mg]<8,'8to16':(sizes[mg]>=8)&(sizes[mg]<16),'ge16':sizes[mg]>=16}
        for cohort,mask in masks.items():
            n=int(mask.sum());row[cohort+'_n']=n
            for threshold in (.5,.75):
                key=cohort+'_iou'+str(threshold)
                row[key+'_stage_counts']=[int((stage_iou[mask,l]>=threshold).sum()) for l in range(4)]
                row[key+'_projected_assigned']=int((projected_assigned[mask]>=threshold).sum())
            row[cohort+'_projected75_fraction']=float(np.mean(projected_assigned[mask]>=.75)) if n else None
            bad=(stage_iou[:,-1]<.75)&mask
            earlier=(stage_iou[:,1:3].max(-1)>=.75)&bad
            row[cohort+'_final75_fail']=int(bad.sum())
            row[cohort+'_earlier_decoder75_pass_final_fail']=int(earlier.sum())
            row[cohort+'_pre75_pass_final_fail']=int(((stage_iou[:,0]>=.75)&bad).sum())
        image_rows.append(row)
        for qi,gi,ii,oo in zip(mq,mg,stage_iou,projected_assigned):
            gt_rows.append({'case':index,'gt_index':int(gi),'query':int(qi),'size800':float(sizes[gi]),'stage_ious':ii.tolist(),'projected_assigned_iou':float(oo),'prefix':case['prefix']})
        tiny=sizes<8
        for threshold in (.5,.75):
            counts={}
            for key,matrix in [('actual_native_geometry',actual_iou),('source_FP64_native_geometry',native64_iou),('projected_free_logits_geometry',oracle_iou)]:
                counts[key]=maximum_count(matrix[tiny]>=threshold);graph_counts+=1
            global_rows.append({'case':index,'cohort':'lt8','threshold':threshold,'GT':int(tiny.sum()),**counts})
    primary=ci(image_rows,'lt8_projected75_fraction')
    tiny_n=sum(r['lt8_n'] for r in image_rows)
    decision='DEPRIORITIZE_FIXED_SUPPORT_RANGE_ALONE_AS_MAINLINE' if tiny_n>=120 and primary['prefixes']>=8 and primary['mean']>=.95 and primary['low']>.90 else 'SUPPORT_SCREEN_INCONCLUSIVE_NO_MODULE_APPROVED'
    aggregates=[]
    for cohort in ('all','lt8','8to16','ge16'):
        aggregates.append({'cohort':cohort,'GT':sum(r[cohort+'_n'] for r in image_rows),'stage_iou50_counts':[sum(r[cohort+'_iou0.5_stage_counts'][l] for r in image_rows) for l in range(4)],'stage_iou75_counts':[sum(r[cohort+'_iou0.75_stage_counts'][l] for r in image_rows) for l in range(4)],'projected_assigned50':sum(r[cohort+'_iou0.5_projected_assigned'] for r in image_rows),'projected_assigned75':sum(r[cohort+'_iou0.75_projected_assigned'] for r in image_rows),'final75_fail':sum(r[cohort+'_final75_fail'] for r in image_rows),'earlier_decoder75_pass_final_fail':sum(r[cohort+'_earlier_decoder75_pass_final_fail'] for r in image_rows),'pre75_pass_final_fail':sum(r[cohort+'_pre75_pass_final_fail'] for r in image_rows)})
    global_aggregate=[{'cohort':'lt8','threshold':t,**{k:sum(r[k] for r in global_rows if r['threshold']==t) for k in ('GT','actual_native_geometry','source_FP64_native_geometry','projected_free_logits_geometry')}} for t in (.5,.75)]
    result={'utc':datetime.now(timezone.utc).isoformat(),'status':'COMPLETED_CACHED_SUPPORT_AND_TRAJECTORY_AUDIT','decision':decision,'images':12,'primary_projected_assigned75':primary,'aggregate':aggregates,'global_tiny_capacity':global_aggregate,'independent_graph_count_checks':graph_counts,'FP32_replay_75_disagreements':sum(r['final_FP32_replay_threshold_disagreements'] for r in image_rows),'new_full_model_forwards':0,'cumulative_full_model_forwards':213,'new_fits':0,'main_model_updates':0,'mainline_validated':False,'scope':p['not_claimed']}
    write(OUT/'per_image.json',image_rows);write(OUT/'per_GT.json',gt_rows);write(OUT/'global_capacity.json',global_rows);write(OUT/'summary.json',result)
    write(OUT/'completed.json',{'status':'COMPLETED','summary_sha256':sha(OUT/'summary.json'),'protocol_sha256':sha(OUT/'protocol.json'),'source_code_sha256':sha(Path(__file__)),'new_full_model_forwards':0,'new_fits':0})
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
