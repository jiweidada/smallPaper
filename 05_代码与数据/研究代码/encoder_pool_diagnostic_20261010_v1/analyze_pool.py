"""All-GT geometric supply, unique matching and native TopK omission.

The lexicographic matcher is certified against independent SciPy cardinalities
on the remote copies. The local run independently checks archives and recomputes
every adjacency/count from full FP32 arrays without needing local SciPy.
"""
from pathlib import Path
import argparse,csv,hashlib,json,sys,time
from collections import defaultdict
import numpy as np
sys.setrecursionlimit(10000)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def xyxy(box):return np.concatenate((box[:,:2]-box[:,2:]/np.float32(2),box[:,:2]+box[:,2:]/np.float32(2)),1)
def iou_matrix(pred,gt):
    a,b=xyxy(pred),xyxy(gt);awh=a[:,2:]-a[:,:2];bwh=b[:,2:]-b[:,:2];ar=awh[:,0]*awh[:,1];br=bwh[:,0]*bwh[:,1];inter_wh=np.maximum(np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]),np.float32(0));inter=inter_wh[:,:,0]*inter_wh[:,:,1];return inter/(ar[:,None]+br[None,:]-inter)
def match_all(edge,tiny,scipy_check):
    # Candidate x GT graph. Add tiny GT first, then every other GT. Each
    # successful augmenting path preserves all previously matched GT vertices.
    n,m=edge.shape;adj=[np.flatnonzero(edge[:,g]).tolist() for g in range(m)];candidate_gt=np.full(n,-1,dtype=np.int64);gt_candidate=np.full(m,-1,dtype=np.int64)
    def dfs(g,visited):
        for c in adj[g]:
            if visited[c]:continue
            visited[c]=True;other=int(candidate_gt[c])
            if other<0 or dfs(other,visited):candidate_gt[c]=g;gt_candidate[g]=c;return True
        return False
    for g in np.concatenate((np.flatnonzero(tiny),np.flatnonzero(~tiny))):dfs(int(g),np.zeros(n,dtype=bool))
    covered=gt_candidate>=0;assigned=gt_candidate[covered];assert len(set(assigned))==len(assigned) and edge[assigned,np.flatnonzero(covered)].all()
    certificate=None
    if scipy_check:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import maximum_bipartite_matching
        sparse=csr_matrix(edge.T);maximum=int((maximum_bipartite_matching(sparse,perm_type='column')>=0).sum());max_tiny=int((maximum_bipartite_matching(sparse[tiny],perm_type='column')>=0).sum()) if tiny.any() else 0
        assert int(covered.sum())==maximum and int((covered&tiny).sum())==max_tiny
        certificate={'all_GT_cardinality':maximum,'tiny_cardinality':max_tiny,'edge_sha256':hashlib.sha256(np.ascontiguousarray(edge).tobytes()).hexdigest()}
    return covered,gt_candidate,certificate
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--seed',type=int);p.add_argument('--scipy-check',action='store_true');a=p.parse_args();root=Path(a.root);protocol=read(root/'protocol.json');start=time.monotonic();target=root/('analysis_scipy_reference' if a.scipy_check else 'local_analysis');target.mkdir(exist_ok=False)
    seeds=[a.seed] if a.seed else [1,2];rows=[];images=[];certificates=[];batch_checks=[]
    for seed in seeds:
      capture=root/'capture_v1' if a.seed else root/f'collection_seed{seed}/capture_v1';manifest=read(capture/'manifest.json')['files']
      for name,h in manifest.items():assert sha(capture/name)==h,name
      complete=read(capture/'completion.json');assert complete['status']=='COMPLETE_ENCODER_POOL_CAPTURE' and complete['recorded_batches']==40 and complete['model_forward_calls']==46 and complete['diagnostic_dense_head_calls']==40 and complete['optimizer_steps']==complete['epochs_added']==0
      ident=read(capture/'input_identity.json');source=read(capture/'source_identity.json');assert ident['augmented_reconstruction_all16_batches_exact'] and ident['val_train_eval_input_identity'];seen_val={}
      for endpoint in (12,132):
       for context,batches in [('aug_train',[0,5,10,15]),('val_train',range(8)),('val_eval',range(8))]:
        for batch in batches:
          sample=capture/f'endpoint{endpoint}/{context}/batch_{batch:03d}';evidence=read(sample/'evidence.json')
          with np.load(sample/'arrays.npz',allow_pickle=False) as npz:data={k:npz[k] for k in npz.files}
          assert len(data)==evidence['array_count'];assert all(np.isfinite(x).all() for x in data.values())
          for name,v in data.items():assert hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()==evidence['arrays'][name]['sha256']
          logits=data['pool.logits'];idx=data['pool.selected_indices'];selected=data['encoder.selected_boxes'];raw=data['pool.raw_boxes'];valid=data['pool.valid'];assert np.array_equal(data['encoder.selected_logits'],np.take_along_axis(logits,idx[:,:,None],axis=1));assert valid.shape==raw.shape[:2] and valid.dtype==bool
          if context.startswith('val'):
            signature=(evidence['image_tensor_sha256'],[(data[f'target.{j}.boxes'].tobytes(),data[f'target.{j}.labels'].tobytes(),data[f'target.{j}.source_annotation_ids'].tobytes()) for j in range(4)])
            key=batch
            if key not in seen_val:seen_val[key]=signature
            else:assert seen_val[key]==signature
          for j in range(4):
            gt=data[f'target.{j}.boxes'];labels=data[f'target.{j}.labels'];source_ids=data[f'target.{j}.source_annotation_ids'];n=len(gt);size=evidence['input_shape'][-1];short=gt[:,2:].astype(np.float64).min(1)*size;tiny=short<4;dense=n>300
            assert len(set(idx[j]))==300 and np.array_equal(np.sort(idx[j]),np.unique(idx[j]));assert valid[j,idx[j]].all()
            full=raw[j].copy();error=float(np.abs(full[idx[j]]-selected[j]).max());assert error<=evidence['raw_dense_head_vs_native_selected_max_abs_error']<=2e-6
            full[idx[j]]=selected[j];valid_idx=np.flatnonzero(valid[j]);full=full[valid_idx];pool_iou=iou_matrix(full,gt);enc_iou=iou_matrix(selected[j],gt);final_iou=iou_matrix(data['final.boxes'][j],gt)
            assert np.array_equal(full[np.searchsorted(valid_idx,idx[j])],selected[j]);assert np.array_equal(pool_iou[np.searchsorted(valid_idx,idx[j])],enc_iou)
            pool_cls=logits[j,valid_idx].argmax(-1);enc_cls=data['encoder.selected_logits'][j].argmax(-1);final_cls=data['final.logits'][j].argmax(-1);pool_score=logits[j].max(-1);cutoff=float(pool_score[idx[j]].min())
            matrices={'full':pool_iou,'selected_encoder':enc_iou,'final':final_iou};classes={'full':pool_cls,'selected_encoder':enc_cls,'final':final_cls};graphresults={};im={'seed':seed,'endpoint':endpoint,'context':context,'batch':batch,'item':j,'image_id':int(data[f'target.{j}.image_id'].reshape(-1)[0]),'GT_rows':n,'tiny_rows':int(tiny.sum()),'dense_gt300':dense,'input_size':size,'full_valid_candidates':len(valid_idx),'selected_candidates':300,'native_TopK_min_maxclass_logit':cutoff}
            for threshold in [.5,.75]:
              for semantic in [False,True]:
               for pool,iou in matrices.items():
                edge=iou>=threshold
                if semantic:edge=edge&(classes[pool][:,None]==labels[None,:])
                covered,witness,certificate=match_all(edge,tiny,a.scipy_check);key=pool+f'_{int(threshold*100)}'+('_class' if semantic else '')
                graphresults[key]={'any':edge.any(0),'unique':covered,'witness':witness};im[key+'_all_unique']=int(covered.sum());im[key+'_tiny_unique']=int((covered&tiny).sum())
                if certificate:certificates.append({'seed':seed,'endpoint':endpoint,'context':context,'batch':batch,'item':j,'graph':key,**certificate})
            assert np.all(pool_iou.max(0)+1e-7>=enc_iou.max(0)) if n else True
            # Full graph is an exact superset of selected geometry/class graph;
            # both global and tiny optima must be monotone.
            for threshold in [50,75]:
              for suffix in ['', '_class']:
                for group in ['all','tiny']:assert im[f'full_{threshold}{suffix}_{group}_unique']>=im[f'selected_encoder_{threshold}{suffix}_{group}_unique']
            images.append(im)
            for g in range(n):
              r={'seed':seed,'endpoint':endpoint,'context':context,'batch':batch,'item':j,'image_id':im['image_id'],'GT_row':g,'source_annotation_id':int(source_ids[g]),'category':int(labels[g]),'GT_count_in_composite':n,'dense_gt300':dense,'input_short_pixels':float(short[g]),'input_size_bin':'lt4' if short[g]<4 else '4to8' if short[g]<8 else '8to16' if short[g]<16 else 'ge16','max_full_IoU':float(pool_iou[:,g].max()),'max_selected_encoder_IoU':float(enc_iou[:,g].max()),'max_final_IoU':float(final_iou[:,g].max()),'native_TopK_cutoff_logit':cutoff}
              for key,v in graphresults.items():r[key+'_any']=bool(v['any'][g]);r[key+'_unique']=bool(v['unique'][g])
              for threshold in [50,75]:
                candidates=np.flatnonzero((pool_iou[:,g]>=threshold/100)&(pool_cls==labels[g]));name=f'full_{threshold}_class_best'
                if len(candidates):
                  best_local=int(candidates[np.argmax(pool_score[valid_idx[candidates]])]);candidate=int(valid_idx[best_local]);score=float(pool_score[candidate]);r[name+'_native_index']=candidate;r[name+'_maxclass_logit']=score;r[name+'_strict_rank']=int((pool_score>score).sum())+1;r[name+'_tie_rank_upper']=int((pool_score>=score).sum());r[name+'_was_selected']=bool(candidate in idx[j])
                else:r[name+'_native_index']=-1;r[name+'_maxclass_logit']=None;r[name+'_strict_rank']=None;r[name+'_tie_rank_upper']=None;r[name+'_was_selected']=False
              rows.append(r)
          batch_checks.append({'seed':seed,'endpoint':endpoint,'context':context,'batch':batch,'all_arrays_sha256_exact':True,'all_selected_logits_exact':True,'raw_dense_vs_selected_max_abs_error':evidence['raw_dense_head_vs_native_selected_max_abs_error']})
    for name,values in [('per_GT_pool_coverage.csv',rows),('per_image_unique_coverage.csv',images)]:
      with (target/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    def summarize(rs):
        count=len(rs);result={'GT_rows':count}
        for key in graphresults:
          for k in ['any','unique']:result[key+'_'+k]=sum(r[key+'_'+k] for r in rs)
        if count:
          result['full_vs_selected_encoder_unique50_gain_pp']=100*(result['full_50_unique']-result['selected_encoder_50_unique'])/count
          result['full_vs_selected_encoder_unique75_gain_pp']=100*(result['full_75_unique']-result['selected_encoder_75_unique'])/count
        else:result['full_vs_selected_encoder_unique50_gain_pp']=None;result['full_vs_selected_encoder_unique75_gain_pp']=None
        return result
    groups=[];primary=[]
    for seed in seeds:
      for endpoint in (12,132):
       for context in ['aug_train','val_train','val_eval']:
        current=[r for r in rows if r['seed']==seed and r['endpoint']==endpoint and r['context']==context]
        for size in ['lt4','4to8','8to16','ge16']:
         for dense in [False,True]:
          subset=[r for r in current if r['input_size_bin']==size and r['dense_gt300']==dense];entry={'seed':seed,'endpoint':endpoint,'context':context,'input_size_bin':size,'dense_gt300':dense,**summarize(subset)};groups.append(entry)
          if endpoint==132 and context=='val_eval' and size=='lt4' and not dense:primary.append(entry)
        for category in sorted({r['category'] for r in current}):groups.append({'seed':seed,'endpoint':endpoint,'context':context,'dimension':'category','value':category,**summarize([r for r in current if r['category']==category])})
    eligible=all(r['full_vs_selected_encoder_unique50_gain_pp'] is not None and r['full_vs_selected_encoder_unique50_gain_pp']>=10 for r in primary)
    summary={'status':'PASS_FULL_POOL_TOPK_GEOMETRY_AND_UNIQUE_MATCHING','seeds':seeds,'primary':primary,'strata':groups,'GT_exposures':len(rows),'image_exposures':len(images),'recorded_batches':len(batch_checks),'scipy_independent_optimality_checked':a.scipy_check,'certificates':len(certificates),'diagnostic_priority_gate_both_seed_ge10pp':eligible if len(seeds)==2 else None,'single_seed_gate_ge10pp':eligible,'geometry_oracle_is_AP':False,'root_cause_established':False,'epochs_added':0,'optimizer_steps':0,'elapsed_seconds':time.monotonic()-start,'analysis_code_sha256':sha(Path(__file__)),'protocol_sha256':sha(root/'protocol.json')}
    save(target/'summary.json',summary);save(target/'array_verification.json',{'status':'PASS_ALL_FP32_ARRAYS_SHA_NATIVE_SELECTION_AND_POOL_SUPERSET','batch_checks':batch_checks});save(target/'matching_certificates.json',{'status':'PASS_INDEPENDENT_SCIPY_DUAL_OPTIMUM_CERTIFICATES' if a.scipy_check else 'LOCAL_VALID_WITNESSES_PENDING_REMOTE_SCIPY_CERTIFICATE_COMPARISON','certificates':certificates})
    print(json.dumps({k:v for k,v in summary.items() if k!='strata'},ensure_ascii=False))
if __name__=='__main__':main()
