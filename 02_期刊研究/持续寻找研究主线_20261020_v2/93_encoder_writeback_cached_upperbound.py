"""Post-hoc necessary exposure bound for cross-source *geometry* writeback. No model."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import pathlib,sys,json,importlib.util
ROOT=pathlib.Path(__file__).resolve().parent;WORK=ROOT.parents[1]
sys.path.insert(0,str(WORK/'英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1'))
import common as n
torch,np=n.torch,n.np

def main():
 resource=n.resource_limits();torch.set_grad_enabled(False)
 spec=importlib.util.spec_from_file_location('source84u',ROOT/'84_source_self_attention.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 selected=json.loads((ROOT/'85_source_self_attention/00_selected.json').read_text(encoding='utf8'))
 source=torch.load(ROOT/'85_source_self_attention/00_source_maps.pt',map_location='cpu',weights_only=False)['source_maps']
 inputs=torch.load(ROOT/'77_all_count_frozen/00_all_count_inputs.pt',map_location='cpu',weights_only=False)
 cases=[next(c for c in inputs['cases'] if c['event']==s['event']) for s in selected]
 table=json.loads((ROOT/'91_source_quota_frozen/00_selections.json').read_text(encoding='utf8'))
 attention=json.loads((ROOT/'85_source_self_attention/02_attention.json').read_text(encoding='utf8'))
 tags=json.loads((ROOT/'49_augmentation_replay_audit_v4/05_tagged_source_identity.json').read_text(encoding='utf8'));ids={(r['seed'],r['position']):r for r in tags}
 recipes=json.loads((WORK/'英文小论文/data/08_匹配聚合覆盖探针_20261019_v1/baseline_recipes.json').read_text(encoding='utf-8-sig'))
 anchors=[]
 for level,(h,w) in enumerate([(80,80),(40,40),(20,20)]):
  yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij');xy=(torch.stack([xx,yy],-1).unsqueeze(0)+.5)/torch.tensor([w,h],dtype=torch.float32)
  wh=torch.ones_like(xy)*.05*(2.**level);anchors.append(torch.cat([xy,wh],-1).reshape(-1,h*w,4))
 anchors=torch.cat(anchors,1);valid=((anchors>.01)&(anchors<.99)).all(-1,keepdim=True);logit=torch.log(anchors/(1-anchors));logit=torch.where(valid,logit,torch.inf)
 centers=logit.sigmoid()[...,:2].expand(4,-1,-1);all_source=helper.lookup(source,centers)
 rows=[];totals={}
 for seed in [1,2]:
  pred=torch.load(ROOT/f'91_source_quota_frozen/seed{seed}_predictions.pt',map_location='cpu',weights_only=False)['original']
  criterion=n.build_criterion(recipes[str(seed)]);matches=criterion.matcher(pred,[c['target'] for c in cases])['indices']
  expected=next(r['source_counts'] for r in attention if r['model_seed']==seed and r.get('layer')==0)
  tt={}
  for b,c in enumerate(cases):
   saved=next(r for r in table if r['model_seed']==seed and r['arm']=='original' and r['event']==c['event'])
   idx=torch.tensor(saved['selected_indices']);sid=all_source[b,idx]
   assert [int((sid==k).sum()) for k in range(5)]==expected[b], 'reconstructed native anchors source counts not exact'
   target=c['target'];gs=torch.tensor([g['source_slot']+1 for g in ids[c['event']['seed'],c['event']['position']]['instances']])
   iou=n.box_iou(n.box_cxcywh_to_xyxy(pred['pred_boxes'][b]),n.box_cxcywh_to_xyxy(target['boxes']))[0]
   foreign=(sid[:,None]>0)&(sid[:,None]!=gs[None,:]);geom=foreign&(iou>=.5)
   q,g=matches[b];chosen=foreign[q,g]&(iou[q,g]>=.5)
   _,top=torch.topk(pred['pred_logits'][b].sigmoid().flatten(),300);query,cl=top//10,top%10
   correct=geom[query]&(cl[:,None]==target['labels'][None,:])
   r={'model_seed':seed,'event':c['event'],'GT':len(gs),'queries':len(sid),'foreign_origin_geo05_edges':int(geom.sum()),
      'GT_with_foreign_origin_geo05':int(geom.any(0).sum()),'native_final_matched_foreign_origin_geo05':int(chosen.sum()),
      'GT_with_foreign_origin_class05_top300':int(correct.any(0).sum()),'source_counts_reconstruction_exact':True,
      'foreign_final_match_query_GT':torch.stack([q[chosen],g[chosen]],-1).tolist()}
   rows.append(r)
   for k in ['GT','queries','foreign_origin_geo05_edges','GT_with_foreign_origin_geo05','native_final_matched_foreign_origin_geo05','GT_with_foreign_origin_class05_top300']:tt[k]=tt.get(k,0)+r[k]
  totals[str(seed)]=tt
 n.save_json(ROOT/'94_encoder_writeback_cached_upperbound.json',{'state':'complete','analysis_type':'posthoc_necessary_upperbound_not_preregistered_gate','resources':resource,
  'totals':totals,'rows':rows,'scope':'No union/encoder outputs are available here. Final foreign geometry edges upper-bound the final-correct cross-source writeback cases, not all union conflicts or all datasets. Class budget kept separate.',
  'model_constructed':False,'model_forwards':0,'model_backwards':0,'optimizer_steps':0,'new_training':False,'heldout_reads':False,'script_sha256':n.sha(__file__),
  'sources':{f:n.sha(ROOT/f) for f in ['85_source_self_attention/00_source_maps.pt','91_source_quota_frozen/00_selections.json','91_source_quota_frozen/seed1_predictions.pt','91_source_quota_frozen/seed2_predictions.pt']}})
 print(json.dumps({'state':'complete','totals':totals,'rows':rows},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
