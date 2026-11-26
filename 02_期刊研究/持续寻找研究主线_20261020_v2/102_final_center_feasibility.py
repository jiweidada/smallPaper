"""Cached, query-capacity-correct geometric bounds. No model or optimizer."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import pathlib,sys,json,time,importlib.util
ROOT=pathlib.Path(__file__).resolve().parent;WORK=ROOT.parents[1]
sys.path.insert(0,str(WORK/'英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1'))
import common as n
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
torch,np=n.torch,n.np
OUT=ROOT/'103_final_center_feasibility'

def count(a):
 if a.shape[1]==0:return 0
 return int((maximum_bipartite_matching(csr_matrix(a.numpy()),perm_type='column')>=0).sum())

def pair_iou(a,b):
 aa=n.box_cxcywh_to_xyxy(a);bb=n.box_cxcywh_to_xyxy(b)
 wh=(torch.minimum(aa[...,2:],bb[...,2:])-torch.maximum(aa[...,:2],bb[...,:2])).clamp(min=0)
 inter=wh.prod(-1)
 return inter/(a[...,2:].prod(-1)+b[...,2:].prod(-1)-inter).clamp(min=1e-12)

def metrics(pair,allowed,labels,small):
 mask=allowed[:,labels];r={}
 for typ,a in [('geo',pair),('class',pair*mask)]:
  for thr,suffix in [(.5,'05'),(.75,'075')]:
   for group,g in [('all',torch.ones_like(small)),('sub4',small)]:
    r[f'{group}_{typ}{suffix}']=count(a[:,g]>=thr)
 return r

def main():
 start=time.monotonic();resource=n.resource_limits();torch.set_grad_enabled(False)
 # Duplicate class rows never create a second physical query capacity.
 assert count(torch.tensor([[True,True]]))==1
 assert count(torch.eye(2,dtype=torch.bool))==2
 a=torch.tensor([[[.5,.5,.1,.1]]],dtype=torch.float64)
 assert abs(float(pair_iou(a,a))-1)<1e-12
 # Closed residual domain includes zero; projection at either boundary is exact.
 assert torch.equal(torch.maximum(torch.minimum(torch.tensor([4.,-4.]),torch.tensor([2.,2.])),torch.tensor([-2.,-2.])),torch.tensor([2.,-2.]))
 OUT.mkdir(exist_ok=False)
 inputs=['77_all_count_frozen/00_all_count_inputs.pt','85_source_self_attention/00_selected.json','91_source_quota_frozen/00_selections.json',
 '91_source_quota_frozen/seed1_predictions.pt','91_source_quota_frozen/seed2_predictions.pt']
 hashes={f:n.sha(ROOT/f) for f in inputs}
 data=torch.load(ROOT/inputs[0],map_location='cpu',weights_only=False)
 selected=json.loads((ROOT/inputs[1]).read_text(encoding='utf8'))
 cases=[next(c for c in data['cases'] if c['event']==r['event']) for r in selected]
 del data
 selections=json.loads((ROOT/inputs[2]).read_text(encoding='utf8'))
 rows=[];totals={};decisions={}
 arms=['original','P3_final_residual_2px','P3_grid_absolute_2px','P3_free_center','P3_GT_size_only']
 for seed in [1,2]:
  pred=torch.load(ROOT/f'91_source_quota_frozen/seed{seed}_predictions.pt',map_location='cpu',weights_only=False)['original']
  ss={a:{k:0 for k in ['all_geo05','all_geo075','all_class05','all_class075','sub4_geo05','sub4_geo075','sub4_class05','sub4_class075']} for a in arms}
  global_shifts={(dx,dy):{k:0 for k in ss['original']} for dx in [-2,-1,0,1,2] for dy in [-2,-1,0,1,2]}
  diagnostic={'GT':0,'sub4':0,'P3_slots':0,'P4_P5_slots':0,'native_detection_row_sub4_class05':0}
  for b,c in enumerate(cases):
   boxes=pred['pred_boxes'][b].double();logits=pred['pred_logits'][b]
   target=c['target'];gt=target['boxes'].double();labels=target['labels'];small=c['original_sub4'].bool()
   idx=torch.tensor(next(r['selected_indices'] for r in selections if r['model_seed']==seed and r['arm']=='original' and r['event']==c['event']))
   p3=idx<80*80
   grid=torch.stack(((idx%80+.5)*8,(torch.div(idx,80,rounding_mode='floor')+.5)*8),-1).double()/640
   _,flat=logits.sigmoid().flatten().topk(300);q=flat//10;cl=flat%10
   allowed=torch.zeros((300,10),dtype=torch.bool);allowed[q,cl]=True
   base=pair_iou(boxes[:,None,:],gt[None,:,:])
   diagnostic['native_detection_row_sub4_class05']+=count((base[q][:,small]*(cl[:,None]==labels[None,small]))>=.5)
   diagnostic['GT']+=len(gt);diagnostic['sub4']+=int(small.sum());diagnostic['P3_slots']+=int(p3.sum());diagnostic['P4_P5_slots']+=int((~p3).sum())
   for arm in arms:
    z=boxes[:,None,:].expand(-1,len(gt),-1).clone()
    if arm in ['P3_final_residual_2px','P3_grid_absolute_2px']:
     center=boxes[:,:2] if arm=='P3_final_residual_2px' else grid
     z[p3,:,:2]=torch.maximum(torch.minimum(gt[None,:,:2],center[p3,None,:]+2/640),center[p3,None,:]-2/640)
    elif arm=='P3_free_center':z[p3,:,:2]=gt[None,:,:2]
    elif arm=='P3_GT_size_only':z[p3,:,2:]=gt[None,:,2:]
    pair=base if arm=='original' else pair_iou(z,gt[None,:,:])
    if arm=='P3_final_residual_2px':assert bool((pair>=base-1e-10).all())
    m=metrics(pair,allowed,labels,small)
    rows.append({'seed':seed,'event':c['event'],'arm':arm,'counts':m})
    for k,v in m.items():ss[arm][k]+=v
   for (dx,dy),total in global_shifts.items():
    z=boxes.clone();z[p3,:2]+=torch.tensor([dx,dy])/640
    m=metrics(pair_iou(z[:,None,:],gt[None,:,:]),allowed,labels,small)
    for k,v in m.items():total[k]+=v
  best={k:max(v[k] for v in global_shifts.values()) for k in ss['original']}
  totals[str(seed)]={'counts':ss,'best_single_constant_P3_shift_per_metric':best,'denominators':diagnostic,
  'constant_shift_table':[{'shift_pixels':list(k),'counts':v} for k,v in global_shifts.items()]}
  decisions[str(seed)]={'nonzero_sub4_class05_recovery':ss['P3_final_residual_2px']['sub4_class05']>ss['original']['sub4_class05'],
  'recovery_above_best_constant':ss['P3_final_residual_2px']['sub4_class05']-best['sub4_class05']}
 n.save_json(OUT/'00_rows.json',rows)
 n.save_json(OUT/'01_results.json',{'state':'complete','resources':resource,'totals':totals,'necessary_gate':decisions,
 'both_nonzero':all(v['nonzero_sub4_class05_recovery'] for v in decisions.values()),'scope':'BASE24 cached development geometric bounds only; no phase information, C96, initial-reference or trained benefit validated',
 'input_sha256':hashes,'script_sha256':n.sha(__file__),'protocol_sha256':n.sha(ROOT/'101_末框中心残差作用面_事前口径.txt'),
 'actual_model_forwards':0,'model_backwards':0,'optimizer_steps':0,'new_training':False,'seconds':time.monotonic()-start})
 n.save_json(OUT/'02_SHA256.json',{p.name:n.sha(p) for p in OUT.iterdir() if p.is_file()})
 print(json.dumps({'state':'complete','totals':{s:{'counts':v['counts'],'best_constant':v['best_single_constant_P3_shift_per_metric'],'GT':v['denominators']} for s,v in totals.items()},'necessary_gate':decisions,'seconds':time.monotonic()-start},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
