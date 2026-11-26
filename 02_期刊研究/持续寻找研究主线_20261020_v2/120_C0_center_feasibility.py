import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUDA_VISIBLE_DEVICES']=''
import pathlib,sys,json,importlib.util,time
ROOT=pathlib.Path(__file__).resolve().parent;WORK=ROOT.parents[1]
sys.path.insert(0,str(WORK/'英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1'))
import common as n
spec=importlib.util.spec_from_file_location('c0_102',ROOT/'102_final_center_feasibility.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
torch=n.torch
def main():
 start=time.monotonic();resource=n.resource_limits();torch.set_grad_enabled(False)
 out=ROOT/'121_C0_center_feasibility';out.mkdir(exist_ok=False)
 source=ROOT/'114_C0_restore_route_frozen/03_eval_predictions_and_queries.pt';data=torch.load(source,map_location='cpu',weights_only=False)
 execution=json.loads((ROOT/'114_C0_restore_route_frozen/04_execution.json').read_text(encoding='utf8'));assert execution['state']=='complete' and execution['checkpoint_unchanged']
 assert len(data['cases'])==4 and data['mode']=='eval'
 arms=['original','P3_final_residual_2px','P3_grid_absolute_2px','P3_free_center','P3_GT_size_only'];rows=[];totals={a:{} for a in arms};shifts={}
 for case in data['cases']:
  box=case['pred_boxes'].double();gt=case['target']['boxes'].double();labels=case['target']['labels'];small=case['original_sub4'].bool();idx=case['selected_indices'];p3=idx<6400
  grid=torch.stack(((idx%80+.5)*8,(torch.div(idx,80,rounding_mode='floor')+.5)*8),-1).double()/640
  _,flat=case['pred_logits'].sigmoid().flatten().topk(300);allow=torch.zeros((300,10),dtype=torch.bool);allow[flat//10,flat%10]=True
  for arm in arms:
   z=box[:,None,:].expand(-1,len(gt),-1).clone()
   if arm in ['P3_final_residual_2px','P3_grid_absolute_2px']:
    center=box[:,:2] if arm=='P3_final_residual_2px' else grid
    z[p3,:,:2]=torch.maximum(torch.minimum(gt[None,:,:2],center[p3,None,:]+2/640),center[p3,None,:]-2/640)
   elif arm=='P3_free_center':z[p3,:,:2]=gt[None,:,:2]
   elif arm=='P3_GT_size_only':z[p3,:,2:]=gt[None,:,2:]
   m=f.metrics(f.pair_iou(z,gt[None,:,:]),allow,labels,small);rows.append({'event':case['event'],'arm':arm,'counts':m})
   for k,v in m.items():totals[arm][k]=totals[arm].get(k,0)+v
  for dx in [-2,-1,0,1,2]:
   for dy in [-2,-1,0,1,2]:
    z=box.clone();z[p3,:2]+=torch.tensor([dx,dy])/640
    m=f.metrics(f.pair_iou(z[:,None,:],gt[None,:,:]),allow,labels,small);t=shifts.setdefault((dx,dy),{})
    for k,v in m.items():t[k]=t.get(k,0)+v
 best={k:max(v[k] for v in shifts.values()) for k in totals['original']}
 n.save_json(out/'00_rows.json',rows)
 result={'state':'complete','resources':resource,'checkpoint_sha256':data['checkpoint_sha256'],'weight_key':'model_online','epoch':94,'GT':sum(len(c['original_sub4']) for c in data['cases']),'sub4':sum(int(c['original_sub4'].sum()) for c in data['cases']),
 'totals':totals,'best_constant_per_metric':best,'constant_shift_table':[{'pixels':list(k),'counts':v} for k,v in shifts.items()],
 'necessary_gate_nonzero':totals['P3_final_residual_2px']['sub4_class05']>totals['original']['sub4_class05'],'scope':'Single C0 online cached geometric bounds, not cross-seed validation, initial-ref intervention, learnability, trained AP or novelty',
 'source_sha256':n.sha(source),'script_sha256':n.sha(__file__),'protocol_sha256':n.sha(ROOT/'119_C0末框中心残差作用面_缓存复核口径.txt'),'helper_sha256':n.sha(ROOT/'102_final_center_feasibility.py'),'model_forwards':0,'model_backwards':0,'optimizer_steps':0,'new_training':False,'seconds':time.monotonic()-start}
 n.save_json(out/'01_results.json',result);print(json.dumps({k:result[k] for k in ['state','totals','best_constant_per_metric','necessary_gate_nonzero','seconds']},ensure_ascii=False))
if __name__=='__main__':main()
