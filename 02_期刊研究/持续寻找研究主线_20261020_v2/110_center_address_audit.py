import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUDA_VISIBLE_DEVICES']=''
import pathlib,sys,json,importlib.util,time
ROOT=pathlib.Path(__file__).resolve().parent;WORK=ROOT.parents[1]
sys.path.insert(0,str(WORK/'英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1'))
import common as n
torch=n.torch
spec=importlib.util.spec_from_file_location('f102',ROOT/'102_final_center_feasibility.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
def main():
 start=time.monotonic();resources=n.resource_limits();torch.set_grad_enabled(False)
 out=ROOT/'111_center_address_audit';out.mkdir(exist_ok=False)
 selected=json.loads((ROOT/'85_source_self_attention/00_selected.json').read_text(encoding='utf8'))
 data=torch.load(ROOT/'77_all_count_frozen/00_all_count_inputs.pt',map_location='cpu',weights_only=False)
 cases=[next(c for c in data['cases'] if c['event']==r['event']) for r in selected];del data
 selections=json.loads((ROOT/'91_source_quota_frozen/00_selections.json').read_text(encoding='utf8'))
 rows=[];totals={};gate={}
 for seed in [1,2]:
  pred=torch.load(ROOT/f'91_source_quota_frozen/seed{seed}_predictions.pt',map_location='cpu',weights_only=False)['original'];tt={}
  for b,c in enumerate(cases):
   box=pred['pred_boxes'][b].double();gt=c['target']['boxes'].double();lab=c['target']['labels'];small=c['original_sub4'].bool()
   idx=torch.tensor(next(r['selected_indices'] for r in selections if r['model_seed']==seed and r['arm']=='original' and r['event']==c['event']));p3=idx<6400
   lo=torch.stack((idx%80,torch.div(idx,80,rounding_mode='floor')),-1).double()*8/640;hi=lo+8/640
   inside_gt=((gt[None,:,:2]>=lo[:,None,:])&(gt[None,:,:2]<hi[:,None,:])).all(-1)&p3[:,None]
   inside_final=((box[:,:2]>=lo)&(box[:,:2]<hi)).all(-1)&p3
   z=box[:,None,:].expand(-1,len(gt),-1).clone();z[p3,:,:2]=torch.maximum(torch.minimum(gt[None,:,:2],box[p3,None,:2]+2/640),box[p3,None,:2]-2/640)
   base=f.pair_iou(box[:,None,:],gt[None,:,:]);bound=f.pair_iou(z,gt[None,:,:])
   _,flat=pred['pred_logits'][b].sigmoid().flatten().topk(300);allow=torch.zeros((300,10),dtype=torch.bool);allow[flat//10,flat%10]=True
   for typ,cm in [('geo',torch.ones_like(base,dtype=torch.bool)),('class',allow[:,lab])]:
    for threshold,suffix in [(.5,'05'),(.75,'075')]:
     old=(base>=threshold)&cm;new=(bound>=threshold)&cm&~old
     masks={'original':old,'all_new':old|new,'GT_in_old_grid':old|(new&inside_gt),'GT_out_old_grid':old|(new&~inside_gt),
     'final_center_in_old_grid':old|(new&inside_final[:,None]),'final_center_out_old_grid':old|(new&~inside_final[:,None])}
     for group,g in [('all',torch.ones_like(small)),('sub4',small)]:
      key=f'{group}_{typ}{suffix}';r={k:f.count(v[:,g]) for k,v in masks.items()}
      rows.append({'seed':seed,'event':c['event'],'metric':key,'maximum_matchings':r,'new_feasible_edges':int(new[:,g].sum()),
      'new_edges_GT_out_old_grid':int((new&~inside_gt)[:,g].sum()),'new_edges_final_center_out_old_grid':int((new&~inside_final[:,None])[:,g].sum())})
      total=tt.setdefault(key,{k:0 for k in masks})
      for k,v in r.items():total[k]+=v
  totals[str(seed)]=tt;rr=tt['sub4_class05'];gate[str(seed)]={'extra_capacity_from_GT_out_old_grid':rr['GT_out_old_grid']-rr['original'],'extra_capacity_from_final_out_old_grid':rr['final_center_out_old_grid']-rr['original']}
 n.save_json(out/'00_rows.json',rows);n.save_json(out/'01_results.json',{'state':'complete','resources':resources,'totals':totals,'necessary_address_evidence':gate,'script_sha256':n.sha(__file__),'protocol_sha256':n.sha(ROOT/'109_中心残差读取地址审计_事前口径.txt'),
 'source_sha256':{s:n.sha(ROOT/s) for s in ['103_final_center_feasibility/01_results.json','91_source_quota_frozen/00_selections.json','91_source_quota_frozen/seed1_predictions.pt','91_source_quota_frozen/seed2_predictions.pt']},
 'model_forwards':0,'model_backwards':0,'optimizer_steps':0,'new_training':False,'heldout_reads':False,'scope':'Address-specific geometric opportunity only; no source feature absence or local head novelty established','seconds':time.monotonic()-start})
 print(json.dumps({'state':'complete','totals':totals,'address_evidence':gate},ensure_ascii=False))
if __name__=='__main__':main()
