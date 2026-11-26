import os, sys, json, torch
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import train340 as A, train341 as B
torch.set_num_threads(2)
OUT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
IDS = json.load(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\08_匹配聚合覆盖探针_20261019_v1\sample_plan.json", encoding='utf-8'))['source_pools']['development'][:32]
es=[A.load_entry(1,i) for i in IDS]; bs=[A.collate(es[0:16]), A.collate(es[16:32])]
res={'item':'341 317v4 落实验证','tests':{}}
# 1) lr curves step-by-step identical for N/S/R
v3=[A.lr_at(s,1218) for s in range(1218)]
v4=[B.lr_at(s,1218) for s in range(1218)]
res['tests']['non_kappa_lr_identical']={'max_diff':max(abs(a-b) for a,b in zip(v3,v4)),'pass':all(abs(a-b)<1e-12 for a,b in zip(v3,v4))}
# 2) N/S/R optimizer groups identical (no kappa group)
for arm in ['N','S','R']:
    torch.manual_seed(1); m1=A.make_common_init(arm,1); o1=A.build_optimizer(m1)
    torch.manual_seed(1); m2=B.make_common_init(arm,1); o2=B.build_optimizer(m2)
    res['tests'][f'{arm}_optimizer']={'groups_v3':len(o1.param_groups),'groups_v4':len(o2.param_groups),'same':len(o1.param_groups)==len(o2.param_groups) and all(abs(g1['lr']-g2['lr'])<1e-12 for g1,g2 in zip(o1.param_groups,o2.param_groups))}
# 3) G: only kappa group differs
torch.manual_seed(1); m1=A.make_common_init('G',1); o1=A.build_optimizer(m1)
torch.manual_seed(1); m2=B.make_common_init('G',1); o2=B.build_optimizer(m2)
res['tests']['G_optimizer']={'groups_v3':len(o1.param_groups),'groups_v4':len(o2.param_groups),
  'non_kappa_lrs_same':all(abs(o1.param_groups[i]['lr']-o2.param_groups[i]['lr'])<1e-12 for i in range(2)),
  'kappa_group_lr':o2.param_groups[2]['lr'],'kappa_ratio':o2.param_groups[2]['lr']/o2.param_groups[0]['lr']}
# 4) G 20 steps
torch.manual_seed(2026102101)
m=B.make_common_init('G',1); m.train(); opt=B.build_optimizer(m)
rows=[]
for step in range(20):
    lr=B.lr_at(step,1218); klr=B.lr_at_kappa(step,1218)
    for gi,g in enumerate(opt.param_groups): g['lr']= klr if gi==len(opt.param_groups)-1 else lr
    opt.zero_grad(set_to_none=True)
    losses,_=B.compute_losses_340('G',m,bs[step%2])
    losses['loss_total'].backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
    g=m.class_null.grad
    rows.append({'step':step,'grad_max':float(g.abs().max()) if g is not None else None,
                 'kappa_max':float(m.class_null.detach().abs().max()),'loss':float(losses['loss_total'].detach())})
    opt.step()
finite=all(all(isinstance(v,(int,float)) and v==v and abs(v)<1e9 for v in [r['grad_max'],r['kappa_max'],r['loss']]) for r in rows)
res['tests']['G_20step_smoke']={'finite':bool(finite),'final_kappa_max':rows[-1]['kappa_max'],
  'max_grad':max(r['grad_max'] for r in rows),'loss_first':rows[0]['loss'],'loss_last':rows[-1]['loss'],
  'kappa_displacement_max':rows[-1]['kappa_max'],
  'extrapolated_1218':round(rows[-1]['kappa_max']/20*1218,4),
  'rows':rows}
print(json.dumps({k:res['tests'][k] for k in ['non_kappa_lr_identical','N_optimizer','S_optimizer','R_optimizer','G_optimizer']},indent=1))
print('G smoke:', {k:v for k,v in res['tests']['G_20step_smoke'].items() if k!='rows'})
json.dump(res,open(os.path.join(OUT,'341_result_317v4.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=2)