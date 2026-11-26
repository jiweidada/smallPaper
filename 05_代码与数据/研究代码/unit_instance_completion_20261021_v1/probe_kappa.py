import os, sys, json, torch
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import train340 as T
torch.set_num_threads(2)
OUT = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
IDS = json.load(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\08_匹配聚合覆盖探针_20261019_v1\sample_plan.json", encoding='utf-8'))['source_pools']['development'][:32]
es = [T.load_entry(1,i) for i in IDS]
bs = [T.collate(es[0:16]), T.collate(es[16:32])]
torch.manual_seed(2026102101)
m = T.make_common_init('G', 1); m.train()
opt = T.build_optimizer(m)
rows = []
def rec(step):
    g = m.class_null.grad
    gm = 0.0 if g is None else float(g.abs().max())
    nz = 0 if g is None else int((g.abs() > 0).sum())
    rows.append({'step': step, 'grad_max_abs': gm, 'grad_nonzero': nz,
                 'kappa_max_abs': float(m.class_null.detach().abs().max()),
                 'classifier_w_max_abs': float(m.classifier.weight.detach().abs().max()),
                 'boxhead_w_max_abs': float(m.box_head[-1].weight.detach().abs().max())})
opt.zero_grad(set_to_none=True)
losses, _ = T.compute_losses_340('G', m, bs[0])
losses['loss_total'].backward()
rec(0)
for step in range(1, 21):
    lr = T.lr_at(step, 1218)
    for gi in opt.param_groups: gi['lr'] = lr
    opt.step()
    opt.zero_grad(set_to_none=True)
    losses, _ = T.compute_losses_340('G', m, bs[step % 2])
    losses['loss_total'].backward()
    rec(step)
sel = [r for r in rows if r['step'] in (0,1,5,10,20)]
print(json.dumps(sel, indent=1))
json.dump({'item':'340b kappa_c per-step gradient probe','rows':rows},
          open(os.path.join(OUT,'340b_kappa_probe.json'),'w',encoding='utf-8'), ensure_ascii=False, indent=2)