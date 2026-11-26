"""340 CPU smoke: four arms x 20 steps on the 32-image 340 subset. NOT a training authorization."""
import os, sys, json, time, math
import numpy as np, torch
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import train340 as T
torch.set_num_threads(2)
OUT = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
IDS = json.load(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\08_匹配聚合覆盖探针_20261019_v1\sample_plan.json", encoding="utf-8"))["source_pools"]["development"][:32]

def make_batches(seed, ids, bs=16):
    entries = [T.load_entry(seed, i) for i in ids if os.path.exists(os.path.join(OUT, f'340_cache_seed{seed}', f'{i:06d}.pt'))]
    batches = []
    for i in range(0, len(entries), bs):
        batches.append(T.collate(entries[i:i+bs]))
    return batches

def main():
    res = {"item": "340 阶段2 CPU冒烟", "arms": {}, "not_training_authorization": True}
    batches = make_batches(1, IDS)
    res["batch_info"] = {"n_batches": len(batches), "batch_sizes": [len(b["entries"]) for b in batches]}
    for arm in ['N','S','G','R']:
        torch.manual_seed(2026102101)
        model = T.make_common_init(arm, 1)
        model.train()
        opt = T.build_optimizer(model)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        t0 = time.time()
        log = T.run_steps(model, opt, batches, 0, 20, total_steps=1218)
        dt = time.time() - t0
        changed = sum(1 for n, p in model.named_parameters() if not torch.equal(before[n], p.detach()))
        finite = all(all(np.isfinite(v) for v in row.values()) for row in log)
        last = log[-1] if log else {}
        res['arms'][arm] = {'steps': len(log), 'secs': round(dt,2),
            'loss_finite': bool(finite),
            'last_losses': {k: round(float(v),6) for k,v in last.items()},
            'params_changed': changed,
            'param_count': sum(p.numel() for p in model.parameters()),
            'has_class_null': hasattr(model, 'class_null')}
        if arm == 'G':
            g = model.class_null.grad
            res['arms'][arm]['class_null_grad'] = None if g is None else [round(float(x),8) for x in g]
            res['arms'][arm]['class_null_nonzero'] = int((g.abs()>0).sum()) if g is not None else 0
    json.dump(res, open(os.path.join(OUT,'340_result_smoke.json'),'w',encoding='utf-8'), ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2)[:3000])

if __name__ == "__main__":
    main()
