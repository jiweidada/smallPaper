import sys, json, torch
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import train340 as T
torch.set_num_threads(2)
m = T.make_common_init('G', 1)
opt = T.build_optimizer(m)
for i, g in enumerate(opt.param_groups):
    names = [k for k,v in m.named_parameters() if any(v is p for p in g["params"])]
    print(f"group {i}: n_params={len(g['params'])} wd={g['weight_decay']} lr={g['lr']}")
    print("   class_null in group:", "class_null" in names)
print("total params in optimizer:", sum(len(g["params"]) for g in opt.param_groups))
print("all model params:", len(list(m.parameters())))