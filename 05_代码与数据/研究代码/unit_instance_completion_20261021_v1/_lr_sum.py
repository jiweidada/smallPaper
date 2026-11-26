import sys, math, json
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import train340 as T
total = 1218
lrs = [T.lr_at(s, total) for s in range(total)]
s = sum(lrs)
print("n steps:", total)
print("LR sum:", round(s, 6))
print("LR first 5:", [round(x,7) for x in lrs[:5]])
print("LR at 99,100,101:", [round(lrs[i],7) for i in (99,100,101)])
print("LR at 1217:", round(lrs[-1],7))
print("max LR:", round(max(lrs),7))
print("mean LR:", round(s/total,7))
# Adam upper bound: per-step update <= lr (bias-corrected, |m_hat/sqrt(v_hat)| <= ~1)
print("theoretical max displacement (sum of lr):", round(s,4))