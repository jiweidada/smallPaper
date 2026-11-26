import json, os
p = os.path.join(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1", "340b_kappa_probe.json")
d = json.load(open(p, encoding="utf-8"))
rows = d["rows"]
import math
print("step, kappa_max_abs, grad_max")
for r in rows: print(r["step"], "%.3e" % r["kappa_max_abs"], "%.3e" % r["grad_max_abs"])
# rate: dkappa/dstep around step 10-20
k10 = rows[10]["kappa_max_abs"]; k20 = rows[20]["kappa_max_abs"]
print("kappa delta 10->20:", "%.3e" % (k20-k10), "per step:", "%.3e" % ((k20-k10)/10))
print("extrapolated 1218 steps:", "%.3e" % ((k20-k10)/10*1218))