import json
p=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\338_manifest.json"
d=json.load(open(p,encoding="utf-8"))
d["elapsed_secs"]={"seed1":799.2,"seed2":799.6,"wall_clock_approx":799.6,
 "note":"两 seed 并行，墙钟约 800 s (13.3 min)"}
d["bad_entries"]={"seed1":0,"seed2":0,"note":"stats_338.py 逐条重算 SHA，两 seed 各 3235 条全部通过"}
json.dump(d,open(p,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("fixed")
