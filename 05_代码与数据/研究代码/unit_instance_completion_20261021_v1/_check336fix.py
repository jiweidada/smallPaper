import json
SP=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\08_匹配聚合覆盖探针_20261019_v1\sample_plan.json"
p327=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2\327_训练实施输入索引.json"
raw=json.load(open(SP,encoding="utf-8"))["source_pools"]["development"]
d327=json.load(open(p327,encoding="utf-8"))
lst=d327["sample_plan"]["development_ids"]
print("raw len",len(raw),"327 len",len(lst))
print("raw first 8:",raw[:8])
print("327 first 8:",lst[:8])
print("sorted(raw)==327 list:", sorted(raw)==lst)
print("raw==327 list:", raw==lst)
print("same set:", set(raw)==set(lst))
