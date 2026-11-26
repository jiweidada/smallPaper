
import json
p=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_redo1_real_model.json"
d=json.load(open(p,encoding="utf-8")); d.setdefault("not_executed",[])
json.dump(d,open(p,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("keys", list(d.keys()))
