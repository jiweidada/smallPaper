import json
from pathlib import Path
p=Path(r"C:\Users\jiwei\Desktop\小论文投稿\01_会议论文\修订记录_20261103\第三十八轮第320轮结果核验_20261111\candidate_structural_and_evidence_check.json")
d=json.loads(p.read_text(encoding="utf-8"))
d["FULL_minus_NO_MSCA_AP_percentage_points"]=0.331560050934858
d["FULL_minus_NO_MSCA_APs_percentage_points"]=0.658547292182916
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({k:d[k] for k in ["FULL_minus_NO_MSCA_AP_percentage_points","FULL_minus_NO_MSCA_APs_percentage_points"]}))
