from pathlib import Path
import json,hashlib,tarfile
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/"A43_ECDet三组蒸馏最小验证_20261031_v1"
SRC=OLD/"source/EdgeCrafter-d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecdetseg"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
exp=json.loads((ROOT/"asset_expectations.json").read_text(encoding="utf-8"))
for rel,h in exp["native_source_sha256"].items():assert sha(SRC/rel)==h,rel
files={}
for p in SRC.rglob("*"):
    if p.is_file() and p.suffix.lower() in {".py",".yaml",".yml",".json"} and "__pycache__" not in p.parts:
        files["assets_A43_baseline/source/EdgeCrafter-d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecdetseg/"+p.relative_to(SRC).as_posix()]=p
for rel,key in [("runtime_common.py","runtime_common_sha256"),("common_init_10class.pth","common_init_sha256"),("train_data_manifest.json","train_data_manifest_sha256")]:
    p=OLD/rel;assert sha(p)==exp[key];files["assets_A43_baseline/"+rel]=p
p=OLD/"completed_results/B/arm_B/epoch12.pth"
assert sha(p)==exp["checkpoint_sha256"];files["assets_A43_baseline/arm_B/epoch12.pth"]=p
for name in ["asset_expectations.json","protocol_FROZEN.json","deployment_30172_amendment_FROZEN.json",
             "run_A47_match_analysis.py","run_A47_frozen_match_capture_30172_v1.py"]:
    files[name]=ROOT/name
manifest={arc:{"bytes":p.stat().st_size,"sha256":sha(p)} for arc,p in files.items()}
(ROOT/"deployment_bundle_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
files["deployment_bundle_manifest.json"]=ROOT/"deployment_bundle_manifest.json"
archive=ROOT/"A47_minimal_assets_30172_v1.tar"
assert not archive.exists()
with tarfile.open(archive,"w") as tf:
    for arc,p in files.items():tf.add(p,arcname=arc,recursive=False)
receipt={"status":"PACKED","files":len(files),"bytes":archive.stat().st_size,"sha256":sha(archive),
         "manifest_sha256":sha(ROOT/"deployment_bundle_manifest.json"),
         "scope":"Exact old baseline source/init/checkpoint plus new frozen diagnostic scripts. No dataset/images or training queue."}
(ROOT/"deployment_bundle_receipt.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
print(json.dumps(receipt))

