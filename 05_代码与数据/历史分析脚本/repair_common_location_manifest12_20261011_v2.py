"""Versioned metadata fix; recompute actual saved-array joins without model calls."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent.parent
old=R/'英文小论文/code/native_selection_seed2_common_location12_20261011_v1'
new=R/'英文小论文/code/native_selection_seed2_common_location12_20261011_v2';new.mkdir(exist_ok=False)
source=(old/'analyze.py').read_text(encoding='utf8')
before="    with (output/'manifest.json').open('x',encoding='utf8') as f:json.dump(dict(files={x.name:sha(x) for x in output.iterdir() if x.is_file()}),f,indent=2)"
after="    files={x.name:sha(x) for x in output.iterdir() if x.is_file()}\n    with (output/'manifest.json').open('x',encoding='utf8') as f:json.dump(dict(files=files),f,indent=2)"
assert source.count(before)==1
source=source.replace(before,after).replace("output=ROOT/'outputs/native_selection_seed2_common_location12_20261011_v1'","output=ROOT/'outputs/native_selection_seed2_common_location12_20261011_v2'")
assert "output=ROOT/'outputs/native_selection_seed2_common_location12_20261011_v2'" in source
(new/'analyze.py').write_text(source,encoding='utf8')
spec=json.loads((old/'protocol.json').read_bytes());spec['reader_sha256']=hashlib.sha256((new/'analyze.py').read_bytes()).hexdigest()
spec['preserved_v1_source_sha256']=hashlib.sha256((old/'analyze.py').read_bytes()).hexdigest()
spec['repair_scope']='Only output location and manifest creation order. V1 self-entry hashed its newly opened empty manifest; all original numeric arrays remain unchanged. Recompute all saved-array joins in new output.'
(new/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
failure=R/'outputs/common_location_manifest12_20261011_v1_failure.json'
failure.write_text(json.dumps(dict(status='V1_METADATA_SELF_HASH_REJECTED_NO_MODEL_FAILURE',original_manifest_sha256=hashlib.sha256((R/'outputs/native_selection_seed2_common_location12_20261011_v1/manifest.json').read_bytes()).hexdigest(),reason='manifest opened before file enumeration; it bound its own empty bytes',new_model_forwards=0,new_training_epochs=0,original_outputs_preserved=True),indent=2),encoding='utf8')
print(spec['reader_sha256'])
