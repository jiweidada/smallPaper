from pathlib import Path
import hashlib,importlib.metadata,json,subprocess,sys,zipfile
import numpy as np
from PIL import Image
D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs=json.loads((D/'collector_inputs.json').read_text(encoding='utf-8'));assert str(D)==inputs['remote_root']
bundle=D/'new_val_inputs_bundle.zip';assert sha(bundle)==inputs['archive_sha256']
expected={}
for r in inputs['cases']:
    if r['split']=='val':
        expected['inputs/val/images/'+r['file']]=r['image_sha256']
        expected['inputs/val/annotations/'+Path(r['annotation_path']).name]=r['annotation_sha256']
assert len(expected)==114
with zipfile.ZipFile(bundle) as z:
    assert set(z.namelist())==set(expected)
    for n,h in expected.items():
        raw=z.read(n);assert hashlib.sha256(raw).hexdigest()==h
        target=(D/n).resolve();assert target.is_relative_to(D)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(raw)
annotation_rows=0
for r in inputs['cases']:
    allowed=Path('/root/c10_region_restore_20261029_v1/dataset/train') if r['split']=='train' else D/'inputs/val'
    for field,hfield in [('image_path','image_sha256'),('annotation_path','annotation_sha256')]:
        p=Path(r[field]).resolve();assert p.is_relative_to(allowed) and sha(p)==r[hfield]
    with Image.open(r['image_path']) as image:assert image.size==(r['width'],r['height'])
    text=Path(r['annotation_path']).read_text().strip()
    a=np.loadtxt(r['annotation_path'],delimiter=',',ndmin=2).reshape(-1,8) if text else np.empty((0,8))
    assert np.isfinite(a).all() and ((a[:,5]>=0)&(a[:,5]<=11)).all();annotation_rows+=len(a)
assets=Path('/root/c10_region_restore_20261029_v1/architecture_probe_v1/dome_assets_v1')
assert not (D/'forward_entries.jsonl').exists() and not (D/'captures').exists()
result=dict(status='ALL112_ADDITIONAL_INPUTS_VERIFIED_BEFORE_MODEL',asset_hashes_verified=224,
    new_train=55,new_val=57,annotation_rows=annotation_rows,new_author_model_calls=0,
    loader_sha256=sha(assets/'dome_probe_loader.py'),checkpoint_sha256=sha(assets/'dome-s-visdrone_converted.pth'),
    torch=importlib.metadata.version('torch'),numpy=np.__version__,
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader'],text=True).strip())
with (D/'remote_input_preflight.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
