from pathlib import Path
import hashlib,json,subprocess,sys,zipfile
from PIL import Image

D=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
inputs=json.loads((D/'collector_inputs.json').read_text(encoding='utf-8'))
assert str(D)==inputs['remote_root']
archive=D/'val_inputs_bundle.zip';assert sha(archive)==inputs['archive_sha256']
expected={}
for r in inputs['cases']:
    if r['split']=='val':
        expected['inputs/val/images/'+r['file']]=r['image_sha256']
        expected['inputs/val/annotations/'+Path(r['annotation_path']).name]=r['annotation_sha256']
assert len(expected)==96
with zipfile.ZipFile(archive) as z:
    assert set(z.namelist())==set(expected)
    for name,digest in expected.items():
        raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==digest
        target=(D/name).resolve();assert target.is_relative_to(D)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(raw)
for r in inputs['cases']:
    allowed=Path('/root/c10_region_restore_20261029_v1/dataset/train') if r['split']=='train' else D/'inputs/val'
    for key,hkey in [('image_path','image_sha256'),('annotation_path','annotation_sha256')]:
        target=Path(r[key]).resolve();assert target.is_relative_to(allowed)
        assert sha(target)==r[hkey],r['file']
    with Image.open(r['image_path']) as im:assert im.size==(r['width'],r['height'])
assert not (D/'captures').exists() and not (D/'forward_entries.jsonl').exists()
result=subprocess.run([sys.executable,str(D/'scaled_pool_fixture.py')],cwd=str(D),capture_output=True,text=True)
print(result.stdout);print(result.stderr);assert result.returncode==0
receipt=dict(status='ALL112_INPUTS_VERIFIED_BEFORE_AUTHOR_MODEL',cases=112,asset_hashes_verified=224,
    uploaded_val_files=96,archive_sha256=sha(archive),captures_exist=False,forward_ledger_exists=False,
    author_detector_calls=0,diagnostic_data_probe_fits=0)
with (D/'remote_input_preflight.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
