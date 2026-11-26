from pathlib import Path,PurePosixPath
import collections,hashlib,json,sys,zipfile
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=read(D/'phase_protocol_FROZEN.json');complete=read(D/'remote_complete.json');archive=D/'native_evidence.zip'
assert archive.stat().st_size==complete['archive_bytes'] and sha(archive)==complete['archive_sha256']
out=D/'downloaded_evidence';out.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:
 assert len(z.infolist())==complete['archived_files']
 for info in z.infolist():
  path=PurePosixPath(info.filename);assert not path.is_absolute() and '..' not in path.parts
  target=out.joinpath(*path.parts).resolve();assert out.resolve() in target.parents
  target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as f:f.write(z.read(info))
manifest=read(out/'file_manifest.json')['files_sha256']
for p,h in manifest.items():assert sha(out/p)==h,p
assert len(manifest)+1==complete['archived_files']
assert sha(out/'phase_protocol_FROZEN.json')==sha(D/'phase_protocol_FROZEN.json')
assert sha(out/'native_phase_worker.py')==sha(D/'native_phase_worker.py')
proof=read(out/'pixel_and_zero_fidelity.json');assert len(proof['proof'])==168 and all(x['real_pixels_exact'] and x['full_inverse_exact'] for x in proof['proof'])
assert proof['native_repeat_bitwise'] and proof['zero_fidelity_cases']==24
old=D.parent/'A30_YOLO26双头原生读出核查_v1/downloaded_evidence'
native_names={'M':'Mbox_Mscore_NMS_predictions_original.json','O':'Obox_Oscore_TOPK_predictions_original.json'}
old_pred={}
for k,name in native_names.items():
 old_pred[k]=collections.defaultdict(list)
 for x in read(old/name):old_pred[k][x['image_id']].append(x)
 fresh=collections.defaultdict(list)
 for x in read(out/(k+'_0_predictions_original.json')):fresh[x['image_id']].append(x)
 for c in cfg['cases']:assert fresh[c['image_id']]==old_pred[k][c['image_id']]
checks=0
for i,c in enumerate(cfg['cases']):
 a=np.load(out/'captures'/('case_%03d_shift_%+03d.npz'%(i,0)),allow_pickle=False)
 b=np.load(old/'captures'/('case_%03d.npz'%c['original_A30_index']),allow_pickle=False)
 for k in ['M','O']:
  for field in ['boxes','scores','decoded_xyxy']:assert np.array_equal(a[k+'_'+field],b[k+'_'+field]);checks+=1
 a.close();b.close()
receipt=dict(status='PASS_ARCHIVE_ALL_FILES_PIXEL_PROOFS_AND_ZERO_NATIVE_FIDELITY',archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),archive_file_count=complete['archived_files'],manifest_files_verified=len(manifest),zero_raw_and_decoded_arrays_independently_compared=checks,zero_native_image_branch_records_compared=48,pixel_preservation_conditions=168,native_repeat_bitwise=proof['native_repeat_bitwise'],counts=complete['counts'],scope='Artifactandfidelityverificationonly;not mechanismoralgorithmvalidation.')
with (D/'archive_verification.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print(json.dumps(receipt,ensure_ascii=False))
