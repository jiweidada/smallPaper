from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,shutil,sys,zipfile
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent; B=D.parent
P=B/'A37_局部差分编码与语义判别证据核查_v1'
Z=B/'official_assets_v1/Dome_DETR_2dde3bc_v1/author_source.zip'
assert hashlib.sha256(Z.read_bytes()).hexdigest()=='6df0a0f5d7f27ce4304b7643a1b3a87b5ad7bf50f10b7a2fbc92ed8ddc231f41'
prior=json.loads((B/'85_A37原文源码及诊断边界核验回执.json').read_text(encoding='utf-8'))
for path,digest in prior['files_sha256'].items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
rows=[]
with zipfile.ZipFile(Z) as archive:
    prefix='Dome-DETR-2dde3bc1946a3e9fad9abd0612b59fc39bd6b861/'
    for name in ['src/zoo/dome/dome_decoder.py','src/zoo/dome/dome.py','src/zoo/dome/postprocessor.py',
                 'src/zoo/dome/hybrid_encoder.py','configs/dome/Dome-S-VisDrone.yml']:
        raw=archive.read(prefix+name); target=D/'pinned_source'/name; target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as output: output.write(raw)
        rows.append(dict(path=name,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
loader=P/'existing_dome_probe_loader.py'; expected='8dc66cf817d387733a895501dc160dad703eedad9d061ce8f4ffd95296b7c1ab'
assert hashlib.sha256(loader.read_bytes()).hexdigest()==expected
with (D/'pinned_dome_loader.py').open('xb') as output: output.write(loader.read_bytes())
receipt=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),previous_goal_turn='PROGRESS_A37added2nearestpapers,25verifiedauthorblobs,andfixed112imageoutcome-blindsample.',
    prior_A37_files_verified=75,source_commit='2dde3bc1946a3e9fad9abd0612b59fc39bd6b861',
    source_zip_sha256=hashlib.sha256(Z.read_bytes()).hexdigest(),selected_source_files=rows,
    loader_sha256=expected,source_scope='Selected copies from previously293Git-blob-verified archive; archive SHA rechecked; static only.',
    new_model_call_allocation=0,new_probe_fit_allocation=0,new_optimizer_update_allocation=0)
with (D/'source_pin_and_handoff.json').open('x',encoding='utf-8') as output: json.dump(receipt,output,ensure_ascii=False,indent=2)
print(json.dumps(dict(status='SOURCE_PINNED_NOT_EXECUTED',prior_files_verified=75,selected_source_files=len(rows))))
