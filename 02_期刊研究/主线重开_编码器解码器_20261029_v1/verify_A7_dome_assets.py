"""Verify an author source archive and published LFS weight hash; do not import it."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import hashlib
import json
import zipfile

root=Path(__file__).resolve().parent
assets=root/'official_assets_v1'/'Dome_DETR_2dde3bc_v1'
tree=json.loads((root/'source_audit_v1'/'Dome_author_tree.json').read_text(encoding='utf-8-sig'))
expected={x['path']:x for x in tree['tree'] if x['type']=='blob'}
assert not tree.get('truncated')
commit='2dde3bc1946a3e9fad9abd0612b59fc39bd6b861'
prefix='Dome-DETR-'+commit+'/'
dest=assets/'source'
verified=[]
with zipfile.ZipFile(assets/'author_source.zip') as z:
    assert z.testzip() is None
    names=[i for i in z.infolist() if not i.is_dir()]
    actual={i.filename[len(prefix):] for i in names if i.filename.startswith(prefix)}
    assert actual==set(expected),(set(expected)-actual,actual-set(expected))
    assert all(i.filename.startswith(prefix) for i in names)
    for i in names:
        rel=i.filename[len(prefix):]
        p=PurePosixPath(rel)
        assert not p.is_absolute() and '..' not in p.parts and '\\' not in rel
        content=z.read(i)
        gitsha=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        assert gitsha==expected[rel]['sha'],rel
        assert len(content)==expected[rel]['size'],rel
        target=(dest/rel).resolve()
        assert target.is_relative_to(dest.resolve())
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content)
        verified.append({'path':rel,'git_blob_sha':gitsha,'bytes':len(content),
                         'sha256':hashlib.sha256(content).hexdigest()})
weight_repo=json.loads((assets/'author_weight_repo_metadata.json').read_text(encoding='utf-8'))
target_name='best_ckpts_dome_2026/dome-s-visdrone_converted.pth'
entry=next(x for x in weight_repo['siblings'] if x['rfilename']==target_name)
weight=assets/'dome-s-visdrone_converted.pth'
weight_result={'status':'NOT_PRESENT_OR_DOWNLOAD_INCOMPLETE','path':str(weight),
               'repo_commit':weight_repo['sha'],'expected':entry}
if weight.exists() and weight.stat().st_size==entry['size']:
    digest=hashlib.sha256(weight.read_bytes()).hexdigest()
    assert digest==entry['lfs']['sha256']
    with zipfile.ZipFile(weight) as z:
        assert z.testzip() is None
        members=len(z.infolist())
    weight_result.update(status='PASS_PUBLISHED_LFS_SHA256_SIZE_ZIP_CRC',sha256=digest,
                         bytes=weight.stat().st_size,archive_members=members)
receipt={'utc':datetime.now(timezone.utc).isoformat(),'source_status':'ALL293_GIT_BLOBS_VERIFIED',
         'source_commit':commit,'source_zip_sha256':hashlib.sha256((assets/'author_source.zip').read_bytes()).hexdigest(),
         'source_files':verified,'weight':weight_result,
         'execution':'No author modules imported, no pickle deserialization, no detector construction/forward/training. Bundled compiled binaries were copied as inert files only.'}
(assets/'asset_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k!='source_files'},ensure_ascii=False,indent=2))
