from pathlib import Path
import hashlib,json,zipfile
OUT=Path(__file__).resolve().parent
PAPER=OUT.parent
proposal=json.loads((PAPER/'公平对照方案_20261103/protocol.json').read_text(encoding='utf-8'))
archive=OUT/'verified_dataset.zip'
if archive.exists():raise SystemExit('Existing archive retained; inspect before replacement.')
records=[]
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
    for split in ['train','val']:
        path=Path(proposal['data'][split]['current_image_manifest'])
        rows=json.loads(path.read_text(encoding='utf-8'))
        if isinstance(rows,dict):rows=rows['images']
        annotation=Path(proposal['data'][split]['annotation_path'])
        files=[(annotation,f'{split}/annotations/{split}.json',proposal['data'][split]['annotation_sha256'])]
        files += [(annotation.parent.parent/'images'/r['file_name'],f'{split}/images/'+r['file_name'],r['sha256']) for r in rows]
        for i,(source,name,expected) in enumerate(files):
            h=hashlib.sha256();size=0
            with source.open('rb') as src,z.open(name,'w',force_zip64=True) as dest:
                while chunk:=src.read(4*1024*1024):h.update(chunk);size+=len(chunk);dest.write(chunk)
            assert h.hexdigest()==expected,source
            records.append({'path':name,'bytes':size,'sha256':expected})
            if i%1000==0:print(f'{split}: {i}/{len(rows)} images packaged',flush=True)
h=hashlib.sha256()
with archive.open('rb') as f:
    while b:=f.read(8*1024*1024):h.update(b)
manifest={'zip_sha256':h.hexdigest(),'bytes':archive.stat().st_size,'files':records}
(OUT/'dataset_bundle_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'BUNDLED_FROM_VERIFIED_LOCAL_BYTES','bytes':archive.stat().st_size,'files':len(records),'sha256':h.hexdigest()}),flush=True)
