from pathlib import Path,PurePosixPath
import requests,hashlib,json,zipfile,io
ROOT=Path(__file__).resolve().parent
LIB=ROOT.parent.parent/'文献_相关原始论文_20261004'
repo='VisDrone/VisDrone2018-DET-toolkit'
r=requests.get('https://api.github.com/repos/'+repo+'/commits/master',timeout=(12,30));r.raise_for_status()
info=r.json();commit=info['sha']
url='https://codeload.github.com/'+repo+'/zip/'+commit
r=requests.get(url,timeout=(12,35));r.raise_for_status()
dest=LIB/'_implementation_sources'/('A51_VisDroneOfficialEval_'+commit[:12])
dest.mkdir(exist_ok=False)
archive=dest/'official_source.zip';archive.write_bytes(r.content)
entries={}
with zipfile.ZipFile(io.BytesIO(r.content)) as z:
    assert sum(x.file_size for x in z.infolist())<20*1024**2
    for item in z.infolist():
        if item.is_dir():continue
        rel=PurePosixPath(item.filename)
        assert not rel.is_absolute() and '..' not in rel.parts
        p=(dest/Path(*rel.parts)).resolve();assert p.is_relative_to(dest.resolve())
        p.parent.mkdir(exist_ok=True,parents=True)
        data=z.read(item);p.write_bytes(data)
        entries[str(p)]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
record={'repo':'https://github.com/'+repo,'commit':commit,'commit_url':'https://github.com/'+repo+'/commit/'+commit,
 'archive_url':url,'archive':str(archive),'archive_sha256':hashlib.sha256(r.content).hexdigest(),'files':entries,
 'scope':'archived only; selected evaluator source inspection follows; no evaluator/model executed',
 'status':'SOURCE_ACQUIRED_NOT_EXECUTED'}
(ROOT/'official_eval_source_acquisition_v1.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
(dest/'source.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'commit':commit,'files':len(entries),'directory':str(dest),'scope':record['scope']}))
