import hashlib, json, pathlib, tarfile, urllib.request

D=pathlib.Path(__file__).resolve().parent
PIN='d8752b75c9beaecfcdd0e5cd207bae3b6946ae23'
URL='https://codeload.github.com/Intellindust-AI-Lab/EdgeCrafter/tar.gz/'+PIN
P=D/'edgecrafter_pinned_source.tar.gz'
if not P.exists():
    with urllib.request.urlopen(URL, timeout=35) as r:data=r.read(100_000_001)
    assert len(data)<=100_000_000
    P.write_bytes(data)
dest=D/'author_source'
dest.mkdir(exist_ok=True)
count=0
with tarfile.open(P,'r:gz') as tf:
    for member in tf.getmembers():
        path=pathlib.Path(member.name)
        assert not path.is_absolute() and '..' not in path.parts
        if member.isdir():continue
        assert member.isfile(), 'Unexpected non-file archive entry'
        target=dest/path
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            with tf.extractfile(member) as src:target.write_bytes(src.read())
        count+=1
receipt={'commit':PIN,'url':URL,'local':str(P),'bytes':P.stat().st_size,'sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'members':count,'extracted_under':str(dest),'imported':False}
(D/'source_acquisition_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt))
