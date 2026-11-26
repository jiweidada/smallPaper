from pathlib import Path
import hashlib
import json

root=Path(__file__).resolve().parent/'source_audit_v1'
payload=json.loads((root/'browser_source_snapshots.json').read_text(encoding='utf-8'))
target=root/('TinyFormer_'+payload['commit'][:7])
receipts=[]
for item in payload['snapshots']:
    core=item['retrieved_text'].rstrip('\n')
    matched=None
    for newline in ('\n','\r\n'):
        for suffix_count in range(4):
            blob=(core.replace('\n',newline)+newline*suffix_count).encode('utf-8')
            gitsha=hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()
            if gitsha==item['expected_git_blob_sha']:
                matched=blob
                break
        if matched is not None:
            break
    record={'path':item['path'],'expected_git_blob_sha':item['expected_git_blob_sha'],'status':'PASS' if matched is not None else 'FAILED_HASH_MISMATCH'}
    if matched is not None:
        path=target/item['path']
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(matched)
        record.update(bytes=len(matched),sha256=hashlib.sha256(matched).hexdigest(),source_url=item['source_url'])
    receipts.append(record)
(root/'browser_source_verification.json').write_text(json.dumps({'commit':payload['commit'],'files':receipts},indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipts))
assert all(r['status']=='PASS' for r in receipts), 'Unverified text is not treated as official source'
