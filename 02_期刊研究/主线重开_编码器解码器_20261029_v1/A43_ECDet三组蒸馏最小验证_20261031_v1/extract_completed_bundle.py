"""Verify an immutable completed-arm archive and extract inside this workspace."""
import argparse,hashlib,json,pathlib,zipfile
D=pathlib.Path(__file__).resolve().parent
LOG=D.parent.parent/'前期调研_码率受限航拍小目标_20261028_v1/远端验证_20261029_v1/ssh_operation_receipts.jsonl'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main(arm,receipt_id):
    found=[]
    with LOG.open(encoding='utf-8')as f:
        for line in f:
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if r.get('id')==receipt_id and r.get('action')=='exec':found.append(r)
    assert len(found)==1 and found[0]['exit_code']==0
    meta=json.loads(found[0]['stdout']);assert meta['status']=='PACKAGED_TERMINAL_ARM'and meta['arm']==arm
    archive=D/('completed_arm_'+arm+'_bundle.zip')
    assert archive.stat().st_size==meta['bytes']and sha(archive)==meta['sha256']
    dest=D/'completed_results'/arm;dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive)as z:
        manifest=json.loads(z.read('member_sha256.json'))
        assert len(manifest)==meta['member_files']
        for n,h in manifest.items():
            rel=pathlib.PurePosixPath(n);assert not rel.is_absolute()and '..'not in rel.parts
            target=dest/pathlib.Path(*rel.parts);target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert sha(target)==h,n;continue
            tmp=target.with_suffix(target.suffix+'.partial');digest=hashlib.sha256()
            with z.open(n)as source,tmp.open('wb')as output:
                for chunk in iter(lambda:source.read(1<<20),b''):digest.update(chunk);output.write(chunk)
            assert digest.hexdigest()==h,n
            tmp.replace(target)
    assert sha(dest/('arm_'+arm)/'epoch12.pth')==meta['epoch12_checkpoint_sha256']
    record={'status':'PASS','arm':arm,'metadata':meta,'member_files_verified':len(manifest),'extracted_to':str(dest),'original_ssh_receipt':found[0]}
    (D/('completed_arm_'+arm+'_download_verification.json')).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:record[k]for k in ['status','arm','member_files_verified','extracted_to']}))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--arm',choices=['B','M','Q'],required=True);a.add_argument('--receipt-id',required=True);v=a.parse_args();main(v.arm,v.receipt_id)
