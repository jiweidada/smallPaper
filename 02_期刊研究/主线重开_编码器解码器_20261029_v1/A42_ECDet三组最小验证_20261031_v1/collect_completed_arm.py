"""Read-only collection of a terminal arm; never launches/resumes training."""
import argparse,hashlib,json,os,pathlib,zipfile
R=pathlib.Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        for v in iter(lambda:f.read(1<<20),b''):h.update(v)
    return h.hexdigest()

def main(arm):
    directory=R/('arm_'+arm)
    if not (directory/'exit_receipt.json').exists():
        print(json.dumps({'status':'NOT_TERMINAL','arm':arm,'training_action':'none'}));return
    result=read(directory/'result.json');exit_receipt=read(directory/'exit_receipt.json')
    assert exit_receipt['exit_code']==0 and result['status']=='COMPLETED' and result['epochs']==12
    assert result['optimizer_updates']==9708
    plan=read(R/'execution_protocol_FROZEN.json')
    for name,h in plan['files_sha256'].items():assert sha(R/name)==h,name
    events=[json.loads(s)for s in (directory/'epochs.jsonl').read_text().splitlines()]
    endpoint=[e for e in events if e['epoch']==12];assert len(endpoint)==1
    assert sha(directory/'epoch12.pth')==endpoint[0]['checkpoint_sha256']
    archive=R/('completed_arm_'+arm+'_bundle.zip');receipt=archive.with_suffix('.receipt.json')
    if archive.exists():
        existing=read(receipt);assert sha(archive)==existing['sha256']
        print(json.dumps(existing));return
    paths=[p for p in directory.iterdir()if p.is_file()and(p.suffix!='.pth'or p.name=='epoch12.pth')]
    paths += [R/n for n in plan['files_sha256']if n!='common_init_10class.pth']
    paths += [R/'execution_protocol_FROZEN.json',R/'supervisor_started.json',R/'strict_official_80class_load.json',R/'common_initialization_and_paths.json',R/'isolated_source_patch_receipt.json']
    temp=archive.with_suffix('.partial.zip');manifest={}
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1)as z:
        for p in sorted(set(paths)):
            relative=str(p.relative_to(R));manifest[relative]=sha(p);z.write(p,relative)
        z.writestr('member_sha256.json',json.dumps(manifest,indent=2))
    os.replace(temp,archive)
    meta={'status':'PACKAGED_TERMINAL_ARM','arm':arm,'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'member_files':len(manifest),'epoch12_checkpoint_sha256':endpoint[0]['checkpoint_sha256'],'common_init_excluded_but_pinned_sha256':plan['common_init_sha256'],'training_action':'none'}
    receipt.write_text(json.dumps(meta,indent=2));print(json.dumps(meta))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--arm',choices=['B','S','L'],required=True);main(a.parse_args().arm)
