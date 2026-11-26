"""Archive an immutable completed arm. No model import or training operation."""
from pathlib import Path
import argparse,hashlib,json,zipfile

ROOT=Path('/root/uav_a51_20261101_v1')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['M','Q'],required=True);a=parser.parse_args().arm
    folder=ROOT/('arm_'+a)
    result=json.loads((folder/'result.json').read_text())
    terminal=json.loads((folder/'exit.json').read_text())
    assert result['status']=='COMPLETED' and result['epochs']==12 and result['optimizer_updates']==9708
    assert result['full_model_forwards']==10159 and terminal['exit_code']==0
    assert result['model_parameters']==9876858 and result['extra_parameters']==73746
    protocol=ROOT/'protocol_FROZEN_v1.json'
    assert sha(protocol)==result['protocol_sha256']=='a301ec5fb1f6a5c4675ef129f11e8359fa46c5d49640f636ccbeb343703cb8e8'
    files=sorted(p for p in folder.rglob('*') if p.is_file())+[protocol,ROOT/'conservative_scale_exchange.py',ROOT/'runtime_A51.py',ROOT/'train_A51.py',ROOT/'capacity_result.json',ROOT/'capacity_protocol_FROZEN_v1.json']
    assert all(not p.is_symlink() and p.resolve().is_relative_to(ROOT) for p in files)
    manifest={'arm':a,'status':'COMPLETED_EXIT0','files':{str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files},
              'model_forward_operations_added':0,'optimizer_updates_added':0,'originals_modified':False}
    target=ROOT/(a+'_completed_bundle_v1.zip')
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        z.writestr('bundle_manifest.json',json.dumps(manifest,indent=2))
        for p in files:z.write(p,str(p.relative_to(ROOT)))
    for p in files:assert sha(p)==manifest['files'][str(p.relative_to(ROOT))]['sha256']
    receipt={'status':'BUNDLE_READY','arm':a,'path':str(target),'bytes':target.stat().st_size,'sha256':sha(target),
             'payload_files':len(files),'endpoint':result['endpoint_metrics'],'source_hashes_unchanged':True}
    (ROOT/(a+'_bundle_ready.json')).write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)
