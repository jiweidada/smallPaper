from pathlib import Path
import argparse,base64,hashlib,json,tarfile
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/encoder_pool_diagnostic_20261010_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();desc=next(d for d in read(OUT/'transfer_descriptors.json') if d['seed']==a.seed);archive=OUT/f'collection_seed{a.seed}.tar.gz'
    if desc.get('parts'):
        assert desc['next']==desc['bytes']
        with archive.open('xb') as f:
            for part in desc['parts']:
                encoded=(OUT/f'transfer_seed{a.seed}_parts'/part['name']).read_text().strip();assert len(encoded)==part['encoded_length'];decoded=base64.b64decode(encoded,validate=True);assert len(decoded)==part['count'];f.write(decoded)
    assert archive.stat().st_size==desc['bytes'] and sha(archive)==desc['sha256']
    target=OUT/f'collection_seed{a.seed}';target.mkdir(exist_ok=False)
    with tarfile.open(archive) as tf:tf.extractall(target,filter='data')
    assert (target/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes();protocol=read(OUT/'protocol.json')
    for name in ['capture_pool.py','controller.py']:assert sha(target/'code'/name)==protocol['code_sha256'][name]
    capture=target/'capture_v1';manifest=read(capture/'manifest.json')['files']
    for name,h in manifest.items():assert sha(capture/name)==h,name
    complete=read(capture/'completion.json');controller=read(target/'controller_completion.json');assert complete['status']=='COMPLETE_ENCODER_POOL_CAPTURE' and complete['model_forward_calls']==46 and complete['recorded_batches']==40 and controller['exit_code']==0
    result={'status':'PASS_EXACT_ARCHIVE_ALL_CAPTURE_FILES_CODE_PROTOCOL_AND_TERMINAL','seed':a.seed,'archive_bytes':desc['bytes'],'archive_sha256':desc['sha256'],'all_capture_files_verified':len(manifest),'completion':complete,'controller':controller}
    (OUT/f'download_verification_seed{a.seed}.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k not in ['completion','controller']}))
if __name__=='__main__':main()
