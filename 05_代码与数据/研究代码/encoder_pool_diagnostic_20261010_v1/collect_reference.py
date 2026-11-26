"""Collect independent remote SciPy certificates without changing capture archives."""
from pathlib import Path
import base64,hashlib,json,tarfile

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/encoder_pool_diagnostic_20261010_v1'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    desc=read(OUT/'reference_transfer_descriptors.json')
    checks=[]
    for seed in (1,2):
        d=desc[f'seed{seed}'];assert d['next']==d['bytes']
        archive=OUT/f'analysis_reference_seed{seed}.tar.gz'
        with archive.open('xb') as f:
            position=0
            for part in d['parts']:
                assert part['start']==position
                s=(OUT/f'reference_seed{seed}_parts'/part['name']).read_text().strip()
                assert len(s)==part['encoded_length']
                b=base64.b64decode(s,validate=True);assert len(b)==part['count']
                f.write(b);position+=len(b)
        assert position==d['bytes'] and sha(archive)==d['sha256']
        target=OUT/f'analysis_reference_seed{seed}';target.mkdir(exist_ok=False)
        with tarfile.open(archive) as tf:tf.extractall(target,filter='data')
        assert sha(target/'code/analyze_pool.py')==sha(Path(__file__).with_name('analyze_pool.py'))
        assert (target/'matching_clarification.json').read_bytes()==(OUT/'matching_clarification.json').read_bytes()
        receipt=read(target/'analysis_completion.json');assert receipt['analysis_exit_code']==0
        summary=read(target/'analysis_scipy_reference/summary.json')
        cert=read(target/'analysis_scipy_reference/matching_certificates.json')
        assert summary['scipy_independent_optimality_checked'] and summary['certificates']==1920
        assert cert['status']=='PASS_INDEPENDENT_SCIPY_DUAL_OPTIMUM_CERTIFICATES' and len(cert['certificates'])==1920
        checks.append({'seed':seed,'archive_bytes':position,'archive_sha256':d['sha256'],'files':{str(p.relative_to(target)):sha(p) for p in sorted(target.rglob('*')) if p.is_file()},'receipt':receipt})
    (OUT/'reference_download_verification.json').write_text(json.dumps({'status':'PASS_REMOTE_REFERENCE_ARCHIVES_CODE_MATCHING_DEFINITION_AND_EXIT0','checks':checks},indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':'PASS_REMOTE_REFERENCE_COLLECTION','seeds':[1,2],'certificates':3840}))

if __name__=='__main__':main()
