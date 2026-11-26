"""Download pinned public original-format UAVDT assets, verify SHA and full ZIP CRC."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

REPO='vanthanh/UAVDT-Benchmark-M'
REV='ad1cfbe161c183cfdee26f6c42c6fff6552a54dc'
AUTHOR_TOOLKIT_SHA='1e565da8c2a035bf1e56b1322de76c74b73702ea3b0f918f45debf69c2b26799'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true');args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    api='https://huggingface.co/api/datasets/'+REPO+'/tree/'+REV+'?expand=true'
    fetched=subprocess.run(['curl.exe','--fail','--silent','--show-error','--location','--connect-timeout','10',
                            '--max-time','45',api],capture_output=True,timeout=50)
    if fetched.returncode:raise RuntimeError(fetched.stderr.decode(errors='replace')[-500:])
    entries=json.loads(fetched.stdout);by_name={e['path']:e for e in entries}
    assert by_name['UAV-benchmark-MOTD_v1.0.zip']['lfs']['oid']==AUTHOR_TOOLKIT_SHA
    for name in ['M_attr.zip','UAV-benchmark-M.zip']:
        meta=by_name[name];expected=meta['lfs']['oid'];size=meta['size'];target=args.out/name
        receipt=target.with_name(name+'.receipt.json');partial=target.with_name(name+'.part')
        request_file=target.with_name(name+'.request.json')
        url='https://huggingface.co/datasets/'+REPO+'/resolve/'+REV+'/'+name+'?download=true'
        if target.exists():
            assert receipt.is_file() and digest(target)==expected,'Unverified existing final asset'
            print(json.dumps({'event':'already_verified','file':name}),flush=True);continue
        request={'repository':REPO,'revision':REV,'url':url,'bytes':size,'sha256':expected,
                 'official_toolkit_matches_author_sha':True}
        if request_file.exists():
            assert json.loads(request_file.read_text())==request,'Previous partial has different source metadata'
        else:
            with request_file.open('x') as f:json.dump(request,f,indent=2)
        if partial.exists() and not args.resume:raise FileExistsError('Explicit --resume required for verified partial '+str(partial))
        command=['curl.exe','--fail','--location','--silent','--show-error','--connect-timeout','10',
                 '--max-time','7200','--speed-limit','1024','--speed-time','90','--output',str(partial)]
        if partial.exists():command+=['--continue-at','-']
        command+=[url]
        started=time.monotonic();initial=partial.stat().st_size if partial.exists() else 0
        error_file=target.with_name(name+'.curl.stderr.log')
        with error_file.open('ab') as errors:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=errors)
            print(json.dumps({'event':'download_started','file':name,'bytes':size,'curl_pid':proc.pid,'initial_bytes':initial}),flush=True)
            next_print=started+20
            while proc.poll() is None:
                time.sleep(1)
                if time.monotonic()>=next_print:
                    current=partial.stat().st_size if partial.exists() else 0;elapsed=time.monotonic()-started
                    print(json.dumps({'event':'progress','file':name,'bytes':current,'percent':round(current/size*100,2),
                                     'MiB_per_second':round((current-initial)/elapsed/1024**2,2)}),flush=True)
                    next_print=time.monotonic()+20
        if proc.returncode:raise RuntimeError('Public mirror download failed, verified partial retained; see '+str(error_file))
        assert partial.stat().st_size==size,'Incorrect downloaded size'
        print(json.dumps({'event':'sha_and_full_zip_crc_validation','file':name,'bytes':size}),flush=True)
        actual=digest(partial);assert actual==expected,'SHA mismatch'
        with zipfile.ZipFile(partial) as archive:
            bad=archive.testzip();assert bad is None,'ZIP CRC mismatch: '+str(bad)
            members=archive.namelist()
        partial.rename(target)
        record=dict(request,path=str(target.resolve()),collected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    zip_crc_verified=True,zip_members=members,download_elapsed_seconds=time.monotonic()-started)
        with receipt.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
        print(json.dumps({'event':'verified_download_complete','file':name,'sha256':actual,'members':len(members)}),flush=True)

if __name__=='__main__':main()
