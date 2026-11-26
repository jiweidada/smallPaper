"""Resumable, bounded public HTTP range download. Verify every range and whole-file SHA/CRC."""
import concurrent.futures
import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import zipfile

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'data/07_第二数据集_20261017/原始下载/UAVDT'
WORK=OUT/'http_ranges_v1';WORK.mkdir(exist_ok=True)
REQUEST=json.loads((OUT/'UAV-benchmark-M.zip.request.json').read_text())
assert REQUEST['revision']=='ad1cfbe161c183cfdee26f6c42c6fff6552a54dc'
SIZE=6790452113;CHUNK=4*1024**2;WORKERS=32
assert REQUEST['bytes']==SIZE and REQUEST['sha256']=='e82675c938e4e6bd65b4533dba2fb6a028e3e60bd3577cc2f3031504bd585b08'
STOP=threading.Event();START=time.monotonic();errors=[]

def size_or_zero(path):
    try:return path.stat().st_size
    except FileNotFoundError:return 0

def write_status(state,extra=None):
    parts=list(WORK.glob('*.piece'));partial=list(WORK.glob('*.partial'))
    completed=sum(size_or_zero(p) for p in parts);received=completed+sum(size_or_zero(p) for p in partial)
    elapsed=time.monotonic()-START
    data={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'state':state,
          'bytes_total':SIZE,'bytes_received':received,'verified_range_bytes':completed,'pieces_complete':len(parts),
          'pieces_total':(SIZE+CHUNK-1)//CHUNK,'percent':round(received/SIZE*100,3),
          'elapsed_seconds':round(elapsed,1),'MiB_per_second':round(received/max(1,elapsed)/1024**2,3),
          'workers':WORKERS,'errors':list(errors)}
    if extra:data.update(extra)
    temporary=WORK/'download_state.tmp';temporary.write_text(json.dumps(data,indent=2));os.replace(temporary,WORK/'download_state.json')
    print(json.dumps(data),flush=True)

def download_piece(index):
    begin=index*CHUNK;end=min(SIZE,begin+CHUNK)-1;expected=end-begin+1
    final=WORK/f'{index:05}.piece';partial=WORK/f'{index:05}.partial'
    if final.exists():assert final.stat().st_size==expected;return
    no_progress=0;attempt=0
    while not STOP.is_set():
        have=partial.stat().st_size if partial.exists() else 0
        if have==expected:partial.rename(final);return
        assert have<expected
        attempt+=1;body=WORK/f'{index:05}.attempt{attempt}.body';headers=WORK/f'{index:05}.attempt{attempt}.headers'
        # Unique attempt files avoid trusting an interrupted unvalidated response on resume.
        while body.exists() or headers.exists():
            attempt+=1;body=WORK/f'{index:05}.attempt{attempt}.body';headers=WORK/f'{index:05}.attempt{attempt}.headers'
        command=['curl.exe','--fail','--silent','--show-error','--location','--connect-timeout','10','--max-time','180',
                 '--speed-limit','1024','--speed-time','60','--range',f'{begin+have}-{end}',
                 '--max-filesize',str(expected-have),'--dump-header',str(headers),'--output',str(body),REQUEST['url']]
        result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=190)
        raw=headers.read_text(errors='replace') if headers.exists() else ''
        statuses=re.findall(r'^HTTP/\S+\s+(\d+)',raw,re.I|re.M)
        if statuses and int(statuses[-1]) in [401,403,429]:
            raise RuntimeError(f'Public download returned HTTP {statuses[-1]}; stop without authentication or quota workarounds')
        ranges=re.findall(r'^Content-Range:\s*bytes (\d+)-(\d+)/(\d+)',raw,re.I|re.M)
        good=bool(ranges and tuple(map(int,ranges[-1]))==(begin+have,end,SIZE))
        count=body.stat().st_size if body.exists() else 0
        if good and 0<count<=expected-have:
            with partial.open('ab') as target,body.open('rb') as source:
                for block in iter(lambda:source.read(1024**2),b''):target.write(block)
            no_progress=0
        else:
            no_progress+=1
        if no_progress>=4:
            raise RuntimeError(f'Range {index} has four attempts without validated progress (curl {result.returncode})')
        if (WORK/'pause.request').exists():STOP.set();return
    return

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume-verified',action='store_true');args=parser.parse_args()
    lock=WORK/'controller.lock'
    if lock.exists():
        previous=json.loads(lock.read_text());assert previous['request']==REQUEST
        if not args.resume_verified:raise FileExistsError('Existing download controller lock; explicit verified resume required')
        kernel=ctypes.windll.kernel32;kernel.OpenProcess.restype=ctypes.c_void_p
        process=kernel.OpenProcess(0x00100000,False,int(previous['pid']))
        if process:
            active=(kernel.WaitForSingleObject(ctypes.c_void_p(process),0)==258);kernel.CloseHandle(ctypes.c_void_p(process))
            if active:raise RuntimeError('Previous download PID is still alive; refusing a duplicate controller')
        elif kernel.GetLastError()!=87:raise RuntimeError('Cannot verify previous download PID exit')
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        lock.rename(WORK/('controller.lock.previous_'+stamp+'.json'))
        pause=WORK/'pause.request'
        if pause.exists():pause.rename(WORK/('pause.request.consumed_'+stamp))
    with lock.open('x') as f:json.dump({'pid':os.getpid(),'request':REQUEST,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},f,indent=2)
    write_status('downloading')
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures=[pool.submit(download_piece,i) for i in range((SIZE+CHUNK-1)//CHUNK)]
        while True:
            done,pending=concurrent.futures.wait(futures,timeout=20,return_when=concurrent.futures.FIRST_EXCEPTION)
            failures=[f.exception() for f in done if f.exception() is not None]
            if failures:
                errors.extend(str(e) for e in failures[:8]);STOP.set();write_status('failed_preserving_validated_ranges');break
            write_status('downloading')
            if not pending:break
            if (WORK/'pause.request').exists() or time.monotonic()-START>86400:
                STOP.set();write_status('stopped_preserving_validated_ranges');break
    if STOP.is_set():write_status('failed_preserving_validated_ranges' if errors else 'stopped_preserving_validated_ranges');return
    write_status('assembling_and_sha_verification')
    target=OUT/'UAV-benchmark-M.zip';assembled=OUT/'UAV-benchmark-M.zip.parallel.part'
    assert not target.exists() and not assembled.exists(),'Do not replace existing output'
    h=hashlib.sha256();total=0
    with assembled.open('xb') as output:
        for i in range((SIZE+CHUNK-1)//CHUNK):
            with (WORK/f'{i:05}.piece').open('rb') as source:
                for block in iter(lambda:source.read(8*1024**2),b''):output.write(block);h.update(block);total+=len(block)
    assert total==SIZE and h.hexdigest()==REQUEST['sha256'],'Full archive integrity check failed'
    write_status('full_zip_crc_verification')
    with zipfile.ZipFile(assembled) as archive:
        bad=archive.testzip();assert bad is None,'ZIP CRC failure: '+str(bad);members=archive.namelist()
    assembled.rename(target)
    receipt=dict(REQUEST,path=str(target.resolve()),zip_crc_verified=True,zip_members=members,
                 download_method='Bounded parallel public HTTP byte ranges, every Content-Range verified',
                 collected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),download_elapsed_seconds=time.monotonic()-START)
    with (OUT/'UAV-benchmark-M.zip.receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
    write_status('complete',{'archive_sha256':h.hexdigest(),'archive_path':str(target.resolve())})

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        errors.append(type(exc).__name__+': '+str(exc));STOP.set();write_status('failed_preserving_validated_ranges');raise
