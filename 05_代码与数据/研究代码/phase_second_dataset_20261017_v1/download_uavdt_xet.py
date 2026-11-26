"""Official Xet client for the pinned public UAVDT mirror. Anonymous token stays in memory."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'data/07_第二数据集_20261017/原始下载/UAVDT'
os.environ['HF_XET_CACHE']=str(OUT/'xet_cache')
os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES']='0'
os.environ['HF_XET_NUM_CONCURRENT_RANGE_GETS']='8'
os.environ['HF_XET_HIGH_PERFORMANCE']='1'
os.environ['RUST_LOG']='error'
sys.path.insert(0,str(HERE/'runtime_hf_xet_1_7_0'))
from hf_xet import XetSession,XetFileInfo

REV='ad1cfbe161c183cfdee26f6c42c6fff6552a54dc'
url='https://huggingface.co/api/datasets/vanthanh/UAVDT-Benchmark-M/xet-read-token/'+REV
response=subprocess.run(['curl.exe','--fail','--silent','--show-error','--include','--connect-timeout','10','--max-time','45',url],
                        capture_output=True,timeout=50)
if response.returncode:raise RuntimeError('Public Xet connection information unavailable')
text=response.stdout.decode()
def header(name):
    values=re.findall(r'^'+re.escape(name)+r':\s*([^\r\n]+)',text,re.I|re.M)
    if not values:raise RuntimeError('Missing Xet response metadata header '+name)
    return values[-1].strip()
endpoint=header('x-xet-cas-url');token=header('x-xet-access-token');expiry=int(header('x-xet-token-expiration'))
del text,response
request=json.loads((OUT/'UAV-benchmark-M.zip.request.json').read_text())
assert request['revision']==REV and request['bytes']==6790452113
target=OUT/'UAV-benchmark-M.zip';partial=OUT/'UAV-benchmark-M.zip.xet_v3.part'
assert not target.exists() and not partial.exists(),'Do not replace an existing asset or Xet attempt'
started=time.monotonic();last_print=[0];schema=[False]
def progress(*reports):
    report=reports[-1]
    if not schema[0]:
        print(json.dumps({'event':'xet_progress_fields','fields':list(report) if isinstance(report,dict) else [n for n in dir(report) if not n.startswith('_')]}),flush=True);schema[0]=True
    if time.monotonic()-last_print[0]<20:return
    values={}
    for pos,obj in enumerate(reports):
        pairs=obj.items() if isinstance(obj,dict) else [(n,getattr(obj,n)) for n in dir(obj) if not n.startswith('_')]
        for name,value in pairs:
            if isinstance(value,(int,float)):values[str(pos)+'.'+name]=value
    print(json.dumps({'event':'xet_progress','elapsed_seconds':round(time.monotonic()-started,1),'counters':values}),flush=True)
    last_print[0]=time.monotonic()
session=XetSession()
print(json.dumps({'event':'xet_download_started','bytes':request['bytes']}),flush=True)
with session.new_file_download_group(endpoint=endpoint,token=token,token_expiry_unix_secs=expiry,
    token_refresh_url=url,token_refresh_headers={},custom_headers={},progress_callback=progress) as group:
    group.start_download_file(XetFileInfo('0d420a48a8b481ba8375f44b66e6e9848794cb1d808bb9ffc49caba0100c9c61',request['bytes']),str(partial.resolve()))
token=None
assert partial.stat().st_size==request['bytes']
print(json.dumps({'event':'sha_and_full_zip_crc_validation','bytes':request['bytes']}),flush=True)
h=hashlib.sha256()
with partial.open('rb') as f:
    for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
assert h.hexdigest()==request['sha256'],'Downloaded SHA mismatch'
with zipfile.ZipFile(partial) as archive:
    bad=archive.testzip();assert bad is None,'ZIP CRC mismatch: '+str(bad);members=archive.namelist()
partial.rename(target)
receipt=dict(request,path=str(target.resolve()),collected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             zip_crc_verified=True,zip_members=members,download_method='official hf_xet 1.7.0 anonymous read',
             download_elapsed_seconds=time.monotonic()-started)
with (OUT/'UAV-benchmark-M.zip.receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print(json.dumps({'event':'verified_download_complete','bytes':request['bytes'],'sha256':h.hexdigest(),'members':len(members)}),flush=True)
