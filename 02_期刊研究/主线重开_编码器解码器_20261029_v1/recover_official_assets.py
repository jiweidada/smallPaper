"""Recover already downloaded public assets and use the configured Windows proxy.

No browser state, private data, system setting changes, package installation or
model execution. Network requests are bounded reads of the official source ZIP.
"""
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import ProxyHandler, HTTPSHandler, Request, build_opener
import hashlib
import json
import os
import shutil
import ssl
import time
import winreg
import zipfile

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/'official_assets_v1'
ASSETS.mkdir(exist_ok=True)
COMMIT='9075d9fdf5ce989ca3c7e830a780dfc7fbbe4ec1'
records={'recorded_utc':datetime.now(timezone.utc).isoformat(),'model_executed':False}

weight_source=Path(r'C:\Users\jiwei\Downloads\TinyFormer-S-pbm-visdrone.pth')
weight_target=ASSETS/weight_source.name
if weight_source.is_file():
    before=weight_source.stat()
    blob_hash=hashlib.sha256()
    with weight_source.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            blob_hash.update(chunk)
    after=weight_source.stat()
    assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
    with zipfile.ZipFile(weight_source) as archive:
        entries=archive.infolist()
        assert sum(e.file_size for e in entries)<1_000_000_000
        assert archive.testzip() is None, 'Incomplete/corrupt torch ZIP'
        pickle_entries=[e for e in entries if e.filename.endswith('/data.pkl')]
        assert len(pickle_entries)==1 and pickle_entries[0].file_size<16_000_000
    shutil.copyfile(weight_source,weight_target)
    assert hashlib.sha256(weight_target.read_bytes()).hexdigest()==blob_hash.hexdigest()
    records['weight']={'status':'LOCAL_COPY_AND_ZIP_CRC_PASS','source':str(weight_source),'path':str(weight_target),
        'bytes':before.st_size,'sha256':blob_hash.hexdigest(),'zip_entries':len(entries),
        'source_page':'https://drive.google.com/file/d/1Sb3jJiCp3vXJBLBcphnP0Gx803Q5vKfl/view',
        'provenance':'Google Drive folder linked in pinned official TinyFormer README',
        'author_published_checksum_available':False,'torch_load_executed':False}
    print(json.dumps({'weight':records['weight']},ensure_ascii=False),flush=True)
else:
    records['weight']={'status':'NOT_FOUND'}

proxy_enabled=False
proxy_value=''
try:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER,r'Software\Microsoft\Windows\CurrentVersion\Internet Settings') as key:
        proxy_enabled=bool(winreg.QueryValueEx(key,'ProxyEnable')[0])
        if proxy_enabled:
            proxy_value=str(winreg.QueryValueEx(key,'ProxyServer')[0])
except FileNotFoundError:
    pass
records['network_configuration']={'system_proxy_enabled':proxy_enabled,'system_proxy_value_present':bool(proxy_value),
    'environment_proxy_variable_names':[k for k in os.environ if k.lower() in ('http_proxy','https_proxy','all_proxy')],
    'proxy_credentials_or_raw_value_recorded':False,'system_settings_modified':False}
print(json.dumps({'network_configuration':records['network_configuration']}),flush=True)
proxy_map={}
if proxy_enabled and proxy_value:
    if '=' in proxy_value:
        for pair in proxy_value.split(';'):
            if '=' not in pair:continue
            scheme,endpoint=pair.split('=',1)
            if scheme in ('http','https'):
                proxy_map[scheme]=endpoint if '://' in endpoint else 'http://'+endpoint
    else:
        endpoint=proxy_value if '://' in proxy_value else 'http://'+proxy_value
        if endpoint.startswith(('http://','https://')):
            proxy_map={'http':endpoint,'https':endpoint}

source_target=ASSETS/('TinyFormer-'+COMMIT+'.zip')
local_zip=Path(r'C:\Users\jiwei\Downloads')/('TinyFormer-'+COMMIT+'.zip')
if local_zip.is_file():
    shutil.copyfile(local_zip,source_target)
    records['source_download']={'status':'COPIED_EXISTING_BROWSER_DOWNLOAD','path':str(source_target)}
elif proxy_map:
    url='https://codeload.github.com/mmpmmpmmpjosh/TinyFormer/zip/'+COMMIT
    started=time.monotonic()
    partial=source_target.with_suffix('.zip.part')
    try:
        opener=build_opener(ProxyHandler(proxy_map),HTTPSHandler(context=ssl.create_default_context()))
        request=Request(url,headers={'User-Agent':'AcademicArchitectureAudit/1.0'})
        with opener.open(request,timeout=12) as response,partial.open('wb') as stream:
            total=0
            while True:
                if time.monotonic()-started>35:
                    raise TimeoutError('35-second source acquisition budget reached')
                chunk=response.read(256*1024)
                if not chunk:break
                stream.write(chunk)
                total+=len(chunk)
                if total>80_000_000:raise ValueError('source archive exceeds 80 MB bound')
        with zipfile.ZipFile(partial) as archive:
            assert archive.testzip() is None
        shutil.copyfile(partial,source_target)
        records['source_download']={'status':'DOWNLOADED_VIA_EXISTING_SYSTEM_PROXY','path':str(source_target),
            'url':url,'bytes':source_target.stat().st_size,'sha256':hashlib.sha256(source_target.read_bytes()).hexdigest(),
            'elapsed_seconds':time.monotonic()-started}
    except Exception as exc:
        records['source_download']={'status':'FAILED_BOUNDED_ATTEMPT','error':type(exc).__name__+': '+str(exc),
            'elapsed_seconds':time.monotonic()-started}
else:
    records['source_download']={'status':'NO_EXISTING_PROXY_OR_SOURCE_ARCHIVE_FOUND','no_additional_direct_retry':True}
(ASSETS/'recovery_receipt.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'source_download':records['source_download']},ensure_ascii=False),flush=True)
