"""Task-local public dataset download; select only DET/MOT frames, no persistent seeding."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'runtime_libtorrent_2_1_1'))
import libtorrent as lt

OUT=HERE.parents[1]/'data/07_第二数据集_20261017/原始下载/UAVDT'
WORK=OUT/'hyperai_torrent';WORK.mkdir(exist_ok=True)
metadata=lt.torrent_info(str(OUT/'UAVDT_HyperAI.torrent'));files=metadata.files()
selected=[i for i in range(files.num_files()) if files.file_path(i).replace('\\','/').endswith('/UAV-benchmark-M.zip')]
assert len(selected)==1;index=selected[0]
assert files.file_size(index)==6790452113
settings={'listen_interfaces':'127.0.0.1:0','enable_dht':False,'enable_lsd':False,'enable_upnp':False,'enable_natpmp':False,
          'connections_limit':12,'upload_rate_limit':1024,'active_downloads':1,'active_seeds':0,
          'announce_to_all_trackers':True,'announce_to_all_tiers':True}
session=lt.session(settings)
handle=session.add_torrent({'ti':metadata,'save_path':str(WORK.resolve()),'file_priorities':[4 if i==index else 0 for i in range(files.num_files())]})
trackers=list(handle.trackers())
trackers.sort(key=lambda t:0 if t['url']=='https://orion.hyper.ai/tracker/announce' else 1)
handle.replace_trackers(trackers);handle.force_reannounce()
started=time.monotonic();next_print=0
print(json.dumps({'event':'public_torrent_started','selected_file':files.file_path(index),'bytes':files.file_size(index),
                  'no_port_mapping':True,'upload_limit_bytes_per_second':1024}),flush=True)
while True:
    status=handle.status();current=handle.file_progress()[index]
    record={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'state':str(status.state),
            'bytes_completed':current,'bytes_total':6790452113,'percent':round(current/6790452113*100,2),
            'download_bytes_per_second':status.download_rate,'upload_bytes_per_second':status.upload_rate,
            'peers':status.num_peers,'seeds':status.num_seeds,'elapsed_seconds':round(time.monotonic()-started,1)}
    if status.errc.value():record['error']=status.errc.message()
    alerts=session.pop_alerts()
    record['recent_tracker_alerts']=[type(a).__name__ for a in alerts if 'tracker' in type(a).__name__ or 'error' in type(a).__name__][-20:]
    tmp=WORK/'download_state.tmp';tmp.write_text(json.dumps(record,indent=2));os.replace(tmp,WORK/'download_state.json')
    if time.monotonic()>=next_print:
        print(json.dumps(record),flush=True);next_print=time.monotonic()+20
    if record.get('error'):raise RuntimeError(record['error'])
    if current==6790452113 and status.is_finished:break
    if time.monotonic()-started>7200:raise TimeoutError('Two-hour task download limit reached; pieces retained')
    time.sleep(2)
handle.pause();session.pause()
source=WORK/files.file_path(index);expected='e82675c938e4e6bd65b4533dba2fb6a028e3e60bd3577cc2f3031504bd585b08'
print(json.dumps({'event':'sha_and_full_zip_crc_validation','bytes':source.stat().st_size}),flush=True)
h=hashlib.sha256()
with source.open('rb') as f:
    for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
assert h.hexdigest()==expected,'Torrent result differs from pinned original-format mirror'
with zipfile.ZipFile(source) as archive:
    bad=archive.testzip();assert bad is None;members=archive.namelist()
target=OUT/'UAV-benchmark-M.zip';assert not target.exists();source.rename(target)
receipt={'path':str(target.resolve()),'bytes':target.stat().st_size,'sha256':h.hexdigest(),'zip_crc_verified':True,'zip_members':members,
         'download_method':'Public HyperAI torrent, DET/MOT file selected only; SHA matches independent pinned Hugging Face mirror',
         'source_page':'https://hyper.ai/cn/datasets/17869','collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'continuous_seeding':False,'elapsed_seconds':time.monotonic()-started}
with (OUT/'UAV-benchmark-M.zip.receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print(json.dumps({'event':'verified_download_complete','sha256':h.hexdigest(),'members':len(members)}),flush=True)
