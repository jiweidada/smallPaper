"""Read-only local download status, including received but not yet committed HTTP ranges."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent;DATA=HERE.parents[1]/'data/07_第二数据集_20261017'
ROOT=DATA/'原始下载/UAVDT';WORK=ROOT/'http_ranges_v1'
state=json.loads((WORK/'download_state.json').read_text()) if (WORK/'download_state.json').is_file() else {}
def size_or_zero(path):
    try:return path.stat().st_size
    except FileNotFoundError:return 0
received=0;committed=0;inflight=0
indices={p.name.split('.')[0] for p in WORK.glob('*.piece')}|{p.name.split('.')[0] for p in WORK.glob('*.partial')}|{p.name.split('.')[0] for p in WORK.glob('*.body')}
for index in indices:
    final=WORK/(index+'.piece');partial=WORK/(index+'.partial');base=int(index)*4*1024**2
    if final.exists():size=size_or_zero(final);committed+=size;received+=size;continue
    have=size_or_zero(partial);pending=0
    for body in WORK.glob(index+'.attempt*.body'):
        header=body.with_suffix('.headers')
        if not header.is_file():continue
        found=re.findall(r'^Content-Range:\s*bytes (\d+)-(\d+)/(\d+)',header.read_text(errors='replace'),re.I|re.M)
        if not found:continue
        start,end,total=map(int,found[-1])
        if total!=6790452113 or start<base or end>=min(6790452113,base+4*1024**2):continue
        pending=max(pending,max(0,start-base+min(size_or_zero(body),end-start+1)-have))
    committed+=have;received+=have+pending;inflight+=pending
pid=int(state.get('pid',0));identity={}
if pid:
    expected='英文小论文*code*phase_second_dataset_20261017_v1*download_uavdt_parallel.py'
    script=f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; if ($p) {{ @{{alive=$true; identity_matches=($p.CommandLine -like '*{expected}*'); creation_date=$p.CreationDate.ToString('o')}} | ConvertTo-Json -Compress }} else {{ @{{alive=$false;identity_matches=$false}} | ConvertTo-Json -Compress }}"
    result=subprocess.run(['powershell.exe','-NoProfile','-Command',script],capture_output=True,text=True,timeout=15)
    if result.returncode==0 and result.stdout.strip():identity=json.loads(result.stdout)
now=datetime.datetime.now(datetime.timezone.utc)
selection=json.loads((DATA/'第二数据集选择状态.json').read_text(encoding='utf-8'))
report={'checked_utc':now.isoformat(),'checked_los_angeles':now.astimezone(ZoneInfo('America/Los_Angeles')).isoformat(),
        'controller':identity,'download_state':state,'bytes_received_with_inflight':received,'committed_validated_range_bytes':committed,
        'inflight_valid_range_bytes':inflight,'bytes_total':6790452113,'percent_received':round(received/6790452113*100,3),
        'complete_archive_exists':(ROOT/'UAV-benchmark-M.zip').is_file(),'final_receipt_exists':(ROOT/'UAV-benchmark-M.zip.receipt.json').is_file(),
        'selected_dataset':selection.get('selected_dataset'),'new_training_held':not selection.get('allow_automatic_training_start',False)}
dest=DATA/'UAVDT完整下载跟进';dest.mkdir(exist_ok=True)
raw=json.dumps(report,ensure_ascii=False,indent=2).encode();name=dest/(now.strftime('%Y%m%dT%H%M%S_%fZ')+'.json')
with name.open('xb') as f:f.write(raw)
(dest/'latest_snapshot.json').write_bytes(raw)
print(json.dumps(dict(report,snapshot=str(name),sha256=hashlib.sha256(raw).hexdigest()),ensure_ascii=False,indent=2))
