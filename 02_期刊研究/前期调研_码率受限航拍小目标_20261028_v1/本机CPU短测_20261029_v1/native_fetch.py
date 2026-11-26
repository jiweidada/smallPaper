"""Fetch pip's resolved, hashed public wheels with bounded Windows curl transfers."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, unquote
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument("--label",choices=["support","cpu"],required=True)
a=parser.parse_args()
plan=json.loads((root/(a.label+"_plan.json")).read_text(encoding="utf-8"))
normalize=lambda name: re.sub(r"[-_.]+","-",name).lower()
excluded=set()
if a.label=="cpu":
    excluded={normalize(p["metadata"]["name"]) for p in json.loads((root/"support_plan.json").read_text(encoding="utf-8"))["install"]}
items=[p for p in plan["install"] if normalize(p["metadata"]["name"]) not in excluded]
folder=root/("wheelhouse_"+a.label)
folder.mkdir(exist_ok=True)
(root/(a.label+"_offline_lock.txt")).write_text("\n".join(p["metadata"]["name"]+"=="+p["metadata"]["version"] for p in items)+"\n",encoding="utf-8")

def fetch(item):
    started=time.monotonic()
    url=item["download_info"]["url"]
    parsed=urlparse(url)
    assert parsed.scheme=="https" and parsed.hostname in {"files.pythonhosted.org","download.pytorch.org","download-r2.pytorch.org"}
    filename=unquote(Path(parsed.path).name)
    assert filename.endswith(".whl") and Path(filename).name==filename
    expected=item["download_info"]["archive_info"]["hashes"]["sha256"]
    output=folder/filename
    record={"name":item["metadata"]["name"],"version":item["metadata"]["version"],"url":url,
            "expected_sha256":expected,"file":str(output)}
    # The large Torch wheel has a separately bounded resumable transfer already in progress.
    if normalize(record["name"])=="torch":
        record.update(status="SEPARATE_TRANSFER",file=str(root/"wheels"/filename))
        return record
    part=folder/(filename+".part")
    if not output.exists() or hashlib.file_digest(output.open("rb"),"sha256").hexdigest()!=expected:
        command=["curl.exe","-L","--fail","--retry","0","--connect-timeout","15","--max-time","90",
                 "--silent","--show-error","--output",str(part),url]
        completed=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=100)
        if completed.returncode:
            record.update(status="FAILED",exit_code=completed.returncode,detail=completed.stdout.decode(errors="replace")[-500:])
            return record
        actual=hashlib.file_digest(part.open("rb"),"sha256").hexdigest()
        if actual!=expected:
            record.update(status="HASH_MISMATCH",actual_sha256=actual)
            return record
        part.replace(output)
    record.update(status="VERIFIED",bytes=output.stat().st_size,seconds=round(time.monotonic()-started,3))
    return record

results=[]
with ThreadPoolExecutor(max_workers=4) as executor:
    futures={executor.submit(fetch,p):p for p in items}
    for future in as_completed(futures):
        try:
            r=future.result()
        except Exception as error:
            r={"name":futures[future]["metadata"]["name"],"status":"EXCEPTION","detail":repr(error)}
        results.append(r)
        print(json.dumps({k:r[k] for k in ("name","status","bytes","seconds","detail") if k in r}),flush=True)
record={"label":a.label,"records":results,"failed":sum(r["status"] not in {"VERIFIED","SEPARATE_TRANSFER"} for r in results),
        "mode":"4 native curl workers, 90 seconds per transfer, SHA256 checked against pip report; no automatic retries"}
(root/("native_fetch_"+a.label+".json")).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
sys.exit(1 if record["failed"] else 0)
