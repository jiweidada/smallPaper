import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path

OUT = Path(r"C:\Users\jiwei\Desktop\小论文投稿\01_会议论文\实验数据汇总_20261108")
PAPER = OUT.parent
RECOVERY = PAPER / "远程公平对照_20261103/模块消融/NO_MSCA/recovery30715_20261109"
SUMMARY = OUT / "实验汇总.json"
SOURCES = OUT / "输入来源.csv"
AUDIT = OUT / "输入核查.json"
def sha(data): return hashlib.sha256(data).hexdigest()
def atomic(path, data):
    tmp = path.with_name(path.name + ".round320.tmp")
    assert not tmp.exists()
    with tmp.open("xb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

raw = SOURCES.read_bytes()
assert sha(raw) == "14871552e37710d428a413ac6ea4d6b58c1756c9aa91870a842adfa54fba6621"
with SOURCES.open("r",encoding="utf-8-sig",newline="") as f:
    reader=csv.DictReader(f); fields=reader.fieldnames; old_rows=list(reader)
assert len(old_rows)==36 and fields==["source_id","relative_to_remote_root","path","bytes","sha256","status_in_source"]
summary=json.loads(SUMMARY.read_text(encoding="utf-8-sig"))
sources=summary["input_sources"]
assert len(sources)==43
assert [x["source_id"] for x in sources[-7:]] == [
 "NO_MSCA_LINEAGE_R320","NO_MSCA_NATIVE_MONITOR_R320","NO_MSCA_NATIVE_RECEIPT_R320",
 "NO_MSCA_NATIVE_SUPERVISOR_R320","NO_MSCA_NATIVE_EPOCH_ROWS_R320",
 "NO_MSCA_NATIVE_EPOCH_MANIFEST_R320","NO_MSCA_NATIVE_BEST_MANIFEST_R302"
]
for got,expected in zip(old_rows,sources[:36]):
    assert all(str(got[k]) == str(expected[k]) for k in fields), got["source_id"]
prefix="模块消融/NO_MSCA/recovery30715_20261109"
for item in sources[-7:]:
    rel=Path(item["path"]).relative_to(RECOVERY).as_posix()
    item["relative_to_remote_root"]=prefix+"/"+rel
    target=Path(item["path"])
    assert target.is_file() and target.stat().st_size==int(item["bytes"]) and sha(target.read_bytes())==item["sha256"]
assert all(not Path(x["relative_to_remote_root"]).is_absolute() and "C:" not in x["relative_to_remote_root"] for x in sources[-7:])
tail=io.StringIO(newline="")
writer=csv.DictWriter(tail,fieldnames=fields,extrasaction="raise",lineterminator="\r\n")
writer.writerows(sources[-7:])
appended=tail.getvalue().encode("utf-8")
atomic(SOURCES,raw+appended)
atomic(SUMMARY,(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode("utf-8"))
report=json.loads(AUDIT.read_text(encoding="utf-8-sig"))
report.setdefault("checks",[]).append({
 "check":"Input source CSV contains all 43 summary sources, including seven round-320 records with project-relative paths and matching bytes/SHA-256.",
 "status":"PASS"
})
report["check_count"]=len(report["checks"])
report["input_sources_csv_sha256"]=sha(SOURCES.read_bytes())
report["input_sources_csv_rows"]=43
atomic(AUDIT,(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode("utf-8"))
# Read back and verify all source rows match the updated JSON.
with SOURCES.open("r",encoding="utf-8-sig",newline="") as f:
    rows=list(csv.DictReader(f))
assert len(rows)==43
for got,expected in zip(rows,sources):
    assert all(str(got[k])==str(expected[k]) for k in fields), got["source_id"]
print(json.dumps({
 "status":"ROUND320_SOURCE_INDEX_COMPLETED",
 "rows_before":36,"rows_after":43,"new_sources":[x["source_id"] for x in sources[-7:]],
 "old_csv_exact_byte_prefix_bytes":len(raw),
 "input_sources_csv_sha256":sha(SOURCES.read_bytes()),
 "summary_sha256":sha(SUMMARY.read_bytes()),
 "input_audit_sha256":sha(AUDIT.read_bytes())
},ensure_ascii=False,indent=2))
