"""Fetch observed primary subset-optimization neighbours after H1 falsification."""
import argparse
import datetime
import hashlib
import json
import urllib.request
from pathlib import Path

RUN = Path(__file__).resolve().parent
URLS = {
    "lddp": "https://openaccess.thecvf.com/content_cvpr_2017/papers/Azadi_Learning_Detection_With_CVPR_2017_paper.pdf",
    "qubo_pedestrian": "https://openaccess.thecvf.com/content_cvpr_2013/papers/Rujikietgumjorn_Optimized_Pedestrian_Detection_2013_CVPR_paper.pdf",
    "matroid_notes": "https://math.mit.edu/~goemans/18433S09/matroid-notes.pdf"
}
parser = argparse.ArgumentParser()
parser.add_argument("slug", choices=URLS)
args = parser.parse_args()
out = RUN / "sources" / (args.slug + ".pdf")
assert not out.exists()
record = {"slug": args.slug, "url": URLS[args.slug], "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
try:
    req = urllib.request.Request(URLS[args.slug], headers={"User-Agent": "Academic-source-verification/1.0"})
    with urllib.request.urlopen(req, timeout=45) as response:
        data = response.read(30 * 1024 * 1024)
        record.update({"status_code": response.status, "final_url": response.url, "content_type": response.headers.get("Content-Type")})
    assert data.startswith(b"%PDF-")
    with out.open("xb") as stream:
        stream.write(data)
    record.update({"status": "PASS", "path": str(out), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
except Exception as error:
    record.update({"status": "FAIL", "error": repr(error)})
with (RUN / (args.slug + "_download.json")).open("x", encoding="utf-8") as stream:
    json.dump(record, stream, ensure_ascii=False, indent=2)
print(json.dumps(record, ensure_ascii=True))
if record["status"] != "PASS":
    raise SystemExit(1)
