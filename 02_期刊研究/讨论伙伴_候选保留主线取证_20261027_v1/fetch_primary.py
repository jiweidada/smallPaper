"""Fetch a specifically verified primary paper into this evidence directory."""
import argparse
import datetime
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

RUN = Path(__file__).resolve().parent
SOURCES = {
    "rankdetr_publisher": "https://proceedings.neurips.cc/paper_files/paper/2023/file/34074479ee2186a9f236b8fd03635372-Paper-Conference.pdf",
    "learning_nms": "https://openaccess.thecvf.com/content_cvpr_2017/papers/Hosang_Learning_Non-Maximum_Suppression_CVPR_2017_paper.pdf",
    "soft_nms": "https://openaccess.thecvf.com/content_ICCV_2017/papers/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.pdf",
    "dqdetr_v6": "https://arxiv.org/pdf/2404.03507v6"
}
parser = argparse.ArgumentParser()
parser.add_argument("slug", choices=SOURCES)
args = parser.parse_args()
target = RUN / "sources" / (args.slug + ".pdf")
assert not target.exists()
receipt = {"requested_url": SOURCES[args.slug], "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
try:
    request = urllib.request.Request(SOURCES[args.slug], headers={"User-Agent": "Academic-source-verification/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(30 * 1024 * 1024)
        receipt.update({"status_code": response.status, "final_url": response.url, "content_type": response.headers.get("Content-Type")})
    if not data.startswith(b"%PDF-"):
        raise ValueError("Response is not a PDF")
    with target.open("xb") as stream:
        stream.write(data)
    receipt.update({"status": "PASS", "path": str(target), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
except Exception as error:
    receipt.update({"status": "FAIL", "error": repr(error)})
with (RUN / (args.slug + "_download.json")).open("x", encoding="utf-8") as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2)
print(json.dumps(receipt, ensure_ascii=True))
if receipt["status"] != "PASS":
    raise SystemExit(1)
