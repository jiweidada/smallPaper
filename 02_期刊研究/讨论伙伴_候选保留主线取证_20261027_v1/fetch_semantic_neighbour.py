import hashlib
import json
import urllib.request
from pathlib import Path
RUN = Path(__file__).resolve().parent
url = "https://openaccess.thecvf.com/content/ACCV2024/papers/Senthivel_QR-DETR__Query_Routing_for_Detection_Transformer_ACCV_2024_paper.pdf"
receipt = {"date": "2026-10-27", "url": url}
try:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Academic-source-verification/1.0"}), timeout=30) as response:
        body = response.read()
        receipt.update(http_status=response.status, final_url=response.url)
    assert body.startswith(b"%PDF")
    path = RUN / "sources/qr_detr_accv2024.pdf"
    with path.open("xb") as stream:
        stream.write(body)
    receipt.update(status="PASS", path=str(path), bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
except Exception as error:
    receipt.update(status="FAIL", error=repr(error))
with (RUN / "qr_detr_accv2024_download.json").open("x", encoding="utf-8") as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2)
print(json.dumps(receipt, ensure_ascii=True))
if receipt["status"] != "PASS":
    raise SystemExit(1)
