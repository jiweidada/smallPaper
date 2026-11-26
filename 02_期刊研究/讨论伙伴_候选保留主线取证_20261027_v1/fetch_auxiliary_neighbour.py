"""Fetch exactly one official primary source and attempt independent bibliographic lookup."""
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path
RUN = Path(__file__).resolve().parent
url = "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690207.pdf"
request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(request, timeout=45) as response:
    body = response.read()
    status, final = response.status, response.url
assert status == 200 and body.startswith(b"%PDF")
path = RUN / "sources/spe_eccv2022.pdf"
with path.open("xb") as stream:
    stream.write(body)
receipt = {"url": url, "final_url": final, "http_status": status, "path": str(path),
           "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "date": "2026-10-27"}
(RUN / "spe_eccv2022_download.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8")
title = "End-to-End Weakly Supervised Object Detection with Sparse Proposal Evolution"
query = urllib.parse.urlencode({"query": title, "limit": 3, "fields": "title,authors,year,externalIds,url"})
record = {"title_query": title, "endpoint": "https://api.semanticscholar.org/graph/v1/paper/search?" + query}
try:
    with urllib.request.urlopen(urllib.request.Request(record["endpoint"], headers={"User-Agent": "ResearchSourceVerification/1.0"}), timeout=20) as response:
        record.update(http_status=response.status, result=json.load(response))
except Exception as error:
    record.update(status="API_FAILED_PRIMARY_ECVA_FALLBACK", error=str(error))
with (RUN / "semantic_scholar_auxiliary_scan.json").open("x", encoding="utf-8") as stream:
    json.dump(record, stream, ensure_ascii=False, indent=2)
print("Official SPE PDF verified, bytes", len(body), "SHA", receipt["sha256"], "S2", record.get("http_status", record.get("status")))
