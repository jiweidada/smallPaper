import json
import urllib.parse
import urllib.request
from pathlib import Path
RUN = Path(__file__).resolve().parent
titles = ["End-to-End Weakly Supervised Object Detection with Sparse Proposal Evolution", "QR-DETR : Query Routing for Detection Transformer", "Align-DETR: Enhancing End-to-end Object Detection with Aligned Loss", "AQS-IDETR: Adaptive Query Selection for Efficient Inference in Real-Time Detection Transformers"]
records = []
for title in titles:
    query = urllib.parse.urlencode({"query": title, "limit": 2, "fields": "title,authors,year,externalIds,url"})
    record = {"title_query": title, "endpoint": "https://api.semanticscholar.org/graph/v1/paper/search?" + query}
    try:
        with urllib.request.urlopen(urllib.request.Request(record["endpoint"], headers={"User-Agent": "Academic-source-verification/1.0"}), timeout=15) as response:
            record.update(http_status=response.status, result=json.load(response))
    except Exception as error:
        record.update(status="API_FAILED_PRIMARY_PUBLISHER_FALLBACK", error=str(error))
    records.append(record)
with (RUN / "semantic_scholar_extension_latest.json").open("x", encoding="utf-8") as stream:
    json.dump(records, stream, ensure_ascii=False, indent=2)
print([(r["title_query"], r.get("http_status", r.get("status"))) for r in records])
