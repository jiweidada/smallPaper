import json
import urllib.parse
import urllib.request
from pathlib import Path
RUN = Path(__file__).resolve().parent
rows = []
for title in ["Cascade R-CNN: Delving Into High Quality Object Detection", "DETRDistill: A Universal Knowledge Distillation Framework for DETR-families"]:
    query = urllib.parse.urlencode({"query": title, "limit": 2, "fields": "title,authors,year,externalIds,url"})
    row = {"title_query": title, "endpoint": "https://api.semanticscholar.org/graph/v1/paper/search?" + query}
    try:
        with urllib.request.urlopen(urllib.request.Request(row["endpoint"], headers={"User-Agent": "Academic-source-verification/1.0"}), timeout=15) as response:
            row.update(http_status=response.status, result=json.load(response))
    except Exception as error:
        row.update(status="API_FAILED_PRIMARY_CVF_FALLBACK", error=str(error))
    rows.append(row)
with (RUN / "semantic_scholar_ensemble_round.json").open("x", encoding="utf-8") as stream:
    json.dump(rows, stream, ensure_ascii=False, indent=2)
print([(x["title_query"], x.get("http_status", x.get("status"))) for x in rows])
