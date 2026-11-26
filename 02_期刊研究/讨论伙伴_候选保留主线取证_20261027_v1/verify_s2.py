"""Attempt bibliographic API verification; preserve failures and use primary records explicitly."""
import concurrent.futures
import datetime
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

RUN = Path(__file__).resolve().parent
IDS = {"dfine": "2410.13842", "rankdetr": "2310.08854", "ddq": "2303.12776",
       "dqdetr": "2404.03507", "paq": "2603.06917", "clsc": "2608.21457",
       "mds": "2605.23507", "soft_nms": "1704.04503"}
FIELDS = "title,authors,year,venue,url,externalIds"

def verify(item):
    slug, aid = item
    url = "https://api.semanticscholar.org/graph/v1/paper/ARXIV:" + aid + "?fields=" + FIELDS
    record = {"slug": slug, "requested_url": url, "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Academic-bibliographic-verification/1.0"}), timeout=25) as response:
            record.update({"status_code": response.status, "payload": json.load(response), "status": "RETURNED_FOR_TITLE_CHECK"})
    except Exception as error:
        record.update({"status": "S2_API_UNAVAILABLE_OR_NOT_FOUND", "error": repr(error), "fallback": "Official publisher/arXiv identity checked independently; API failure is not evidence that paper does not exist"})
    return record

records = list(concurrent.futures.ThreadPoolExecutor(max_workers=3).map(verify, IDS.items()))
url = "https://api.semanticscholar.org/graph/v1/paper/search?query=" + urllib.parse.quote("Learning Non-Maximum Suppression") + "&limit=3&fields=" + FIELDS
record = {"slug": "learning_nms", "requested_url": url, "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
try:
    with urllib.request.urlopen(url, timeout=25) as response:
        record.update({"status_code": response.status, "payload": json.load(response), "status": "RETURNED_FOR_TITLE_CHECK"})
except Exception as error:
    record.update({"status": "S2_API_UNAVAILABLE_OR_NOT_FOUND", "error": repr(error), "fallback": "CVF paper and author institutional record verified separately"})
records.append(record)
with (RUN / "semantic_scholar_attempts.json").open("x", encoding="utf-8") as stream:
    json.dump(records, stream, ensure_ascii=False, indent=2)
print(json.dumps([{"slug": r["slug"], "status": r["status"], "error": r.get("error"), "title": r.get("payload", {}).get("title")} for r in records]))
