"""Download only the author-hosted source PDFs identified in this literature task."""
import hashlib
import json
from pathlib import Path
import urllib.request

out = Path(__file__).resolve().parent / "sources"
sources = {
    "hoiem2012_author.pdf": "https://dhoiem.cs.illinois.edu/publications/eccv2012_detanalysis_derek.pdf",
    "hoiem_analysis_v2_author.pdf": "https://dhoiem.cs.illinois.edu/projects/detectionAnalysis/detectionAnalysis_v2.pdf",
}
receipts = []
for name, url in sources.items():
    destination = out / name
    request = urllib.request.Request(url, headers={"User-Agent": "Academic source reading"})
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = response.read(12_000_001)
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type")
    if len(payload) > 12_000_000 or not payload.startswith(b"%PDF-"):
        raise ValueError("Source is not a bounded PDF")
    with destination.open("xb") as stream:
        stream.write(payload)
    receipts.append({"path": str(destination), "source_url": url, "final_url": final_url,
                     "content_type": content_type, "bytes": len(payload),
                     "sha256": hashlib.sha256(payload).hexdigest()})
(out.parent / "download_receipts.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(receipts, ensure_ascii=True))
