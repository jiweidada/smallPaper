"""Read bounded public source metadata; no model/prediction/evaluator imports."""
import hashlib
import json
from pathlib import Path
import urllib.request

root = Path(__file__).resolve().parent / "aewd_source_metadata"
root.mkdir(exist_ok=True)
url = "https://api.github.com/repos/maml1116/AEWD/git/trees/main?recursive=1"
receipt = {"url": url, "max_bytes": 5_000_000, "operation": "public_git_tree_only"}
try:
    req = urllib.request.Request(url, headers={"User-Agent": "AcademicSourceMetadata/1.0"})
    with urllib.request.urlopen(req, timeout=22) as response:
        payload = response.read(receipt["max_bytes"] + 1)
        receipt["status"] = response.status
        receipt["final_url"] = response.url
    if len(payload) > receipt["max_bytes"]:
        raise ValueError("Source tree exceeds fixed byte bound")
    obj = json.loads(payload)
    (root / "git_tree.json").write_bytes(payload)
    receipt.update(result="saved", bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(), tree_sha=obj.get("sha"), truncated=obj.get("truncated"))
    summary = {
        "tree_sha": obj.get("sha"), "truncated": obj.get("truncated"),
        "entry_count": len(obj.get("tree", [])),
        "annotation_and_doc_entries": [r for r in obj.get("tree", []) if r.get("type") == "blob" and (r["path"].lower().endswith((".json", ".pdf", ".md", ".xlsx", ".yaml")) or "script" in r["path"].lower())],
        "image_blob_count": sum(r.get("type") == "blob" and r["path"].lower().endswith((".jpg", ".jpeg", ".png")) for r in obj.get("tree", [])),
        "image_blob_sizes_are_git_entry_sizes_not_verified_decoded_image_sizes": True,
        "no_images_or_videos_downloaded": True,
    }
    (root / "tree_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
except Exception as exc:
    receipt.update(result="failed", error_type=type(exc).__name__, error=str(exc))
    print(json.dumps(receipt, ensure_ascii=False))
(root / "git_tree_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
