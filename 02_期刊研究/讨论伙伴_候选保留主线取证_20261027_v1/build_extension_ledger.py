"""Append primary subset-selection and classical-math evidence without rewriting earlier reading scope."""
import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
specs = {
    "lddp": ([3, 4, 5], "CVPR2017 primary paper; Algorithm2 output-count limit, class/spatial DPP kernel and inference", "https://openaccess.thecvf.com/content_cvpr_2017/papers/Azadi_Learning_Detection_With_CVPR_2017_paper.pdf"),
    "qubo_pedestrian": ([2, 3, 4], "CVPR2013 primary paper; foreground shape coverage and confidence/overlap QUBO selection", "https://openaccess.thecvf.com/content_cvpr_2013/papers/Rujikietgumjorn_Optimized_Pedestrian_Detection_2013_CVPR_paper.pdf"),
    "matroid_notes": ([2, 3, 4, 6, 7], "Michel Goemans MIT18.433 lecture notes March2009; matching/transversal matroids, rank and submodularity. Mathematical background, not an object detection method or new research theorem.", "https://math.mit.edu/~goemans/18433S09/matroid-notes.pdf")
}
records = []
for slug, (pages, note, url) in specs.items():
    meta = json.loads((RUN / (slug + "_metadata.json")).read_text("utf-8"))
    pre = json.loads((RUN / (slug + "_preflight.json")).read_text("utf-8"))
    assert pre["verdict"] == "PASS"
    assert hashlib.sha256(Path(meta["pdf"]).read_bytes()).hexdigest() == meta["sha256"]
    records.append({"slug": slug, **meta, "actual_text_pages_read": pages,
                    "actual_visual_pages_read": [], "primary_url": url, "scope_note": note,
                    "structural_preflight": "PASS", "source_unchanged_after_reading": True})
root = Path(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\候选保留_机制与创新假说_20261027_v1")
partner = []
for name in ("class_prefix_bound/protocol.json", "class_prefix_bound/attempt01/results.json", "class_prefix_bound/attempt01/verification.json"):
    path = root / name
    partner.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
out = {"date": "2026-10-27", "reason_for_extension": "H1 prefix-capacity hypothesis was falsified; partner asked for direct global subset/matching-preservation neighbours before advancing H2b.",
       "records": records, "additional_reading_of_initial_sources": {"soft_nms": {"pages": [8, 9], "scope": "references used for backward search; not an extra method result"}},
       "total_distinct_detection_papers_with_initial_ledger": 11, "total_detection_PDF_variants": 13,
       "additional_mathematical_background_PDFs": 1, "S2": "Both added detection-paper API searches returned429; primary CVF PDF identities confirmed. See semantic_scholar_extension.json.",
       "parent_prefix_sources": partner, "parent_prefix_result": json.loads((root / "class_prefix_bound/attempt01/results.json").read_text("utf-8")),
       "operations": {"model_forwards": 0, "evaluator_calls": 0, "training": 0}}
with (RUN / "sources_extension_after_H1.json").open("x", encoding="utf-8") as stream:
    json.dump(out, stream, ensure_ascii=False, indent=2)
print("PASS: 3 primary sources and parent certificate snapshot recorded; earlier stage ledger preserved")
