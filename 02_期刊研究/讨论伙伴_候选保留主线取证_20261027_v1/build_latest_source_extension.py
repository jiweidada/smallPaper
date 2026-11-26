"""Record focused correspondence/class-history reading and frozen partner probes."""
import hashlib
import json
from pathlib import Path
RUN = Path(__file__).resolve().parent
ROOT = RUN.parent / "候选保留_机制与创新假说_20261027_v1"
def load(p):
    return json.loads(p.read_text("utf-8-sig"))
def bind(p):
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
records = []
for slug, pages, url, note in [
    ("align_v2", [1,2,3,4,5,6], "https://arxiv.org/abs/2304.07527v2", "arXiv v2; BMVC2024 identity confirmed with publisher, not a claim all arXiv/publisher author metadata are identical."),
    ("qr_detr_accv2024", [1,5,6,7,8,9,10], "https://openaccess.thecvf.com/content/ACCV2024/papers/Senthivel_QR-DETR__Query_Routing_for_Detection_Transformer_ACCV_2024_paper.pdf", "Official ACCV2024 PDF, router feature-to-2D equations5-6 and pass/bypass/scatter; title/abstract portion ofp1 read.")]:
    meta = load(RUN / (slug + "_metadata.json"))
    assert bind(Path(meta["pdf"]))["sha256"] == meta["sha256"]
    assert load(RUN / (slug + "_preflight.json"))["verdict"] == "PASS"
    records.append({"slug": slug, **meta, "actual_text_pages_read": pages, "actual_visual_pages_read": [], "primary_url": url,
                    "structural_preflight": "PASS", "scope_note": note})
online = [
    {"title": "End-to-End Weakly Supervised Object Detection with Sparse Proposal Evolution", "venue": "ECCV2022",
     "primary_url": "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690207.pdf", "access": "Official PDF text via web tool",
     "actual_read_scope": "Sections3.1-3.4; one-to-one proposal matching and testing uses backbone+SPR, no SPG; not full17-page reading.",
     "web_reported_pages": 17, "local_pdf": None, "pdf_sha256": None, "local_structural_preflight": "NOT_AVAILABLE",
     "download_attempts": ["Python urllib SSL certificate chain verification failed", "PowerShell Invoke-WebRequest TLS authentication failed"],
     "TLS_verification_disabled": False, "identity_primary_crosscheck": "ECCV2022 accepted-paper list and official ECVA PDF author/title"},
    {"title": "AQS-IDETR: Adaptive Query Selection for Efficient Inference in Real-Time Detection Transformers", "venue": "Image and Vision Computing2026",
     "doi": "10.1016/j.imavis.2026.106097", "primary_url": "https://www.sciencedirect.com/science/article/pii/S0262885626002040",
     "access": "Publisher-indexed abstract/introduction; direct open403", "actual_read_scope": "Abstract and exposed introduction only; full method not verified.",
     "local_pdf": None, "status": "LEAD_ONLY_NOT_EXACT_METHOD_EXCLUSION"}]
paths = [ROOT / "pre_witness_information/protocol.json", ROOT / "pre_witness_information/attempt01/results.json", ROOT / "pre_witness_information/attempt01/verification.json", ROOT / "class_history_information/protocol.json"]
pre = load(paths[1]); ver = load(paths[2])
assert pre["status"] == "COMPLETE" and ver["status"] == "PASS"
assert pre["protocol_sha256"] == bind(paths[0])["sha256"]
assert pre["checks"]["images_global_proxy_saturated"] == 2192
assert all(v["counts"]["proxy_critical_full"] == 0 for v in pre["conditions"].values())
out = {"date": "2026-10-27", "local_PDF_records": records, "online_primary_records": online,
       "local_total_with_prior_ledgers": {"detection_PDF_versions": 15, "distinct_detection_papers": 13, "mathematical_background_PDFs": 1, "all_local_PDFs": 16},
       "additional_full_method_online_only_papers": 1, "additional_abstract_only_leads": 1,
       "S2": "Four title searches attempted, failed; exact errors in semantic_scholar_extension_latest.json; primary publisher fallback explicit.",
       "partner_bindings": [bind(p) for p in paths], "partner_Hpre_result_snapshot": pre,
       "Hpre_decision": "STOP_FROZEN_PROXY_CONSTANT_ZERO; empty strata and undefined CI are not negative statistical significance; no closure of all pre information.",
       "Hclass_status_at_recording": "frozen protocol; result not incorporated in this source-stage ledger",
       "operations_by_this_chat": {"model_forwards": 0, "prediction_cache_loads": 0, "evaluator_calls": 0, "training": 0}}
with (RUN / "sources_extension_latest.json").open("x", encoding="utf-8") as stream:
    json.dump(out, stream, ensure_ascii=False, indent=2)
print("PASS:2 local PDFs,2 online records and4 partner bindings saved with explicit reading/access limits")
