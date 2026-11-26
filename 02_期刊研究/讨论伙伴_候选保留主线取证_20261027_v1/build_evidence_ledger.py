"""Record exact paper variants, hashes and actual reading scope; no model/data evaluation."""
import csv
import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
scopes = {
    "dfine": ([4, 5, 6, 9], [], "ICLR2025 publisher PDF; FDR/GO-LSD and ablation text; no claim that the paper explicitly describes the local code's LQE", "https://openreview.net/pdf?id=MFZjrTFE7h"),
    "rankdetr_v1": ([1, 3, 4, 5], [], "arXiv2310.08854v1, local watermark confirmed; preliminary and rank-oriented method", "https://arxiv.org/abs/2310.08854v1"),
    "rankdetr_publisher": ([3, 4, 5], [], "NeurIPS2023 publisher version, 14 pages, method re-read rather than assuming local v1 is final", "https://proceedings.neurips.cc/paper_files/paper/2023/file/34074479ee2186a9f236b8fd03635372-Paper-Conference.pdf"),
    "ddq_v2": ([3, 4, 5], [], "arXiv2303.12776v2; sparse/dense analysis, DQS and DDQ DETR setup; page1 first28 lines also read", "https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Dense_Distinct_Query_for_End-to-End_Object_Detection_CVPR_2023_paper.html"),
    "dqdetr_v4": ([5, 7], [], "Local older arXiv2404.03507v4; dynamic query figure and method; not silently treated as current", "https://arxiv.org/abs/2404.03507v4"),
    "dqdetr_v6": ([7, 8], [], "Latest arXiv version v6 at verification date; query numbers and encoder feature topK formulation re-read", "https://arxiv.org/abs/2404.03507v6"),
    "paq": ([3, 4, 5], [4], "CVPR2026 publisher PDF; dynamic query representation, quality-aware supervision and fixed 300/900 configurations", "https://openaccess.thecvf.com/content/CVPR2026/html/Kang_PaQ-DETR_Learning_Pattern_and_Quality-Aware_Dynamic_Queries_for_Object_Detection_CVPR_2026_paper.html"),
    "clsc": ([1, 2, 3, 4], [3], "arXiv2608.21457v1 preprint; equations1-5 and experiments/limits; reference page5 not claimed read", "https://arxiv.org/abs/2608.21457v1"),
    "mds": ([3, 4], [4], "arXiv2605.23507v1 preprint; final suppression stage and masking, not treated as accepted IEEE Access because PDF retains template headers", "https://arxiv.org/abs/2605.23507v1"),
    "learning_nms": ([2, 3, 4, 6, 7], [], "CVPR2017 publisher PDF; pure rescoring, class handling, conditional interaction and experiment scope; all page numbers here are PDF ordinals", "https://openaccess.thecvf.com/content_cvpr_2017/papers/Hosang_Learning_Non-Maximum_Suppression_CVPR_2017_paper.pdf"),
    "soft_nms": ([3, 4], [], "ICCV2017 publisher PDF; per-class detection-stage suppression and linear/Gaussian rescoring; abstract/page1 also read via primary web PDF", "https://openaccess.thecvf.com/content_ICCV_2017/papers/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.pdf")
}
records = []
for slug, (pages, images, note, url) in scopes.items():
    meta = json.loads((RUN / (slug + "_metadata.json")).read_text("utf-8"))
    pre = json.loads((RUN / (slug + "_preflight.json")).read_text("utf-8"))
    assert pre["verdict"] == "PASS"
    pdf = Path(meta["pdf"])
    current = hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert current == meta["sha256"]
    assert all(1 <= number <= meta["pages"] for number in pages)
    records.append({"slug": slug, "primary_url": url, **meta, "structural_preflight": "PASS",
                    "actual_text_pages_read": pages, "partial_extra_scope": "first28 lines of p1" if slug == "ddq_v2" else None,
                    "actual_visual_pages_read": images, "scope_note": note,
                    "all_pages_extracted_is_not_all_pages_read": True, "source_unchanged_after_reading": True})
ledger = {"date": "2026-10-27", "distinct_papers": 9, "pdf_variants": len(records), "records": records,
          "code_sources": json.loads((RUN / "local_code_sources.json").read_text("utf-8")),
          "verification": "Primary publisher/arXiv identity checked for all9; S2 attempted for all9, 7 returned429; PaQ and Learning NMS exact title/authors/year matched in successful payloads. Failures use explicitly logged primary-source fallback.",
          "limitations": "Focused nearest-method scan, not a systematic novelty search. No independent replication of any paper's performance; source text and code inspection do not validate the current trained model's claimed mechanism.",
          "operations": {"new_model_forwards": 0, "evaluator_calls": 0, "training": 0},
          "AI_assistance": "Primary-source retrieval, extraction and synthesis were AI-assisted; statements and reading scope are logged for human verification."}
with (RUN / "sources_and_reading.json").open("x", encoding="utf-8") as stream:
    json.dump(ledger, stream, ensure_ascii=False, indent=2)
print(json.dumps({"papers": 9, "variants": len(records), "all_source_hashes_unchanged": True}))
