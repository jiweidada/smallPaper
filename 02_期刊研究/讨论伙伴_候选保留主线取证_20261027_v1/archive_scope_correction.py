"""Bind scope correction to original documents; no experimental data or new literature."""
import datetime
import hashlib
import json
from pathlib import Path
RUN = Path(__file__).resolve().parent
ROOT = RUN.parent
OLD = ROOT / "data/00_实验计划与证据索引"
specs = [
    ("original_handoff", OLD / "双基线问题与创新思路_供新模型分析交接_20261008_v1/00_完整交接_已有问题证据与研究思路.txt", "Entire document read; lines8-9 original two problems and population;36-53 early groups;56-57 pipeline;110-118 encoderTop300;132-138 priorities;169-173 limits."),
    ("original_priority", OLD / "主问题收敛_20261008_v1/03_主要问题与研究收敛.txt", "Entire document read; lines7-9 priorities;24 unknown causes;31-33 key discriminating question; final limitations."),
    ("historical_controlled_path_supplement", OLD / "双基线问题与创新思路_供新模型分析交接_20261008_v1/07_交接后的新增证据_先读此补充.txt", "Entire document read; lines8-28 controlled path and group limits;35-46 no main-cause/recovery claim;69-72 geometry versus class counterexample."),
    ("later_problem_first_scope", ROOT / "持续寻找研究主线_20261020_v2/369_问题优先诊断_寻找新创新点_Prompt_v1.txt", "Entire document read; lines6-8 broad research scope;20-24 whole-pipeline diagnosis;54-68 interpretation/closure limits."),
    ("recent_final_pair_protocol", ROOT / "raw_EMA逐GT对照_20261026_v1/rawBC操作地图与EMA迁移_20261026_v1/protocol.json", "Entire protocol read; final300x10 pairs and A/B/C arms; cap100; no96/132 epoch transfer inference."),
    ("BASE24_lineage_plan_scope", ROOT / "候选保留_机制与创新假说_20261027_v1/Hlineage_下一最小零训练检验_冻结计划.json", "Previously read full plan; current binding for BASE24 raw/EMA population and pre/final stages."),
    ("overbroad_prior_review", RUN / "13_Hlineage实测与最终科学结论.txt", "Prior authored report; title and conclusion reviewed as the target of this scope erratum, not source evidence for original problems.")]
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
before = {str(p): digest(p) for _, p, _ in specs}
rows = [{"id": key, "path": str(p), "sha256": before[str(p)], "bytes": p.stat().st_size, "actual_read_scope": scope} for key, p, scope in specs]
out = {
    "date": "2026-10-27", "status": "SCOPE_ERRATUM_DOCUMENT_REVIEW_COMPLETE",
    "sources": rows,
    "key_claims": [
        {"claim": "Original two problems are early fine localization and early wrong-class prediction given good geometry; selection is a separate branch.", "source": "original_handoff", "anchor": "lines8-9", "judgment": "DIRECTLY_SUPPORTED"},
        {"claim": "Original priority was early localization first, common class confusion second.", "source": "original_priority", "anchor": "lines7-9", "judgment": "SUPPORTED_AS_OF_2026_10_08_NOT_IMMUTABLE_PRIORITY"},
        {"claim": "Old encoder8400-to300 query selection differs from recent final300x10-to300 pair selection.", "source": ["original_handoff", "recent_final_pair_protocol"], "anchor": "pipeline and section8 versus arms", "judgment": "DISTINCT_INTERFACES"},
        {"claim": "Old C wrong-class group differs from recent C export arm; finalG/K do not identify original early causes.", "source": ["original_handoff", "recent_final_pair_protocol"], "judgment": "SCOPE_TRANSFER_NOT_ESTABLISHED"},
        {"claim": "Old section7 diagnostic matching has no added class100; do not generalize this to all oldAP.", "source": ["original_handoff", "recent_final_pair_protocol"], "anchor": "section7 matching definition versus evaluation", "judgment": "DISTINCT_DIAGNOSTIC_MATCHING_RULES"},
        {"claim": "Existing controlled H/L path evidence is local and not mere counting; it did not repair most original never targets.", "source": "historical_controlled_path_supplement", "judgment": "HISTORICAL_REPORT_EVIDENCE_ONLY_NOT_RECOMPUTED"},
        {"claim": "Later broad diagnosis and explicit user-directed branch work had authorization and local value; it cannot stand in for original early-problem causal diagnosis.", "source": ["later_problem_first_scope", "recent_final_pair_protocol"], "judgment": "KEEP_BRANCH_RESULTS_WITHDRAW_OVERGENERALIZATION"}
    ],
    "correction": "Narrow prior no-mainline wording to the specific tested BASE24 final-candidate proxies/rules. It does not invalidate original observations or judge feasibility/new-method space for mature132EMA early localization/classification.",
    "unknowns_retained": ["Original feature/detail availability versus readout/learning utilization", "Semantic separability versus reading/classification/assignment-history factors", "Historical132EMA localCPU versus training-end original numerical-parity failure", "No original raw array reanalysis in this review"],
    "counterlimits": ["Recent controlled A/B/C and single-swap readout effects are more than descriptive partitions; they remain valid within their branch.", "369 allowed wholeGT diagnosis and did not make original localization priority permanently mandatory.", "Failure to find a method does not entail false observations, and reproducible counts do not entail a causal problem definition."],
    "preservation": "Original source documents,13, prior source audits and sealed delivery manifests remain unmodified; new CURRENT_SCOPE file and15 supersede scope interpretation only.",
    "operations": {"prediction_cache_loads":0,"new_experiments":0,"new_statistical_recomputations":0,"new_literature_searches":0,"model_forwards":0,"evaluator_calls":0,"training":0}
}
assert all(digest(Path(p)) == h for p, h in before.items())
with (RUN / "scope_correction_review.json").open("x", encoding="utf-8") as stream:
    json.dump(out, stream, ensure_ascii=False, indent=2)
print("PASS:7 source documents bound and unchanged; original-issue scope correction saved;0 experiments/literature searches")
