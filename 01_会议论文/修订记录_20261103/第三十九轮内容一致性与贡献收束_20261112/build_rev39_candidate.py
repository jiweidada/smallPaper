from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

from lxml import etree as ET


ROOT = Path(r"C:\Users\jiwei\Desktop\小论文投稿\01_会议论文")
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "稿件" / "MPSMA-DEIM_conference_EN_revised.docx"
BACKUP = HERE / "MPSMA-DEIM_conference_EN_revised_rev38_backup.docx"
CANDIDATE = HERE / "MPSMA-DEIM_conference_EN_revised_rev39.docx"
SOURCE_SHA = "f598650e48c02ca2ffa46d5d9cc72cc8e24531641817000fe050f0fbc4068b23"
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(element) -> str:
    return "".join(
        node.text or "" for node in element.iter() if node.tag.endswith("}t")
    )


def save_json(name: str, value) -> None:
    (HERE / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


REPLACEMENTS = {
    5: (
        "Small objects in unmanned aerial vehicle (UAV) imagery provide limited visual evidence amid clutter and scale variation. "
        "We present MPSMA-DEIM, a DEIM-D-FINE adaptation that combines a Multi-branch Progressive Stem (MPS) with Multi-scale Context-aware Attention (MSCA). "
        "MPS fuses detail, downsampling and high-frequency paths into stride-four features; MSCA weights backbone features using pooled or dilated context and SimAM-derived energy gates. "
        "We evaluate the n configuration on 6,471 training images and all 548 VisDrone2019 validation images using a common training recipe and a frozen local Common Objects in Context (COCO)-style protocol. "
        "At the fixed round-320 exponential moving average (EMA) endpoint, the complete model (FULL) achieves average precision (AP) of 0.2255 and small-object AP (APs) of 0.1539, exceeding the original baseline (BASE) by 3.53 and 3.23 percentage points. "
        "FULL has 7.54 million trainable parameters versus 3.74 million for BASE. "
        "The conditional stem replacement (NO_MPS) yields AP/APs of 0.2099/0.1362; removing the three MSCA gates (NO_MSCA) yields 0.2222/0.1473. "
        "The latter comparison changes both the gates and model capacity, with 3.84 million fewer parameters in NO_MSCA. "
        "Results describe single-seed runs on one local validation split under documented checkpoint continuations."
    ),
    10: (
        "The design combines two backbone modifications. "
        "The Multi-branch Progressive Stem (MPS) aligns detail, main and high-frequency paths, applies residual spatial weighting, and fuses them before the final stem downsampling. "
        "Multi-scale Context-aware Attention (MSCA) applies pooled-context gates at stride 16 and a dilated-context gate at stride 32. "
        "The design contribution is the placement and combination of these paths and gates within the HGNetv2 backbone of DEIM-D-FINE, using established convolution, pooling and SimAM-style weighting operations. "
        "The complete configuration also uses full-width aggregation projections and the sigmoid-weighted linear unit (SiLU) in configurable blocks. "
        "We evaluate the complete n configuration and two conditional removals under a common frozen recipe, reporting fixed-round and validation-selected accuracy, parameter counts and the associated continuation histories."
    ),
    13: (
        "Feature recalibration and multi-scale aggregation are established design tools. "
        "Effective squeeze-excitation (ESE), introduced with VoVNetV2 in CenterMask [7], recalibrates channels. "
        "SimAM [8] derives neuron-wise weights from an energy function without additional learned parameters. "
        "DMF-Net pairs multi-branch detail extraction with dual-attention fusion [9], while TF-DEIM-DFINE combines a Mamba backbone, attention and a feature pyramid [10]. "
        "MPSMA-DEIM places branch fusion in the stem and contextual gates after HG-block aggregation. "
        "The pooled-context gate applies a SimAM-style rule to a mean of aligned feature maps; the dilated-context gate computes the rule on learned contextual features and combines it with channel weights. "
        "The evaluation treats module removal as a comparison conditional on the remaining backbone configuration."
    ),
    20: (
        "Given an input image X with even height and width, MPS replaces the standard stem with three paths (Fig. 2). "
        "In B0, the detail path uses two stride-one 3 × 3 convolution blocks with 8 channels, then 2 × 2 average pooling. "
        "The main path uses two 16-channel 3 × 3 blocks with strides two and one. "
        "The high-frequency path forms X − AvgPool3×3(X), using stride one and padding one, then applies a 1 × 1 projection and a depthwise 3 × 3 convolution with 4 channels. "
        "Bilinear interpolation with align_corners=False aligns this path to the main path. "
        "Concatenating the three outputs produces a 28-channel feature Z at half the input resolution."
    ),
    71: (
        "FULL has AR@100 of 0.3743 versus 0.3621 for NO_MPS. "
        "Its large-object recall is lower: ARl is 0.5446 versus 0.5615, a difference of −1.69 percentage points."
    ),
    72: (
        "Replacing the stem removes 5,325 trainable parameters while retaining the other FULL components. "
        "The stem operator and realized training histories both differ, so this result describes the conditional comparison of these runs and provides no estimate of variability across seeds."
    ),
    73: (
        "At round 320, FULL exceeds NO_MSCA by 0.33 percentage points in AP, 0.66 in APs and 0.28 in AR@100 (Table 2). "
        "This accuracy difference accompanies 3,841,280 additional trainable parameters in FULL, an increase from 3.70 million to 7.54 million. "
        "The comparison therefore combines gate removal with a substantial capacity change. "
        "The NO_MSCA AP-best secondary endpoint is round 302 (AP 0.2253), selected on the same validation split. "
        "BASE lacks the ESE/SimAM gates and differs in aggregation paths and activations, so it does not provide a joint-removal control. "
        "The design provides no MPS–MSCA interaction estimate."
    ),
    79: (
        "MPSMA-DEIM integrates a three-path stem and stage-specific contextual gates into the DEIM-D-FINE backbone. "
        "At the fixed round-320 EMA endpoint, FULL achieves AP 0.2255 and APs 0.1539, exceeding BASE by 3.53 and 3.23 percentage points with 2.02 times as many trainable parameters. "
        "FULL exceeds NO_MPS by 1.57 and 1.77 points in AP and APs, while its ARl is 1.69 points lower. "
        "NO_MSCA has 3.84 million fewer parameters and AP/APs 0.33/0.66 points lower than FULL at the same endpoint. "
        "These are single-seed observations on one local validation split under documented continuation histories. "
        "Repeated seeds and controls for continuation histories and gate capacity are needed for stronger component attribution. "
        "Latency and onboard performance remain unmeasured."
    ),
}

REASONS = {
    5: "Define EMA/AP/APs and run identifiers at first abstract use; report primary endpoints consistently; retain capacity and continuation limits.",
    10: "State the implemented integration and evaluation contribution without attributing the full comparison to MPS/MSCA alone.",
    13: "Connect the cited prior designs to the actual placement of the current modifications, retaining adaptation scope.",
    20: "Define input X and express the even-dimension requirement directly; preserve the implemented branch operations.",
    71: "Describe the opposing aggregate and large-object recall outcomes directly; remove repeated parameter-count sentence.",
    72: "Keep one parameter-change statement and the run-history limitation for the conditional stem comparison.",
    73: "Present the exact accuracy/capacity trade-off together and avoid repeating the gate definition from the setup.",
    79: "Condense the conclusion around fixed-round findings, trade-offs and the evidence needed for stronger attribution.",
}


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    assert sha(SOURCE) == SOURCE_SHA, "Maintained manuscript changed: re-read before editing."
    if BACKUP.exists():
        assert sha(BACKUP) == SOURCE_SHA
    else:
        shutil.copyfile(SOURCE, BACKUP)

    with zipfile.ZipFile(SOURCE) as z:
        assert z.testzip() is None
        infos = z.infolist()
        original = {info.filename: z.read(info.filename) for info in infos}
    package = dict(original)
    document = ET.fromstring(package["word/document.xml"])
    body = document.find("w:body", NS)
    original_refs = [plain(p) for p in body.findall("w:p", NS) if re.match(r"^\[\d+\]", plain(p))]
    original_tables = [ET.tostring(t) for t in body.findall("w:tbl", NS)]
    original_equations = [ET.tostring(t) for t in document.findall(".//m:oMath", NS)]
    edits = []
    for index, replacement in REPLACEMENTS.items():
        p = body[index]
        assert p.tag == f"{{{NS['w']}}}p"
        assert not p.findall(".//w:drawing", NS) and not p.findall(".//m:oMath", NS)
        text_nodes = p.findall("w:r/w:t", NS)
        assert len(text_nodes) == 1, f"Paragraph {index} requires a different patch strategy."
        before = plain(p)
        assert before and before != replacement
        text_nodes[0].text = replacement
        edits.append({
            "body_child_index": index,
            "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
            "before": before,
            "after": replacement,
            "reason": REASONS[index],
        })
    comments = ET.fromstring(package["word/comments.xml"])
    comment3 = comments.find("w:comment[@w:id='3']", NS)
    assert comment3 is not None
    before_comment = plain(comment3)
    assert re.search(r"MSCA\s*组仍待完成", before_comment)
    text_nodes = comment3.findall(".//w:t", NS)
    assert len(text_nodes) == 1
    text_nodes[0].text = re.sub(
        r"MSCA\s*组仍待完成。",
        "NO_MSCA已完成320轮并列入Table 2；其固定末轮与第302轮AP最佳次要端点的完整证据见批注6及第38轮索引。",
        before_comment,
    )
    package["word/document.xml"] = ET.tostring(document, xml_declaration=True, encoding="UTF-8", standalone=True)
    package["word/comments.xml"] = ET.tostring(comments, xml_declaration=True, encoding="UTF-8", standalone=True)
    with zipfile.ZipFile(CANDIDATE, "w") as z:
        for info in infos:
            z.writestr(info, package[info.filename])
    with zipfile.ZipFile(CANDIDATE) as z:
        assert z.testzip() is None
    changed = sorted(k for k in package if package[k] != original[k])
    assert changed == ["word/comments.xml", "word/document.xml"]
    assert original_tables == [ET.tostring(t) for t in body.findall("w:tbl", NS)]
    assert original_equations == [ET.tostring(t) for t in document.findall(".//m:oMath", NS)]
    assert original_refs == [plain(p) for p in body.findall("w:p", NS) if re.match(r"^\[\d+\]", plain(p))]
    assert not any(re.search(r"MSCA\s*组仍待完成", plain(c)) for c in comments.findall("w:comment", NS))

    paragraphs = [plain(p) for p in body.findall("w:p", NS)]
    abstract = plain(body[5])
    body_text = "\n\n".join(plain(p) for p in list(body)[7:80] if p.tag == f"{{{NS['w']}}}p")
    (HERE / "candidate_for_acronyms.md").write_text(
        "# Abstract\n\n" + abstract + "\n\n# Body\n\n" + body_text + "\n", encoding="utf-8"
    )
    (HERE / "candidate_readback.txt").write_text("\n\n".join(paragraphs), encoding="utf-8")
    # Names of methods, run identifiers, the backbone scale and CUDA product.
    allow = ["BASE", "FULL", "NO_MPS", "NO_MSCA", "DEIM", "DETR", "CUDA", "HG", "B0"]
    (HERE / "acronym_allowlist.txt").write_text("\n".join(allow) + "\n", encoding="utf-8")

    binding_path = ROOT / "远程公平对照_20261103" / "模块消融" / "method_source_binding_20261108.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8-sig"))
    checked_sources = []
    for entry in binding["files"]:
        p = ROOT / entry["method_evidence_path"]
        digest = sha(p)
        assert digest == entry["sha256"]
        checked_sources.append({"file": entry["method_evidence_path"], "sha256": digest})

    report = {
        "status": "REV39_CANDIDATE_PENDING_RENDER_AND_CONTENT_QA",
        "revision": 39,
        "observed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_sha256": SOURCE_SHA,
        "backup_sha256": sha(BACKUP),
        "candidate_sha256": sha(CANDIDATE),
        "changed_package_parts": changed,
        "changed_body_paragraph_count": len(edits),
        "changed_body_paragraphs": edits,
        "changed_comment": {"id": 3, "before": before_comment, "after": plain(comment3)},
        "source_structures_preserved": {
            "body_paragraphs": len(paragraphs),
            "tables": len(original_tables),
            "figures": len(document.findall(".//w:drawing", NS)),
            "native_equations": len(original_equations),
            "reference_entries": len(original_refs),
            "comments": len(comments.findall("w:comment", NS)),
            "all_table_equation_and_reference_bytes_unchanged": True,
            "all_media_parts_unchanged": True,
        },
        "abstract_word_count": len(abstract.split()),
        "abstract_was_targeted_revised": True,
        "abstract_numbers_all_fixed_round320_primary": True,
        "result_values_and_comparison_scope_unchanged": True,
        "four_method_sources_rehashed": checked_sources,
        "method_source_binding_file": str(binding_path),
        "method_source_binding_sha256": sha(binding_path),
        "workflow_note": "Authorized targeted DOCX OOXML patch with exact source hash and paragraph preimages. This is not a formal ARS markdown revision-patch schema certification or independent review.",
        "reviewer_independence": "Single in-session review; no independent or cross-model reviewer.",
        "new_training_runs": 0,
        "model_forwards": 0,
        "evaluator_runs": 0,
        "new_citations": 0,
        "venue_selected": False,
        "submission_ready": False,
    }
    save_json("targeted_patch_and_preliminary_check_rev39.json", report)
    print(json.dumps({k: report[k] for k in ["status", "source_sha256", "candidate_sha256", "changed_body_paragraph_count", "abstract_word_count", "source_structures_preserved"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
