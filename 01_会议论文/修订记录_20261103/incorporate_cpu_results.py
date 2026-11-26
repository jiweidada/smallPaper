"""Append the authorized CPU checks to the maintained manuscript, preserving history."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.text.paragraph import Paragraph

AUDIT = Path(__file__).resolve().parent
PAPER = AUDIT.parent
CPU = PAPER / "本地CPU核验_20261103"
DEST = PAPER / "稿件/MPSMA-DEIM_conference_EN_revised.docx"
EXPECTED = "d38bcbb479f2ed815385330064508b88849abef279c68c6c989bb680963ad451"
HISTORY = AUDIT / "第二轮留档"
PATCH = AUDIT / "cpu_revision_patch.json"
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()

current = sha(DEST)
if current != EXPECTED:
    if not PATCH.exists() or current != json.loads(PATCH.read_text(encoding="utf-8"))["after_sha256"]:
        raise RuntimeError("The maintained manuscript changed; do not overwrite manual edits")
HISTORY.mkdir(exist_ok=True)
base = HISTORY / DEST.name
if not base.exists():
    shutil.copy2(DEST, base)
if sha(base) != EXPECTED:
    raise RuntimeError("Second-round backup is not the expected manuscript")
for name in ("validation_report.json", "visual_qa.json", "revision_patch.json", "revised_manuscript.txt",
             "修订说明与未解决问题.txt", "revise_manuscript.py", "finalize_audit.py"):
    if not (HISTORY / name).exists():
        shutil.copy2(AUDIT / name, HISTORY / name)

comparison = json.loads((CPU / "comparison_report.json").read_text(encoding="utf-8"))
if comparison["status"] != "THREE_AUTHORIZED_EVALUATIONS_COMPLETED_AND_ARTIFACTS_CHECKED":
    raise ValueError("Completed and checked CPU evidence required")
doc = Document(base)
old_tables = [[[cell.text for cell in row.cells] for row in table.rows] for table in doc.tables]
changes = []

def replace_text(paragraph, new):
    old = paragraph.text
    nodes = paragraph._p.xpath(".//w:t")
    if not nodes:
        raise ValueError("Expected existing prose")
    nodes[0].text = new
    for node in nodes[1:]:
        node.text = ""
    changes.append({"before": old, "after": new})

def style_body(paragraph, size=10.5):
    paragraph.paragraph_format.space_after = Pt(5)
    paragraph.paragraph_format.line_spacing = 1.1
    paragraph.paragraph_format.widow_control = True
    for run in paragraph.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(size)
        run.font.color.rgb = RGBColor(0, 0, 0)

def before(anchor, text, style=None):
    element = OxmlElement("w:p")
    anchor._p.addprevious(element)
    paragraph = Paragraph(element, doc._body)
    if style:
        paragraph.style = style
    paragraph.add_run(text)
    style_body(paragraph)
    return paragraph

anchor = next(p for p in doc.paragraphs if p.text == "4.5 Late-epoch Variation and Evidence Limits")
heading = before(anchor, "4.5 Re-evaluation of Retained Checkpoints")
if anchor._p.pPr is not None:
    if heading._p.pPr is not None:
        heading._p.remove(heading._p.pPr)
    heading._p.insert(0, deepcopy(anchor._p.pPr))
for run in heading.runs:
    run.bold = True
heading.paragraph_format.keep_with_next = True
method = (
    "We re-evaluated the retained N0, N4 and S1 EMA checkpoints once on all 548 validation images using "
    "the archived current implementation. The checks used FP32 on an AMD Ryzen 7 9700X CPU, PyTorch 2.6.0, "
    "TorchVision 0.21.0 and faster-coco-eval 1.6.5. The original validation batch sizes were retained "
    "(16 for N0 and 4 for N4/S1), together with the 640 × 640 transforms, postprocessor and local COCO-style "
    "settings of Section 4.1. Each checkpoint and postprocessor loaded with strict key and shape checks; "
    "no missing or unexpected keys were found. The models were evaluated without deployment conversion. "
    "The verification records bind the predictions and metrics to SHA-256 hashes of the weights, "
    "configuration, current source, annotations and validation images. The historical runtime versions remain unverified."
)
before(anchor, method)
caption = before(anchor,
    "Table 6. Historical best-log rows and separate CPU re-evaluations of retained EMA checkpoints. "
    "AP and APs are fractions; Δ is CPU minus log in percentage points (pp).", "Caption")
caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in caption.runs:
    run.font.size = Pt(9)
    run.bold = True
caption.paragraph_format.keep_with_next = True
rows = [["ID", "Log AP", "CPU AP", "Δ AP / pp", "Log APs", "CPU APs", "Δ APs / pp"]]
for run in comparison["runs"]:
    ap, aps = run["metrics"][0], run["metrics"][3]
    rows.append([run["run_id"], f"{ap['historical_best_row']:.6f}", f"{ap['cpu_re_evaluation']:.6f}",
                 f"{ap['delta_percentage_points']:+.6f}", f"{aps['historical_best_row']:.6f}",
                 f"{aps['cpu_re_evaluation']:.6f}", f"{aps['delta_percentage_points']:+.6f}"])
table = doc.add_table(rows=len(rows), cols=7)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
table.autofit = False
widths = [.4, .95, .95, 1.2, .95, .95, 1.2]
for column, width in zip(table.columns, widths):
    column.width = Inches(width)
for i, values in enumerate(rows):
    for j, value in enumerate(values):
        cell = table.cell(i, j)
        cell.width = Inches(widths[j])
        cell.text = value
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        properties = cell._tc.get_or_add_tcPr()
        borders = OxmlElement("w:tcBorders")
        margins = OxmlElement("w:tcMar")
        for side in ("top", "left", "bottom", "right"):
            border = OxmlElement("w:" + side)
            for key, item in (("val", "single"), ("sz", "4"), ("color", "D9D9D9")):
                border.set(qn("w:" + key), item)
            borders.append(border)
            margin = OxmlElement("w:" + side)
            margin.set(qn("w:w"), "65")
            margin.set(qn("w:type"), "dxa")
            margins.append(margin)
        properties.append(borders)
        properties.append(margins)
        if i == 0:
            shade = OxmlElement("w:shd")
            shade.set(qn("w:fill"), "E8EDF2")
            properties.append(shade)
        for paragraph in cell.paragraphs:
            style_body(paragraph, 9)
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.line_spacing = 1
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.keep_with_next = i < len(rows) - 1
            for run in paragraph.runs:
                run.bold = i == 0
    props = table.rows[i]._tr.get_or_add_trPr()
    props.append(OxmlElement("w:cantSplit"))
    if i == 0:
        props.append(OxmlElement("w:tblHeader"))
anchor._p.addprevious(table._tbl)
before(anchor,
    "Table 6 is kept separate from the historical configuration comparisons in Tables 2 and 3. "
    "The maximum absolute AP difference is 0.00147 pp. The complete 12-metric vectors are not numerically "
    "identical: the largest absolute discrepancy is 0.08462 pp for N0 large-object recall. "
    "No equivalence tolerance was specified before these checks, and the causes of the differences have "
    "not been isolated. The results establish a current evaluation record for these retained weights; "
    "they do not certify the exact historical training implementation or resolve the training-condition "
    "confounds. N1, N2 and N3 were not re-evaluated.")
replace_text(anchor, "4.6 Late-epoch Variation and Evidence Limits")

limits = next(p for p in doc.paragraphs if p.text.startswith("The available evidence supports a documented architectural design"))
replace_text(limits,
    "The available evidence supports a documented architectural design and single-run configuration results. "
    "It does not yet isolate shallow-detail preservation, contextual weighting or activation effects. "
    "Some retained checkpoints contain last_epoch metadata that differs from the best-AP log index. "
    "The available solver can reset last_epoch when restoring a stage-one checkpoint, so this field alone "
    "cannot identify the evaluation that produced a historical value. Section 4.5 establishes current "
    "checkpoint-to-evaluation records for N0, N4 and S1, while exact historical attribution still requires "
    "the matching training-code, data and environment records. Qualitative box overlays without matched "
    "thresholds and ground truth are insufficient to resolve these limitations.")
conclusion = next(p for p in doc.paragraphs if p.text.startswith("MPSMA-DEIM combines a three-path shallow stem"))
old = conclusion.text
key = "The recorded n and s configurations reach validation AP of 0.2270 and 0.2876, respectively."
replacement = key + " Separate CPU checks of N0, N4 and S1 strictly load the retained EMA weights and yield AP differences of at most 0.00147 pp from the historical best-log values, with remaining differences in other metrics."
replace_text(conclusion, old.replace(key, replacement))

comment_texts = {
    "2": "第三轮已明确授权并完成 N0/N4/S1 留存 EMA 权重各一次 548 图 CPU 重评，均严格加载通过，整体 AP 最大差 0.00147 个百分点，但完整12指标并非逐项一致，差异原因未隔离。当前评价链已有哈希记录；历史代码/数据/环境与日志时刻仍未认证，N2 未重评。last_epoch 不能单独用于归属判定。详见本地CPU核验_20261103。",
    "4": "第二轮只读核查了现存 train/val 标注与原始有效标注对应，以及存量评价元数据。第三轮随后已授权重评 N0/N4/S1，使用当前文件和明确固定的 CPU 环境，结果单列 Table 6。当前标注与历史训练时逐字节相同仍未证实，本地 COCO-style 口径仍不同于官方 ignore/maxDets 协议。"
}
for comment in doc.comments:
    if str(comment.comment_id) in comment_texts:
        nodes = comment._element.xpath(".//w:t")
        nodes[0].text = comment_texts[str(comment.comment_id)]
        for node in nodes[1:]:
            node.text = ""

assert len(doc.tables) == 6 and len(doc.inline_shapes) == 7
assert [[[c.text for c in r.cells] for r in t.rows] for t in doc.tables[:5]] == old_tables
temporary = DEST.with_name("MPSMA-DEIM_conference_EN_revised.pending.docx")
doc.save(temporary)
temporary.replace(DEST)
with zipfile.ZipFile(base) as before_zip, zipfile.ZipFile(DEST) as after_zip:
    media = [name for name in before_zip.namelist() if name.startswith("word/media/")]
    assert all(before_zip.read(name) == after_zip.read(name) for name in media)
patch = {"before_sha256": EXPECTED, "after_sha256": sha(DEST), "backup": str(base),
         "cpu_evidence_sha256": sha(CPU / "comparison_report.json"),
         "prose_replacements": changes, "new_section": "4.5 Re-evaluation of Retained Checkpoints",
         "new_table": rows, "historical_tables_unchanged": True, "embedded_media_unchanged": True,
         "comments_updated": list(comment_texts), "submission_ready": False,
         "render_status": "PENDING_NEW_VISUAL_REVIEW"}
PATCH.write_text(json.dumps(patch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
body = []
for child in doc.element.body:
    if child.tag == qn("w:p"):
        body.append("".join(node.text or "" for node in child.xpath(".//w:t | .//m:t")))
    elif child.tag == qn("w:tbl"):
        for row in child.xpath("./w:tr"):
            body.append(" | ".join("".join(node.text or "" for node in cell.xpath(".//w:t")) for cell in row.xpath("./w:tc")))
(AUDIT / "revised_manuscript.txt").write_text("\n\n".join(body) + "\n", encoding="utf-8")
print(json.dumps({"docx_sha256": patch["after_sha256"], "tables": len(doc.tables),
                  "images": len(doc.inline_shapes), "comments": len(doc.comments), "render_status": patch["render_status"]}))
