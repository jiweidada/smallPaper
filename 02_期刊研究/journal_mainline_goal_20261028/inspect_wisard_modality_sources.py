"""Source-label and pixel inspection only. No model or prediction evaluation."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

archive = Path(r"D:\微信聊天记录\xwechat_files\wxid_mgvv5yspt9mf22_ab41\msg\file\2026-10\WiSARD_Multi_Modal_Sample.zip")
out = Path(__file__).resolve().parent / "wisard_modal_source_check_20261029"
out.mkdir(exist_ok=True)
before = archive.stat()
prefixes = {"VIS": "210417_MtErie_Enterprise_VIS_0003", "IR": "210417_MtErie_Enterprise_IR_0004"}
counts = {}
labels = {}
members = {}
sources = {}
with zipfile.ZipFile(archive) as zf:
    names = set(zf.namelist())
    for mode, prefix in prefixes.items():
        pattern = re.compile(r"^" + re.escape(prefix) + r"/" + re.escape(prefix) + r"_(\d{8})\.txt$")
        entries = [(int(match.group(1)), name) for name in names if (match := pattern.match(name))]
        for ordinal, name in sorted(entries):
            raw = zf.read(name)
            rows = [[float(t) for t in line.split()] for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
            if any(len(row) != 5 for row in rows):
                raise ValueError("unexpected label row format")
            counts.setdefault(ordinal, {})[mode] = len(rows)
            labels[(ordinal, mode)] = rows
            members[(ordinal, mode)] = name[:-4] + ".jpeg"
    common_ordinals = [n for n, c in sorted(counts.items()) if all(mode in c for mode in prefixes)]
    diffs = [{"filename_ordinal": n, "author_label_rows": counts[n]} for n in common_ordinals if counts[n]["VIS"] != counts[n]["IR"]]
    # Fixed anchors plus the first two label-count differences. No model outputs exist here.
    ordinals = list(dict.fromkeys([common_ordinals[0], 132] + [d["filename_ordinal"] for d in diffs[:2]]))
    fig, axes = plt.subplots(len(ordinals), 2, figsize=(14, 4.6 * len(ordinals)), squeeze=False)
    for i, ordinal in enumerate(ordinals):
        for j, mode in enumerate(["VIS", "IR"]):
            name = members[(ordinal, mode)]
            payload = zf.read(name)
            if len(payload) > 12_000_000:
                raise ValueError("selected image exceeds bound")
            img = Image.open(io.BytesIO(payload))
            img.load()
            width, height = img.size
            local = out / f"source_{mode.lower()}_{ordinal:08d}.jpeg"
            local.write_bytes(payload)
            ax = axes[i, j]
            ax.imshow(img, cmap="gray" if img.mode == "L" else None)
            box_records = []
            for line, row in enumerate(labels[(ordinal, mode)], start=1):
                _, cx, cy, bw, bh = row
                x, y, w, h = (cx-bw/2)*width, (cy-bh/2)*height, bw*width, bh*height
                ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="cyan", linewidth=1.2))
                ax.text(x, max(0, y-3), str(line), color="yellow", fontsize=8)
                box_records.append({"source_label_line": line, "native_xywh": [x, y, w, h]})
            ax.set_title(f"{mode} filename ordinal {ordinal}; {width}x{height}; author rows {counts[ordinal][mode]}")
            ax.axis("off")
            sources[f"{ordinal}:{mode}"] = {"image_member": name, "label_member": name[:-5] + ".txt", "image_sha256": hashlib.sha256(payload).hexdigest(), "native_dimensions": [width, height], "boxes": box_records, "saved_source": str(local)}
    fig.suptitle("Author label overlays only. Same filename ordinal does not prove temporal/spatial registration.", fontsize=11)
    fig.subplots_adjust(top=0.96, bottom=0.02, left=0.02, right=0.98, hspace=0.16, wspace=0.06)
    figure = out / "source_modal_views.png"
    fig.savefig(figure, dpi=145)
    plt.close(fig)
after = archive.stat()
summary = {
    "scope": "source annotation rows, image dimensions and author-label visualization; not model performance",
    "archive": str(archive), "archive_bytes": before.st_size,
    "archive_size_mtime_stable": (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
    "counts_by_modality": {mode: dict(Counter(v[mode] for v in counts.values() if mode in v)) for mode in prefixes},
    "label_row_count_differences": diffs,
    "filename_ordinal_presence": {mode: [n for n, c in sorted(counts.items()) if mode in c] for mode in prefixes},
    "selected_ordinals": ordinals, "selection_rule": "first common filename ordinal/132 plus first two same-number annotation-count differences; not temporally registered pairs; no detector output used",
    "sources": sources, "source_plot": str(figure),
    "not_inferred": ["equal ordinals prove synchronization", "different label rows prove an IR missed detection", "labels supply physical identity", "native boxes establish a model's detection error", "a thermal bright region represents two people"],
    "no_forward_training_prediction_matching_reevaluation_or_profiling": True,
}
(out / "source_modal_receipt.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k not in ["sources", "label_row_count_differences"]}, ensure_ascii=False))
print(json.dumps({"difference_count": len(diffs), "first10_differences": diffs[:10]}, ensure_ascii=False))
