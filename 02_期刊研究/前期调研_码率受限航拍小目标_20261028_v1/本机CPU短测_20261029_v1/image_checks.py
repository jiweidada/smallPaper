"""Three train images only. Native JPEG encoding and label-transform smoke checks."""
from pathlib import Path
from io import BytesIO
from collections import Counter
import hashlib
import json
import math
import time
import numpy as np
from PIL import Image, features, __version__ as pillow_version

root = Path(__file__).resolve().parent
source_root = Path(r"D:\数据集\visdrone2019\VisDrone2019-DET-train")
audit = json.loads((root / "annotation_audit.json").read_text(encoding="utf-8"))
data = json.loads(Path(audit["source"]).read_text(encoding="utf-8"))
image_ids = {im["id"] for im in audit["candidate_fit_images_for_smoke"]}
selected_ann = {imid: [] for imid in image_ids}
overshoots = []
examples = []
ims = {im["id"]: im for im in data["images"]}
for a in data["annotations"]:
    x, y, w, h = a["bbox"]
    im = ims[a["image_id"]]
    amount = max(0, -x, -y, x+w-im["width"], y+h-im["height"])
    if w > 0 and h > 0 and amount > 0:
        overshoots.append(amount)
        if amount > 1 and len(examples) < 3:
            examples.append({"id": a["id"], "file_name": im["file_name"], "bbox": a["bbox"], "overshoot": amount})
    if a["image_id"] in image_ids:
        selected_ann[a["image_id"]].append(a)
out_dir = root / "payloads"
out_dir.mkdir(exist_ok=True)
records = []
encodes = 0
for im in audit["candidate_fit_images_for_smoke"]:
    path = source_root / "images" / im["file_name"]
    source_bytes = path.read_bytes()
    with Image.open(BytesIO(source_bytes)) as opened:
        rgb = opened.convert("RGB")
    assert rgb.size == (im["width"], im["height"])
    width, height = rgb.size
    native = np.asarray(rgb)
    r = min(640/height, 640/width)
    resized_w, resized_h = round(width*r), round(height*r)
    left, top = round((640-resized_w)/2-.1), round((640-resized_h)/2-.1)
    right, bottom = 640-resized_w-left, 640-resized_h-top
    clipped = []
    skipped = 0
    for a in selected_ann[im["id"]]:
        x, y, w, h = a["bbox"]
        x1, y1 = min(width,max(0,x)), min(height,max(0,y))
        x2, y2 = min(width,max(0,x+w)), min(height,max(0,y+h))
        if x2 <= x1 or y2 <= y1:
            skipped += 1
            continue
        box = [((x1+x2)*.5*r+left)/640, ((y1+y2)*.5*r+top)/640, (x2-x1)*r/640, (y2-y1)*r/640]
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in box) and box[2] > 0 and box[3] > 0
        clipped.append({"class": a["category_id"], "xywh_normalized": box})
    raw_lines = (source_root / "annotations" / (path.stem + ".txt")).read_text(encoding="utf-8-sig").splitlines()
    raw_counts = Counter()
    for line in raw_lines:
        f = [float(v) for v in line.strip().rstrip(",").split(",")]
        if int(f[4]) == 1 and 1 <= int(f[5]) <= 10:
            raw_counts[int(f[5])-1] += 1
    assert raw_counts == Counter(a["category_id"] for a in selected_ann[im["id"]])
    payloads = []
    for quality in (15,20,70):
        t = time.perf_counter()
        buf = BytesIO()
        rgb.save(buf,format="JPEG",quality=quality,subsampling=2,optimize=True,progressive=False)
        payload = buf.getvalue()
        encodes += 1
        repeat = BytesIO()
        rgb.save(repeat,format="JPEG",quality=quality,subsampling=2,optimize=True,progressive=False)
        encodes += 1
        assert payload == repeat.getvalue()
        target = out_dir / (path.stem + f"_q{quality}.jpg")
        target.write_bytes(payload)
        with Image.open(BytesIO(payload)) as dec:
            decoded = np.asarray(dec.convert("RGB"))
        assert decoded.shape == native.shape
        payloads.append({"quality": quality, "path": str(target), "bytes": len(payload),
                         "sha256": hashlib.sha256(payload).hexdigest(), "bpp": 8*len(payload)/(width*height),
                         "repeat_bytes_identical": True, "decoded_shape": list(decoded.shape),
                         "seconds_including_duplicate_encode_decode": round(time.perf_counter()-t,5)})
    records.append({"file_name": path.name,"source": str(path), "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                    "source_is_original_jpeg_not_raw": True, "size": [width,height], "image_id": im["id"],
                    "letterbox_dry_run": {"ratio":r, "resized_wh":[resized_w,resized_h],"pad_ltrb":[left,top,right,bottom]},
                    "gt_count": len(clipped), "skipped_boxes":skipped, "gt":clipped,
                    "raw_annotation_class_counts_match_converted": True, "payloads": payloads})
result={"status":"PASS", "scope":"3 train-fit images only; no val/test; no detector/restorer execution",
        "pillow":pillow_version,"libjpeg":features.version("jpg"),"libjpeg_turbo":features.version_feature("libjpeg_turbo"),
        "numpy":np.__version__, "native_resolution_encoding":True,"subsampling":2,"optimize":True,"progressive":False,
        "jpeg_encode_calls":encodes,"unique_encoded_files":9,"source_images_decoded":3,
        "outside_box_followup":{"count":len(overshoots),"at_most_0_01_pixel":sum(v<=.01 for v in overshoots),
                                "above_1_pixel":sum(v>1 for v in overshoots),"max_pixels":max(overshoots,default=0),"examples":examples},
        "label_policy":"clip xyxy to image bounds before transform, drop nonpositive-area boxes; original files unchanged",
        "images":records}
(root/"image_checks.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"status":"PASS","images":len(records),"jpeg_encode_calls":encodes,"unique_files":9,
                  "outside_box_followup":result["outside_box_followup"],"source_shapes":[r["size"] for r in records]},ensure_ascii=True),flush=True)
