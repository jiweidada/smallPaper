"""Inspect the official wheel; separate non-Torch dependencies to keep CPU build fixed."""
from pathlib import Path
from email.parser import Parser
import hashlib
import json
import zipfile
from packaging.requirements import Requirement

root=Path(__file__).resolve().parent
wheel=root/"wheels/ultralytics-8.4.191-py3-none-any.whl"
with zipfile.ZipFile(wheel) as z:
    text=z.read("ultralytics-8.4.191.dist-info/METADATA").decode()
metadata=Parser().parsestr(text)
reqs=metadata.get_all("Requires-Dist")
support=["numpy==1.26.4","pillow==12.3.0","opencv-python==4.11.0.86"]
excluded={"torch","torchvision","ultralytics-thop","numpy","pillow","opencv-python"}
for text in reqs:
    req=Requirement(text)
    if req.name.lower() not in excluded and (not req.marker or req.marker.evaluate({"extra":""})):
        support.append(text)
(root/"yolo_support_requirements.txt").write_text("\n".join(support)+"\n",encoding="utf-8")
record={"wheel":str(wheel),"sha256":hashlib.sha256(wheel.read_bytes()).hexdigest(),"version":metadata["Version"],
        "all_requires_dist":reqs,"support_requirements":support,
        "torch_source":"separate official CPU index, torch 2.6.0+cpu / torchvision 0.21.0+cpu",
        "no_deps_install_later":["ultralytics-8.4.191-py3-none-any.whl","ultralytics-thop==2.0.18"]}
(root/"ultralytics_wheel_metadata.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"version":metadata["Version"],"sha256":record["sha256"],"support_requirements":support},ensure_ascii=True))
