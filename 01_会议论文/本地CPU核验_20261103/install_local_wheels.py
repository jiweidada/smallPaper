"""Install reviewed local wheel files into the private target; never use an index."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
SITE = ROOT / ".runtime/site-packages"
historical = ROOT.parent.parent / "02_期刊研究/前期调研_码率受限航拍小目标_20261028_v1/本机CPU短测_20261029_v1"
inventory = json.loads((ROOT / "local_dependency_inventory.json").read_text(encoding="utf-8"))
required = {"torchvision", "numpy", "pillow", "pyyaml", "opencv-python", "psutil", "packaging"}
selected = [row for row in inventory["wheels"] if row.get("package", "").lower() in required]
selected.append({"path": str(ROOT / ".runtime/wheels/faster_coco_eval-1.6.5-cp312-cp312-win_amd64.whl"),
                 "package": "faster-coco-eval", "version": "1.6.5",
                 "sha256": "ea816119febb88b994ad4facfb65b2e1717ea776ec8f59ac1ae165dd4f77051f"})
if len(selected) != 8 or not (ROOT / "offline_environment_copy.json").is_file():
    raise ValueError("Expected offline base copy and eight explicit wheel packages")
for row in selected:
    path = Path(row["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError("Wheel hash mismatch")
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Wheel CRC validation failed")
command = [sys.executable, "-I", "-m", "pip", "--isolated", "--disable-pip-version-check",
           "install", "--no-index", "--no-deps", "--no-compile", "--target", str(SITE),
           "--report", str(ROOT / "wheel_install_report.json")]
command += [row["path"] for row in selected]
process = subprocess.run(command, capture_output=True, timeout=600)
(ROOT / "wheel_install.log").write_bytes(process.stdout + b"\n" + process.stderr)
(ROOT / "wheel_install_receipt.json").write_text(json.dumps({
    "command": command, "exit_code": process.returncode, "wheel_files": selected,
    "network_used_by_install": False,
    "public_download": "The faster-coco-eval wheel was previously downloaded with user authorization from PyPI and matched its published SHA-256.",
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"exit_code": process.returncode, "wheel_count": len(selected)}))
raise SystemExit(process.returncode)
