"""Check imports and versions only. Does not construct a model or run inference."""
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
checks = {}
for module in ("numpy", "torch", "torchvision", "yaml", "cv2", "psutil", "tqdm", "faster_coco_eval"):
    imported = importlib.import_module(module)
    checks[module] = {"file": imported.__file__, "version": getattr(imported, "__version__", None)}
for value in checks.values():
    if not Path(value["file"]).resolve().is_relative_to((ROOT / ".runtime/site-packages").resolve()):
        raise RuntimeError("An imported dependency came from outside the private environment")
result = {"status": "DEPENDENCY_IMPORTS_OK_NO_MODEL_RUN", "python": sys.version, "modules": checks,
          "packages": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
          "model_constructed": False, "inference_run": False}
(ROOT / "runtime_imports.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": result["status"], "versions": {key: value["version"] for key, value in checks.items()}}))
