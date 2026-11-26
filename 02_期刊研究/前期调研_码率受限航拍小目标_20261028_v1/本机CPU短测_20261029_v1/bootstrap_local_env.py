"""Copy the existing bundled pip into this test-only venv; never edit source runtime."""
from pathlib import Path
import importlib.metadata
import json
import shutil
import sys

root = Path(__file__).resolve().parent
source_site = Path(sys.base_prefix) / "Lib/site-packages"
target_site = root / ".venv312/Lib/site-packages"
target_site.mkdir(parents=True, exist_ok=True)
source_dirs = [source_site / "pip"] + list(source_site.glob("pip-*.dist-info"))
copied = []
for src in source_dirs:
    shutil.copytree(src, target_site / src.name, dirs_exist_ok=True)
    copied.append(src.name)
receipt = {"python_source": sys.executable, "target": str(target_site), "copied": copied,
           "pip_version": importlib.metadata.version("pip"),
           "reason": "ensurepip bundled pip 25.0.1 registry fallback fails under sandbox; portable venv requires explicit base DLL directories",
           "global_files_modified": False}
(root / "environment_bootstrap.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(receipt, ensure_ascii=True))
