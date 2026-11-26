"""Activate DLL search paths for the portable Python venv, then run a module/script."""
from pathlib import Path
import os
import runpy
import sys

root = Path(__file__).resolve().parent
handles = []
for directory in (Path(sys.base_prefix), Path(sys.base_prefix) / "DLLs"):
    if directory.is_dir() and hasattr(os, "add_dll_directory"):
        handles.append(os.add_dll_directory(str(directory)))
for name in ("LOCAL_APPDATA", "APPDATA", "COMMON_APPDATA"):
    path = root / "cache" / name.lower()
    path.mkdir(parents=True, exist_ok=True)
    os.environ["WIN_PD_OVERRIDE_" + name] = str(path)
if sys.argv[1] == "--isolated-cpu":
    sys.argv.pop(1)
    runtime = root.parents[2] / "tmp/cpu_smoke_runtime_20261029"
    support = root.parents[2] / "tmp/yolo_smoke_runtime_20261029"
    sys.path = [str(support), str(runtime), str(root)] + [p for p in sys.path if "site-packages" not in p.replace("\\", "/").split("/")]
if sys.argv[1] == "-m":
    module = sys.argv[2]
    sys.argv = [module] + sys.argv[3:]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
else:
    script = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script)] + sys.argv[2:]
    runpy.run_path(str(script), run_name="__main__")
