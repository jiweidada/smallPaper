"""Use the private local packages with the recorded portable Python interpreter."""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")
SITE = ROOT / ".runtime" / "site-packages"
if not SITE.is_dir():
    raise RuntimeError("Private local environment is not prepared")
if len(sys.argv) < 2:
    raise SystemExit("Choose a script inside the evaluation bundle")
script = Path(sys.argv[1]).resolve()
if not script.is_relative_to(ROOT) or not script.is_file():
    raise ValueError("Script must be inside the evaluation bundle")
handles = []
for directory in (Path(sys.base_prefix), Path(sys.base_prefix) / "DLLs", SITE / "torch" / "lib"):
    if directory.is_dir() and hasattr(os, "add_dll_directory"):
        handles.append(os.add_dll_directory(str(directory)))
sys.dont_write_bytecode = True
sys.path = [str(SITE), str(ROOT)] + [p for p in sys.path
           if "site-packages" not in p.replace("\\", "/").split("/") and p not in (str(ROOT), "")]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for variable in ("LOCAL_APPDATA", "APPDATA", "COMMON_APPDATA"):
    destination = ROOT / ".runtime" / "cache" / variable.lower()
    destination.mkdir(parents=True, exist_ok=True)
    os.environ["WIN_PD_OVERRIDE_" + variable] = str(destination)
sys.argv = [str(script)] + sys.argv[2:]
runpy.run_path(str(script), run_name="__main__")
