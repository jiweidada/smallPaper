"""Copy selected existing packages into a new local environment; no network."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
TARGET = ROOT / ".runtime" / "site-packages"
CPU = WORKSPACE / "90_历史与工具" / "临时资料" / "cpu_smoke_runtime_20261029"
CONDA = Path(r"D:\Anaconda\Lib\site-packages")
PREFIXES = ("torch", "functorch", "filelock", "fsspec", "mpmath", "networkx", "sympy",
            "jinja2", "markupsafe", "typing_extensions", "setuptools", "pkg_resources", "_distutils_hack")

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    if TARGET.exists():
        raise FileExistsError("Refusing to replace an existing environment")
    if not TARGET.resolve().is_relative_to(ROOT):
        raise ValueError("Environment path escapes evaluation bundle")
    selected = [p for p in CPU.iterdir() if any(p.name == name or p.name.startswith(name + "-")
                or p.name == name + ".py" for name in PREFIXES)]
    selected += [CPU / "torchgen"]
    selected += [p for p in CONDA.iterdir() if p.name == "tqdm" or p.name.startswith("tqdm-")]
    if not any(p.name == "torch" for p in selected):
        raise ValueError("Local PyTorch source missing")
    TARGET.mkdir(parents=True)
    copied = []
    for source in selected:
        if source.is_symlink():
            raise ValueError("Unexpected package symlink")
        destination = TARGET / source.name
        if source.is_dir():
            shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            files = sorted(p for p in destination.rglob("*") if p.is_file())
        else:
            shutil.copy2(source, destination)
            files = [destination]
        copied.append({"source": str(source), "destination": str(destination),
                       "files": [{"path": p.relative_to(TARGET).as_posix(), "bytes": p.stat().st_size,
                                  "sha256": digest(p)} for p in files]})
    result = {"status": "OFFLINE_PACKAGES_COPIED_NO_MODEL_RUN", "packages": copied,
              "source_paths_unchanged": True, "network_access": False}
    (ROOT / "offline_environment_copy.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "entries": len(copied),
                      "files": sum(len(x["files"]) for x in copied),
                      "bytes": sum(f["bytes"] for x in copied for f in x["files"])}))

if __name__ == "__main__":
    main()
