"""Read-only source PDFs; save extraction and structural receipts in this run only."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
parser = argparse.ArgumentParser()
parser.add_argument("pdf")
parser.add_argument("slug")
parser.add_argument("--render", default="")
args = parser.parse_args()
run_dir = Path(__file__).resolve().parent
pdf = Path(args.pdf).resolve()
slug = args.slug
if not slug.replace("_", "").isalnum():
    raise ValueError("Unsafe slug")
preflight_path = Path(r"C:\Users\jiwei\.codex\skills\academic-research-suite\ars\scripts\pdf_read_preflight.py")
sys.path.insert(0, str(preflight_path.parent))
spec = importlib.util.spec_from_file_location("ars_pdf_preflight", preflight_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
receipt = module.run_preflight(pdf)
(run_dir / f"{slug}_preflight.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
if receipt["verdict"] != "PASS":
    print(json.dumps(receipt, ensure_ascii=True))
    raise SystemExit("Page anchors are not accepted without a structural PASS")
from pypdf import PdfReader
reader = PdfReader(pdf)
pages = [page.extract_text() or "" for page in reader.pages]
for number, body in enumerate(pages, 1):
    (run_dir / "text" / f"{slug}_p{number:03d}.txt").write_text(body, encoding="utf-8")
(run_dir / "text" / f"{slug}_full.txt").write_text("\n\n".join(f"=== PDF PAGE {n} ===\n{body}" for n, body in enumerate(pages, 1)), encoding="utf-8")
meta = {
    "pdf": str(pdf), "pages": len(pages), "bytes": pdf.stat().st_size,
    "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
    "metadata": {str(k): str(v) for k, v in (reader.metadata or {}).items()},
    "extraction_scope": "All pages extracted; extraction is not a claim of having read every page.",
}
(run_dir / f"{slug}_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
if args.render:
    import pypdfium2 as pdfium
    document = pdfium.PdfDocument(pdf)
    for number in map(int, args.render.split(",")):
        document[number - 1].render(scale=1.5).to_pil().save(run_dir / "figures" / f"{slug}_p{number:03d}.png")
print(json.dumps(meta, ensure_ascii=True))
