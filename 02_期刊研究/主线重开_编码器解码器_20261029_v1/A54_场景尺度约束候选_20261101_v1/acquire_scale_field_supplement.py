from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
from urllib.request import Request, urlopen
from pypdf import PdfReader

P = Path(__file__).resolve().parent
LIB = P.parent.parent / "文献_相关原始论文_20261004"
OUT = P / "supplemental_literature"


def fetch(item):
    tag, url, kind = item
    row = {"tag": tag, "requested_url": url, "attempted_utc": datetime.now(timezone.utc).isoformat(),
           "reading_scope": "Acquisition only; no full-paper read attestation."}
    try:
        with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0 (academic source verification)"}), timeout=25) as response:
            row.update(HTTP_status=response.status, final_url=response.url)
            content = response.read(20 << 20)
        if kind == "pdf":
            assert content.startswith(b"%PDF-")
            reader = PdfReader(BytesIO(content))
            dest = LIB / "A54_ScaleField_CVPR2023.pdf"
            assert not dest.exists()
            dest.write_bytes(content)
            textpath = LIB / "_text/A54_ScaleField_CVPR2023.txt"
            assert not textpath.exists()
            textpath.write_text("\n\n".join(f"=== PDF PAGE {i+1} ===\n{page.extract_text()}" for i, page in enumerate(reader.pages)), encoding="utf-8")
            row.update(status="PDF_ACQUIRED_TEXT_EXTRACTED", path=str(dest), text_path=str(textpath), pages=len(reader.pages),
                bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        else:
            data = json.loads(content)
            dest = OUT / (tag + "_crossref.json")
            assert not dest.exists()
            dest.write_bytes(content)
            metadata = data["message"]
            row.update(status="PUBLISHER_DEPOSITED_METADATA_ONLY", path=str(dest),
                sha256=hashlib.sha256(content).hexdigest(), DOI=metadata.get("DOI"), title=metadata.get("title"),
                abstract_available=bool(metadata.get("abstract")), has_primary_pdf_link=any(x.get("content-type") == "application/pdf" for x in metadata.get("link", [])))
    except Exception as exc:
        row.update(status="FAILED_READING_SCOPE_UNCHANGED", error=repr(exc))
    return row


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    receipt = OUT / "acquisition_v1.json"
    assert not receipt.exists()
    items = [
        ("ScaleField", "https://openaccess.thecvf.com/content/CVPR2023/papers/Lee_Single_View_Scene_Scale_Estimation_Using_Scale_Field_CVPR_2023_paper.pdf", "pdf"),
        ("GSDDet", "https://api.crossref.org/works/10.1109%2FTGRS.2023.3309838", "json"),
        ("PixelsToMeters", "https://api.crossref.org/works/10.1109%2FLGRS.2025.3648944", "json"),
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(fetch, items))
    receipt.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False))
