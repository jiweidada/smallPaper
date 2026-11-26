"""Render supplied paper for source reading; does not import model code."""
from pathlib import Path
import json
import pypdfium2 as pdfium

root = Path(__file__).resolve().parent
receipt = json.loads((root / 'PDAD_user_source_receipt.json').read_text(encoding='utf-8'))
document = pdfium.PdfDocument(receipt['archive_path'])
destination = root / 'reading_pages'
destination.mkdir(exist_ok=True)
for number in [4, 5, 6, 7, 8, 9, 10]:
    document[number - 1].render(scale=1.8).to_pil().save(destination / f'PDAD_p{number}.png')
print('Rendered seven source pages for reading only; no model execution.')
