from pathlib import Path
import json
import pypdfium2 as pdfium
from retrieve_static import ROOT
r=json.loads((ROOT/'FSIC_user_source_receipt.json').read_text(encoding='utf-8'))
doc=pdfium.PdfDocument(r['archive_path'])
folder=ROOT/'reading_pages';folder.mkdir(exist_ok=True)
for p in [3,4,5,6,7,8]:
 dest=folder/f'FSIC_p{p}.png'
 doc[p-1].render(scale=1.8).to_pil().save(dest)
print('Rendered six source pages for reading only.')
