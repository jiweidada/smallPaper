"""Render a few original PDF pages for evidence checking only."""
from pathlib import Path
import pypdfium2 as pdfium
root = Path(__file__).resolve().parent
archive = root.parent.parent/'文献_相关原始论文_20261004'
folder=root/'reading_pages';folder.mkdir(exist_ok=True)
items=[('TTAC','20261028_主线补检_TTAC_CVF.pdf',[4]),
       ('QGAC','20261028_主线补检_QGAC_ECCV2020.pdf',[4,5]),
       ('DocQT','20261028_主线补检_DocQT_arxiv_v2.pdf',[7])]
for name,filename,pages in items:
    document=pdfium.PdfDocument(archive/filename)
    for page in pages:
        document[page-1].render(scale=1.7).to_pil().save(folder/f'{name}_p{page}.png')
print('Four original-paper pages rendered for reading; no model execution.')
