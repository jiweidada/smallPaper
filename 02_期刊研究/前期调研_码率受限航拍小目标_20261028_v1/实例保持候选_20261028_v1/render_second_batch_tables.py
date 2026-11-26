"""Render original source-paper tables only; no dataset or model access."""
from pathlib import Path
import pypdfium2 as pdfium
root=Path(__file__).resolve().parent
archive=root.parent.parent/'文献_相关原始论文_20261004'
for key,name,pages in [
 ('DAME','20261028_第二批失败_DAME_arxiv_v1.pdf',[7,8]),
 ('LAF','20261028_第二批失败_LAF_arxiv_v1.pdf',[7,8])]:
    pdf=pdfium.PdfDocument(archive/name)
    for p in pages:
        pdf[p-1].render(scale=1.7).to_pil().save(root/'reading_pages'/f'{key}_p{p}.png')
print('Four original PDF table pages rendered for reading.')
