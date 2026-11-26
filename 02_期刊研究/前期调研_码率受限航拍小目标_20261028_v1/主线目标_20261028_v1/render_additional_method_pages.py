"""Render original-paper method pages; no dataset/model operation."""
from pathlib import Path
import pypdfium2 as pdfium
root=Path(__file__).resolve().parent
archive=root.parent.parent/'文献_相关原始论文_20261004'
items=[('CropSAR','20261028_主线补检_Crop_SAR_RS2021.pdf',[7,9]),
       ('OADG','20261028_主线补检_OADG_arxiv.pdf',[3,4]),
       ('DNTOD','20261028_主线补检_DNTOD_arxiv.pdf',[5,6])]
for name,filename,pages in items:
    document=pdfium.PdfDocument(archive/filename)
    for page in pages:
        document[page-1].render(scale=1.7).to_pil().save(root/'reading_pages'/f'{name}_p{page}.png')
print('Six original-paper method pages rendered.')
