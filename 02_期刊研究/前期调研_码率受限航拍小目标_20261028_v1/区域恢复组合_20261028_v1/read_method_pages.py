"""Read/render selected original-paper method pages only."""
from pathlib import Path
import sys
from pypdf import PdfReader
import pypdfium2 as pdfium
sys.stdout.reconfigure(encoding='utf-8')
root=Path(__file__).resolve().parent
archive=root.parent.parent/'文献_相关原始论文_20261004'
specs={
 'PixelRL':('20261028_区域组合_PixelRL_arxiv_latest.pdf',[2,3,4],[3]),
 'FBCNN':('20261028_区域组合_FBCNN_ICCV2021.pdf',[3,4],[3])}
key=sys.argv[1];name,pages,render=specs[key]
pdf=PdfReader(archive/name)
for p in pages:
    print(f'\nOriginal {key} PDF page {p}\n')
    print((pdf.pages[p-1].extract_text() or '').encode('utf-8',errors='backslashreplace').decode('utf-8'))
dest=root/'reading_pages';dest.mkdir(exist_ok=True)
doc=pdfium.PdfDocument(archive/name)
for p in render:doc[p-1].render(scale=1.7).to_pil().save(dest/f'{key}_p{p}.png')
