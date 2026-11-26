"""Original-paper page rendering for evidence reading only."""
from pathlib import Path
import pypdfium2 as pdfium
root=Path(__file__).resolve().parent
archive=root.parent.parent/'文献_相关原始论文_20261004'
folder=root/'reading_pages';folder.mkdir(exist_ok=True)
items=[('TaskGuard','20261028_实例保持_TaskGuard_arxiv_v1.pdf',[3,4,6]),
 ('PCT','20261028_实例保持_PCT_CVPR2021.pdf',[4]),
 ('LTT','20261028_实例保持_LTT_arxiv_v5.pdf',[3,6]),
 ('DetectionCRC','20261028_实例保持_DetectionCRC_PMLR2023.pdf',[11]),
 ('CFM','20261028_实例保持_CFM_arxiv_v1.pdf',[5])]
for key,name,pages in items:
    pdf=pdfium.PdfDocument(archive/name)
    for p in pages:pdf[p-1].render(scale=1.7).to_pil().save(folder/f'{key}_p{p}.png')
print('Eight source-paper method pages rendered; no dataset images or models.')
