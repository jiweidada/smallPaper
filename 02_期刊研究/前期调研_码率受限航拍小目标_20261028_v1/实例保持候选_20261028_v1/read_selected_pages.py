"""Print selected source-paper pages for actual reading; no experiment."""
from pathlib import Path
import sys
from pypdf import PdfReader
sys.stdout.reconfigure(encoding='utf-8')
archive=Path(__file__).resolve().parent.parent.parent/'文献_相关原始论文_20261004'
mapping={
 'TaskGuard':('20261028_实例保持_TaskGuard_arxiv_v1.pdf',[3,4,8]),
 'PCT':('20261028_实例保持_PCT_CVPR2021.pdf',[3,4]),
 'LTT':('20261028_实例保持_LTT_arxiv_v5.pdf',[1,3,4,6,7]),
 'DetectionCRC':('20261028_实例保持_DetectionCRC_PMLR2023.pdf',[10,11]),
 'CFM':('20261028_实例保持_CFM_arxiv_v1.pdf',[4,5]),
 'JCRR':('20261028_码率调研_JCRR_publisher.pdf',[4,5,6]),
 'DAME':('20261028_第二批失败_DAME_arxiv_v1.pdf',[6,7,8]),
 'LAF':('20261028_第二批失败_LAF_arxiv_v1.pdf',[7,8,9,10]),
 'Denoising':('20261028_第二批失败_UAV_Denoising_Computation2025.pdf',[1,6,7,16,19,21])}
key=sys.argv[1]
name,pages=mapping[key]
pdf=PdfReader(archive/name)
for page in pages:
    print(f'\n{key} source PDF page {page}\n')
    print((pdf.pages[page-1].extract_text() or '').encode('utf-8',errors='backslashreplace').decode('utf-8'))
