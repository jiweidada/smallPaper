"""Render cited source tables for visual reading, not model evaluation."""
from source_fetch import ROOT, ARCHIVE
import pypdfium2 as pdfium
targets={
 '20261028_码率调研_JCRR_publisher.pdf':[5,8,9],
 '20261027_条件新入口_Lost_in_Compression_CVPRW2022_Publisher.pdf':[5],
 '20261028_码率调研_TransTIC_publisher.pdf':[4],
 '20261028_码率调研_Pipeline_arxiv.pdf':[6],
 '20261028_码率调研_RD_MIM_official_CDN.pdf':[15],
}
folder=ROOT/'sources'/'reading_pages'; folder.mkdir(parents=True,exist_ok=True)
for file,pages in targets.items():
 doc=pdfium.PdfDocument(str(ARCHIVE/file))
 for number in pages:
  dest=folder/(file.removesuffix('.pdf')+f'_p{number}.png')
  doc[number-1].render(scale=1.7).to_pil().save(dest)
  print(str(dest),flush=True)
