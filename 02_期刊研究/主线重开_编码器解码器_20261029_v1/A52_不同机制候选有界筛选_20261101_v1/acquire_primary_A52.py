from pathlib import Path
import hashlib,json,subprocess,urllib.request
ROOT=Path(__file__).resolve().parent
LIB=ROOT.parents[1]/'文献_相关原始论文_20261004'
POP=Path('C:/texlive/2025/bin/windows')
url='https://arxiv.org/pdf/2103.13415v3'
dest=LIB/'A52_MipNeRF_arXiv2103.13415v3.pdf'
assert not dest.exists()
with urllib.request.urlopen(url,timeout=45) as response:
    raw=response.read()
    assert raw[:5]==b'%PDF-'
    final_url=response.url
with dest.open('xb') as f:f.write(raw)
text=ROOT/'selected_text';text.mkdir(exist_ok=True)
subprocess.run([str(POP/'pdftotext.exe'),'-layout',str(dest),str(text/'MipNeRF_full_extraction_NOT_full_read.txt')],check=True)
pages=ROOT/'selected_pages';pages.mkdir(exist_ok=True)
mds=LIB/'FDRRA独立复核_20261006_MDS_DETR_arXiv2605.23507.pdf'
for pdf,page,prefix in [(mds,4,'MDS_p4'),(mds,5,'MDS_p5'),(dest,4,'MipNeRF_p4')]:
    subprocess.run([str(POP/'pdftoppm.exe'),'-f',str(page),'-l',str(page),'-singlefile','-scale-to','1700','-png',str(pdf),str(pages/prefix)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
receipt={'status':'PASS','url':url,'final_url':final_url,'pdf':str(dest),'bytes':len(raw),
 'sha256':hashlib.sha256(raw).hexdigest(),'full_pdf_downloaded':True,'full_pdf_read':False,
 'MDS_reused_pdf':str(mds),'MDS_sha256':hashlib.sha256(mds.read_bytes()).hexdigest(),
 'rendered_pages':['MDS_p4.png','MDS_p5.png','MipNeRF_p4.png'],
 'reading_scope':'Will be recorded after actual selected text and image inspection.'}
(ROOT/'primary_acquisition_v1.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False))

