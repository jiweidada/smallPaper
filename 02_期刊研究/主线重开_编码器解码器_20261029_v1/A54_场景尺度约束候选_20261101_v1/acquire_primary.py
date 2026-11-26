from pathlib import Path
import hashlib,json,re,time,urllib.request
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader
P=Path(__file__).resolve().parent;ENG=P.parents[1];LIB=ENG/'文献_相关原始论文_20261004'
SOURCES=[
 ('MetaYOLO_WACV2026','https://openaccess.thecvf.com/content/WACV2026/papers/Yoon_Meta-YOLO_Metadata-Guided_Real-Time_Object_Detector_in_Aerial_Imagery_WACV_2026_paper.pdf'),
 ('TwoPlane_CVPR2023','https://openaccess.thecvf.com/content/CVPR2023/papers/Ghosh_Learned_Two-Plane_Perspective_Prior_Based_Image_Resampling_for_Efficient_Object_CVPR_2023_paper.pdf'),
 ('ClusDet_arXiv1904_08008','https://arxiv.org/pdf/1904.08008')]
def get(row):
    tag,url=row;dest=LIB/('A54_'+tag+'.pdf');out={'tag':tag,'url':url,'attempted_unix':time.time(),'reading_scope':'acquisition and text extraction only; not a reading attestation'}
    try:
        if dest.exists():b=dest.read_bytes();out['reused_existing']=True
        else:
            req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 Academic literature retrieval'})
            with urllib.request.urlopen(req,timeout=24) as r:b=r.read();out['final_url']=r.url;out['HTTP_status']=r.status
        if not b.startswith(b'%PDF-'):raise RuntimeError('Response is not a PDF')
        if not dest.exists():dest.write_bytes(b)
        reader=PdfReader(dest);text='\n\n'.join(f'--- PDF page {i+1} ---\n'+(p.extract_text() or '') for i,p in enumerate(reader.pages))
        tp=LIB/'_text'/('A54_'+tag+'.txt');tp.parent.mkdir(exist_ok=True);tp.write_text(text,encoding='utf-8')
        out.update(status='PDF_ACQUIRED_TEXT_EXTRACTED',path=str(dest),sha256=hashlib.sha256(b).hexdigest(),pages=len(reader.pages),text_path=str(tp),bytes=len(b),arxiv_ids=sorted(set(re.findall(r'arXiv:\s*([0-9.]+v[0-9]+)',text))))
    except Exception as e:out.update(status='NOT_ACQUIRED_OR_NOT_PARSED',error=str(e))
    return out
def main():
    assert LIB.is_dir(),LIB
    record=P/'01_原文获取记录_v1.json'
    if record.exists() and not(P/'01_路径修正前获取记录.json').exists():(P/'01_路径修正前获取记录.json').write_bytes(record.read_bytes())
    with ThreadPoolExecutor(max_workers=3) as ex:rows=list(ex.map(get,SOURCES))
    record.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([{k:r.get(k) for k in ['tag','status','pages','sha256','error']} for r in rows],ensure_ascii=False))
if __name__=='__main__':main()
