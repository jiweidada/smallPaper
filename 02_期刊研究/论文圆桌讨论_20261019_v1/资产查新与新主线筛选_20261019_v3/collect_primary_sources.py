"""Archive primary papers and extract page-located text. No detector or SSH use."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib, json, urllib.request
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
LIB=ROOT/'文献_相关原始论文_20261004'
OUT=Path(__file__).resolve().parent
SOURCES=[
 ('P01','DEIM (CVPR 2025)','14_DEIM_CVPR2025_improved-matching.pdf','https://arxiv.org/abs/2412.04234',None),
 ('P02','D-FINE (ICLR 2025)','105_D_FINE_ICLR2025_Publisher.pdf','https://arxiv.org/abs/2410.13842',None),
 ('P03','GFLv2 (CVPR 2021)','06_GFLv2_CVPR2021_distribution-statistics_LQE__arXiv-2011.12885.pdf','https://arxiv.org/abs/2011.12885',None),
 ('P04','Rank-DETR (NeurIPS 2023), archived v1','109_Rank_DETR__arXiv-2310.08854v1.pdf','https://arxiv.org/abs/2310.08854v1',None),
 ('P05','TIDE (ECCV 2020), author appendix-E update','236_TIDE_ECCV2020_author_updated_appendixE.pdf','https://dbolya.com/tide/',None),
 ('P06','SNIP (CVPR 2018)','221_SNIP_CVPR2018_Publisher.pdf','https://openaccess.thecvf.com/content_cvpr_2018/html/Singh_An_Analysis_of_CVPR_2018_paper.html',None),
 ('P07','NWD, archived 2021 preprint','22_NWD_tiny-object__arXiv-2110.13389.pdf','https://arxiv.org/abs/2110.13389',None),
 ('P08','DQ-DETR (ECCV 2024), archived v4','33_DQ-DETR_ECCV2024__arXiv-2404.03507v4.pdf','https://arxiv.org/abs/2404.03507v4',None),
 ('P09','RT-DETRv3, archived v1','专家组_20261007_RTDETRv3_WACV2025_arXiv2409.08475v1.pdf','https://arxiv.org/abs/2409.08475v1',None),
 ('P10','Scale Match (WACV 2020)','新主线筛选_20261019_Scale_Match_WACV2020.pdf','https://openaccess.thecvf.com/content_WACV_2020/html/Yu_Scale_Match_for_Tiny_Person_Detection_WACV_2020_paper.html','https://openaccess.thecvf.com/content_WACV_2020/papers/Yu_Scale_Match_for_Tiny_Person_Detection_WACV_2020_paper.pdf'),
 ('P11','Scale-aware AutoAug (CVPR 2021)','新主线筛选_20261019_ScaleAware_AutoAug_CVPR2021.pdf','https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Scale-Aware_Automatic_Augmentation_for_Object_Detection_CVPR_2021_paper.pdf','https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Scale-Aware_Automatic_Augmentation_for_Object_Detection_CVPR_2021_paper.pdf'),
 ('P12','Multivariate Confidence Calibration (CVPRW 2020)','新主线筛选_20261019_Multivariate_Calibration_CVPRW2020.pdf','https://arxiv.org/abs/2004.13546','https://openaccess.thecvf.com/content_CVPRW_2020/papers/w20/Kuppers_Multivariate_Confidence_Calibration_for_Object_Detection_CVPRW_2020_paper.pdf'),
 ('P13','Cal-DETR (NeurIPS 2023)','新主线筛选_20261019_Cal_DETR_arxiv2311_03570v1.pdf','https://arxiv.org/abs/2311.03570v1','https://arxiv.org/pdf/2311.03570v1'),
 ('P14','LiG-DETR (2026-10-07 preprint, not claimed peer-reviewed)','新主线筛选_20261019_LiG_DETR_arxiv2610_09511v1.pdf','https://arxiv.org/abs/2610.09511v1','https://arxiv.org/pdf/2610.09511v1'),
]

def collect(item):
 pid,title,name,url,pdfurl=item
 p=LIB/name
 record={'id':pid,'title':title,'url':url,'pdf_url':pdfurl,'path':str(p),'checked_at_utc':datetime.now(timezone.utc).isoformat()}
 try:
  if not p.exists():
   assert pdfurl
   req=urllib.request.Request(pdfurl,headers={'User-Agent':'Mozilla/5.0 academic-source-verification'})
   with urllib.request.urlopen(req,timeout=35) as r: raw=r.read()
   assert raw.startswith(b'%PDF'), 'Response is not a PDF'
   p.write_bytes(raw)
   record['download']='downloaded_new'
  else: record['download']='existing_file_reused'
  raw=p.read_bytes()
  d=PdfReader(p)
  pages=[page.extract_text() or '' for page in d.pages]
  dest=LIB/'_text'/f'新主线筛选_20261019_{pid}.txt'
  text='\n\n'.join(f'===== PDF PAGE {i+1} =====\n{t}' for i,t in enumerate(pages))
  if not dest.exists(): dest.write_text(text,encoding='utf-8')
  else: assert dest.read_text(encoding='utf-8')==text
  record.update(status='pdf_archived_and_text_extracted',sha256=hashlib.sha256(raw).hexdigest(),pages=len(pages),text_path=str(dest),text_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),reading_scope='extracted; specific passages to be reviewed in source_notes, not a claim that all pages were read')
 except Exception as e: record.update(status='unavailable',error=type(e).__name__+': '+str(e))
 print(json.dumps({k:record.get(k) for k in ['id','status','pages','error']},ensure_ascii=False),flush=True)
 return record

if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=3) as pool: rows=list(pool.map(collect,SOURCES))
 (OUT/'primary_sources_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
