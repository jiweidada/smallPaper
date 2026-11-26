"""Archive cited public papers; never overwrite an existing original."""
from pathlib import Path
import concurrent.futures, hashlib, json, urllib.request
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
ARCH=ROOT/'03_文献资料/原始论文_20261004'
TXT=ARCH/'_text'; TXT.mkdir(exist_ok=True)
sources=[
 (1,'MPSMA_ref01_Tang_UAV_Survey_RS2024_20261103.pdf','https://www.mdpi.com/2072-4292/16/1/149/pdf','Publisher 2024'),
 (3,'14_DEIM_CVPR2025_improved-matching.pdf',None,'CVPR 2025 local original'),
 (4,'105_D_FINE_ICLR2025_Publisher.pdf',None,'ICLR 2025 publisher original'),
 (5,'MPSMA_ref05_YOLOv10_NeurIPS2024_20261103.pdf','https://proceedings.neurips.cc/paper_files/paper/2024/file/c34ddd05eb089991f06f3c5dc36836e0-Paper-Conference.pdf','NeurIPS 2024'),
 (6,'MPSMA_ref06_CenterMask_CVPR2020_20261103.pdf','https://openaccess.thecvf.com/content_CVPR_2020/papers/Lee_CenterMask_Real-Time_Anchor-Free_Instance_Segmentation_CVPR_2020_paper.pdf','CVPR 2020'),
 (7,'SimAM_ICML2021_原文_20261102.pdf',None,'ICML 2021 local original'),
 (8,'MPSMA_ref08_SiLU_arXiv1702.03118v3_20261103.pdf','https://arxiv.org/pdf/1702.03118v3','arXiv v3 2017'),
 (9,'MPSMA_ref09_VisDrone_DET2019_ICCVW_20261103.pdf','https://openaccess.thecvf.com/content_ICCVW_2019/papers/VISDrone/Du_VisDrone-DET2019_The_Vision_Meets_Drone_Object_Detection_in_Image_Challenge_ICCVW_2019_paper.pdf','ICCV Workshops 2019'),
 (10,'MPSMA_ref10_COCO_arXiv1405.0312v1_20261103.pdf','https://arxiv.org/pdf/1405.0312v1','arXiv v1; published citation metadata checked separately')
]

def archive(item):
    ref,name,url,version=item; p=ARCH/name
    r={'reference':ref,'pdf':str(p),'source_url':url,'version':version,'reading_scope':'Targeted claim/metadata verification, not a claim of complete full-text reading'}
    try:
        if p.exists():r['download_status']='existing original reused'
        else:
            if not url:raise FileNotFoundError('Expected existing original is absent')
            req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req,timeout=20) as response:
                b=response.read(35_000_001); r['resolved_url']=response.url
            if len(b)>35_000_000:raise ValueError('Download exceeds 35 MB bound')
            if not b.startswith(b'%PDF-'):raise ValueError('Response is not a PDF')
            with p.open('xb') as f:f.write(b)
            r['download_status']='downloaded'
        r['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();r['bytes']=p.stat().st_size
        t=TXT/(p.stem+'.txt')
        if not t.exists():
            reader=PdfReader(str(p))
            t.write_text('\n\n'.join(f'PAGE {i+1}\n'+(page.extract_text() or '') for i,page in enumerate(reader.pages)),encoding='utf-8')
        r['text']=str(t);r['text_sha256']=hashlib.sha256(t.read_bytes()).hexdigest()
    except Exception as e:r['error']=str(e);r['download_status']='unavailable' if not p.exists() else r.get('download_status','existing')
    return r

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(archive,sources))
records.append({'reference':2,'source_url':'https://xplorestaging.ieee.org/document/10640832/','version':'IGARSS 2024','download_status':'No accessible full-text PDF located; publisher abstract and conference program metadata verified','reading_scope':'Abstract/metadata only; no claim of full-text reading'})
records.sort(key=lambda r:r['reference'])
(OUT/'citation_archive.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
for r in records: print(json.dumps({'reference':r['reference'],'status':r['download_status'],'error':r.get('error')},ensure_ascii=True),flush=True)
