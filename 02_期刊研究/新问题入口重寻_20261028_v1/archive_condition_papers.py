import json, hashlib, time
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent.parent
LIB=ROOT/'文献_相关原始论文_20261004'
PAPERS=[
 ('20261028_新问题_Zone_Evaluation_arXiv2310.13215v2.pdf','https://arxiv.org/pdf/2310.13215v2','arxiv2310.13215v2; publication venue to verify independently'),
 ('20261028_新问题_Mind_the_Pad_ICLR2021_OpenReview.pdf','https://openreview.net/pdf?id=m1CD7tPubNy','Official OpenReview accepted ICLR2021 record'),
 ('20261028_新问题_Motion_Blur_CVPR2021_Publisher.pdf','https://openaccess.thecvf.com/content/CVPR2021/papers/Sayed_Improved_Handling_of_Motion_Blur_in_Online_Object_Detection_CVPR_2021_paper.pdf','Official CVF CVPR2021 PDF')]
result=[]
for name,url,label in PAPERS:
    row={'name':name,'url':url,'source_label':label,'read_scope':'NOT YET READ; download only'}
    p=LIB/name
    try:
        assert not p.exists(),p
        r=requests.get(url,timeout=35);r.raise_for_status();assert r.content.startswith(b'%PDF')
        p.write_bytes(r.content)
        row.update(status='DOWNLOADED',path=str(p),bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest(),final_url=r.url)
    except Exception as e:row.update(status='NOT_RETRIEVED',error=repr(e))
    result.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
out=LIB/'_sources_新问题条件下载_20261028_v1.json'
with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
