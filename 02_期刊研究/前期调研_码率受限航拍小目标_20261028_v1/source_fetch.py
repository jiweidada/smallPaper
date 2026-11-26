"""Read-only retrieval of public research documents; no model/data experiments."""
from pathlib import Path
import urllib.request, hashlib, json, re

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parent / '文献_相关原始论文_20261004'
SOURCES = {
    'JCRR_xml': 'https://api-journal.accscience.com/journal/article/export?doi=10.36922%2FDP026260029&exportType=Xml&id=8498',
    'JCRR_page': 'https://accscience.com/journal/DP/articles/online_first/8498',
    'JPEG_UAV_page': 'https://www.mdpi.com/2079-3197/14/9/206',
    'Pipeline_page': 'https://www.mdpi.com/2072-4292/14/21/5508',
    'RD_MIM_publisher': 'https://file.techscience.com/files/cmc/2024/TSP_CMC-80-3/TSP_CMC_52725/TSP_CMC_52725.pdf',
    'TransTIC_publisher': 'https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_TransTIC_Transferring_Transformer-based_Image_Compression_from_Human_Perception_to_Machine_ICCV_2023_paper.pdf',
    'Pipeline_arxiv': 'https://arxiv.org/pdf/2203.00306',
    'UAV_video_arxiv': 'https://arxiv.org/pdf/2301.06115',
}

def fetch(key, url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (research document retrieval)'})
    item = {'key': key, 'url': url, 'access_date': '2026-10-28'}
    try:
        with urllib.request.urlopen(req, timeout=35) as response:
            data = response.read()
            item.update(final_url=response.url, content_type=response.headers.get('Content-Type',''))
        ext = '.pdf' if data.startswith(b'%PDF') else ('.xml' if 'xml' in item['content_type'] else '.html')
        dest = (ARCHIVE / ('20261028_码率调研_' + key + ext)) if ext=='.pdf' else (ROOT / 'sources' / (key+ext))
        dest.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(data).hexdigest()
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()!=digest:
            raise ValueError('Existing different source preserved; choose a new version name')
        if not dest.exists(): dest.write_bytes(data)
        item.update(status='downloaded', path=str(dest), bytes=len(data), sha256=digest)
        if ext!='.pdf':
            txt=data.decode('utf-8',errors='replace')
            item['document_links'] = sorted(set(re.findall(r'https?[^\s\"\'<>]+(?:\.pdf|\.zip)',txt)))[:30]
    except Exception as exc:
        item.update(status='unavailable', error=str(exc))
    return item

if __name__=='__main__':
    receipts=[]
    for key,url in SOURCES.items():
        item=fetch(key,url)
        receipts.append(item)
        print(json.dumps(item,ensure_ascii=False),flush=True)
        (ROOT/'retrieval_receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8')
