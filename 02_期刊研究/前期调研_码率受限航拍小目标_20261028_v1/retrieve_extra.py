from source_fetch import fetch, ROOT
import json
sources = {
 'JCRR_publisher':'https://article.accscience.com/journal/DP/3/3/dp026260029/article_dp0262600291_ce506a.pdf',
 'JPEG_UAV_publisher':'https://www.mdpi.com/2079-3197/14/9/206/pdf',
 'JPEG_UAV_resource':'https://mdpi-res.com/d_attachment/computation/computation-14-00206/article_deploy/computation-14-00206.pdf',
 'KD_compressed_publisher':'https://www.mdpi.com/2313-433X/12/7/325/pdf',
 'RD_MIM_pdf_retry':'https://file.techscience.com/files/cmc/2024/TSP_CMC-80-3/TSP_CMC_52725/TSP_CMC_52725.pdf',
}
receipts=[]
for key,url in sources.items():
 r=fetch(key,url); receipts.append(r)
 print(json.dumps(r,ensure_ascii=True),flush=True)
 (ROOT/'retrieval_extra.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8')
