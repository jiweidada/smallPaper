from source_fetch import fetch, ROOT
import json
sources={
 'RD_MIM_official_CDN':'https://cdn.techscience.cn/files/cmc/2024/TSP_CMC-80-3/TSP_CMC_52725/TSP_CMC_52725.pdf',
 'FGD_publisher':'https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.pdf',
 'UAVDT_publisher':'https://openaccess.thecvf.com/content_ECCV_2018/papers/Dawei_Du_The_Unmanned_Aerial_ECCV_2018_paper.pdf',
 'Michaelis_arxiv':'https://arxiv.org/pdf/1907.07484',
 'FSIC_publisher_preview':'https://www.sciencedirect.com/science/article/abs/pii/S1051200424004470',
}
receipts=[]
for key,url in sources.items():
 r=fetch(key,url); receipts.append(r)
 print(json.dumps(r,ensure_ascii=True),flush=True)
 (ROOT/'retrieval_tail.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8')
