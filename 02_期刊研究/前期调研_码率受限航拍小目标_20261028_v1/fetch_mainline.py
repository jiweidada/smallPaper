from source_fetch import fetch, ROOT
import json
sources={
 'LAD_WACV2022':'https://openaccess.thecvf.com/content/WACV2022/papers/Nguyen_Improving_Object_Detection_by_Label_Assignment_Distillation_WACV_2022_paper.pdf',
 'PDAD_SciRep2026':'https://www.nature.com/articles/s41598-026-63984-0.pdf',
 'GID_CVPR2021':'https://openaccess.thecvf.com/content/CVPR2021/papers/Dai_General_Instance_Distillation_for_Object_Detection_CVPR_2021_paper.pdf',
 'Drone_degradation_Wiley2026':'https://onlinelibrary.wiley.com/doi/full/10.1002/eng2.70597',
}
out=[]
for key,url in sources.items():
 item=fetch(key,url); out.append(item)
 print(json.dumps(item,ensure_ascii=True),flush=True)
 (ROOT/'retrieval_mainline.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
