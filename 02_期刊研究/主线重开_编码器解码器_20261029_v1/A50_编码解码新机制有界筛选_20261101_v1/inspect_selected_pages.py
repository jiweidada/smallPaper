from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
specs=[('23_Uncertainty_aware_DETR__arXiv-2507.14855v1.txt',['Matching Cost.','4.2','Bayes Risk']),
 ('105_D_FINE_ICLR2025_Publisher.txt',['Hungarian Matching','3.2']),
 ('22_NWD_tiny-object__arXiv-2110.13389.txt',['3.2.','Wasserstein Distance']),
 ('21_SA-Matching_DETR_CVPRFindings2026.txt',['3.2. Scale Adaptive Matching']),
 ('A50_SaMer_arXiv2607.04605v3.txt',['3.','instance','prior'])]
for name,needles in specs:
    p=ROOT/'selected_text'/name
    pages=p.read_text(encoding='utf-8').split('\f')
    print(json.dumps({'file':name,'pages_with_matches':[
       {'page1':i+1,'matched':[n for n in needles if n in t]} for i,t in enumerate(pages)
       if any(n in t for n in needles)]},ensure_ascii=False))
    if name.startswith('23_'):
        txt=p.read_text(encoding='utf-8');k=txt.index('    Matching Cost.')
        print(txt[k:k+2000])
    if name.startswith('A50_'):
        txt=p.read_text(encoding='utf-8')
        for marker in ['object-aware','Object-Aware','instance-aware','Instance-Aware','merge prior','supervision']:
            k=txt.find(marker)
            if k>=0: print(marker,txt[max(0,k-150):k+650])
