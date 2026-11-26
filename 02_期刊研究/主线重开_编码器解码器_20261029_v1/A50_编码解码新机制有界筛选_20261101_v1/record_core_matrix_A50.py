from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent;LIB=ROOT.parent.parent/'文献_相关原始论文_20261004'
rows=[]
for name,version,url,scope,comparison in [
 ('A50_SaMer_arXiv2607.04605v3.pdf','arXiv2607.04605v3 2026-09-07','https://arxiv.org/abs/2607.04605v3','Abstract, section4.2/4.3; PDFp4 equations5-7 visually checked','Train-time instance labels guide merging already; generic instance-pure merge proposal held, not trained'),
 ('23_Uncertainty_aware_DETR__arXiv-2507.14855v1.pdf','arXiv2507.14855v1 2025-07-20','https://arxiv.org/abs/2507.14855v1','Sections3.1-3.3 and4.1/4.2; PDFp5 equations8-11 visually checked','Uncertainty-aware matching already exists; A50 exact bins versus same-moment Gaussian is a narrower proposed test, now stopped for weak activation'),
 ('105_D_FINE_ICLR2025_Publisher.pdf','ICLR2025 publisher PDF identified by watermark/hash; not equated to latest arXiv','https://openreview.net/forum?id=MFZjrTFE7h','Sections4.1/4.2 selected; PDFp4 equation2 visually checked','FDR/FGL/self-distillation and distribution-derived LQE are existing; source matcher uses point boxes, not all representation use is absent'),
 ('22_NWD_tiny-object__arXiv-2110.13389.pdf','embedded arXiv2110.13389v2 2022-06-14','https://arxiv.org/abs/2110.13389v2','Section3.2 and opening3.3; PDFp5 equations6-8 visually checked','Spatial Gaussian of deterministic boxes and normalizedW2 prior; not a predictive full marginal-edge distribution'),
 ('21_SA-Matching_DETR_CVPRFindings2026.pdf','CVPRFindings2026 publisher watermark','https://openaccess.thecvf.com/CVPR2026','Abstract and selected section3.2 text only; no formula visual audit','Scale-based GT duplication; landing URL not independently verified to exact article, do not use a guessed article URL')]:
    p=LIB/name
    rows.append({'pdf':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'version':version,
      'source_url':url,'reading_scope':scope,'comparison_and_limit':comparison,
      'human_read_attested':False,'full_paper_read':False})
j={'phase':'A50','scope':'targeted review, not systematic/exhaustive novelty proof','rows':rows,
 'own_inference':'same first two moments can lose shape information; no claim of a new probability/Wasserstein principle',
 'own_actual_result':'fixed64 activation:12/1720 small owners differQvsM, below frozen20-and1percent gate;0 optimizer updates, not an AP-negative training result'}
for p in [ROOT/'06_核心文献矩阵与阅读范围_v1.json',LIB/'A50_核心文献阅读范围_20261101_v1.json']:
    assert not p.exists();p.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'matrix_rows':5,'full_paper_read_claims':0,'research_goal_active':True}))
