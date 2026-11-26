from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent;LIB=ROOT.parent.parent/'文献_相关原始论文_20261004'
entries=[]
for name,scope,page,claim in [
 ('A51_mHC_arXiv2512.24880v1.pdf','Introduction and section4.1, opening4.2; not infrastructure/experiments/full proof audit',8,'Doubly stochastic residual-stream mixing is prior; no first mHC-to-detection claim'),
 ('A51_FreqFusion_arXiv2408.12879v1.pdf','Section4/4.1 overview, figures3/4 and equations4/5; not all adaptive-filter details/experiments',5,'Adaptive low/high-pass and resampling fusion are prior; not equivalent to a proof of new paired cross-resolution transport'),
 ('A51_RevBiFPN_arXiv2206.14098v1.pdf','Section2 and opening3; not appendices/full implementation/training audit',2,'Reversible bidirectional multiscale feature fusion is prior; A51 does not claim reversibility/memory savings'),
 ('A51_NeuralSheafDiffusion_arXiv2202.04579v1.pdf','Selected orthogonal-restriction-map text and section6.1/opening6.2; not complete theorem/experiment audit',6,'Learned orthogonal transport and diffusion are prior; cross-grid D/U implementation is a task-specific proposed increment, not a new diffusion theory')]:
    p=LIB/name;source=json.loads(p.with_suffix('.pdf.source.json').read_text(encoding='utf-8'))
    entries.append({'title':source['title'],'version':source['version'],'source_url':source['source_url'],
      'pdf':str(p),'pdf_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':source['text'],
      'reading_scope':scope,'visually_verified_pdf_page1':page,'original_result_or_operation':claim,
      'assistant_reading_only':True,'human_read_attested':False,'full_paper_read':False})
entries.append({'title':'mHC-DEIM: Object Detection for Shelf Scenes under Dense Arrangement, Multi-Angle Inclination and Low-Illumination',
 'source_url':'https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6557880',
 'reading_scope':'previous A28 primary indexed abstract only; full paper unavailable; not peer-reviewed venue assertion',
 'existing_record':str(LIB/'A28_mHC_DEIM_public_access.source.json'),
 'consequence':'exclude generic first-transfer claim; exact operator overlap remains unknown',
 'pdf_obtained':False,'full_paper_read':False})
registry={'phase':'A51','review_type':'targeted prior check, not systematic review or exhaustive novelty guarantee',
 'entries':entries,'our_inference':'A jointly constrained cross-grid exchange is distinguishable from inspected operators; it remains an incremental engineering hypothesis',
 'hypothesis_not_result':'better VisDrone small AP than native and matched independent gates; requires fixed12 validation',
 'unverified_discovery_only':[{'title':'Infrared Small Object Detection Based on Spatial-Frequency Enhancement and Manifold-Constrained Multi-Residual Redistribution',
   'claimed_DOI_in_secondary_index':'10.1109/TGRS.2026.3744667','status':'primary full text/metadata not verified, no method equivalence assertion'}]}
for p in [ROOT/'05_核心文献矩阵与阅读范围_v1.json',LIB/'A51_核心方法阅读范围_20261101_v1.json']:
    assert not p.exists();p.write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'archived_primary_pdfs':4,'selected_method_reading_only':True,'abstract_only':1,'unverified_secondary':1}))
