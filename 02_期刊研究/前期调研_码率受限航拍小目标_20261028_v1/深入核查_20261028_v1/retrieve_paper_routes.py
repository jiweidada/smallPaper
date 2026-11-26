from retrieve_static import retrieve, ROOT
import json
sources={
 'PDAD_accepted_reference':'https://www.nature.com/articles/s41598-026-63984-0_reference.pdf',
 'FSIC_linkinghub':'https://linkinghub.elsevier.com/retrieve/pii/S1051200424004470',
 'FSIC_publisher_public_xml':'https://api.elsevier.com/content/article/pii/S1051200424004470?httpAccept=text/xml',
}
rows=[]
for key,url in sources.items():
 r=retrieve(key,url);rows.append(r)
 (ROOT/'paper_route_receipts.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({k:r.get(k) for k in ['key','status','path','bytes','is_pdf','error']},ensure_ascii=True),flush=True)
