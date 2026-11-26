import json, hashlib, zipfile, xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(r'C:\Users\jiwei\Desktop\小论文投稿')
OUT = Path(__file__).resolve().parent
AUDIT = ROOT/'英文小论文/研究历史与文献审计_20261026_v1'
directions = json.loads((AUDIT/'direction_table_data.json').read_text(encoding='utf-8'))
literature = json.loads((AUDIT/'literature_table_data.json').read_text(encoding='utf-8'))
book = ROOT/'outputs/research_history_literature_audit_20261026_v1/372_研究方向与文献覆盖审计_20261026_v1_两张明细表.xlsx'
ns = {'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(book) as z:
    shared = []
    if 'xl/sharedStrings.xml' in z.namelist():
        shared = [''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)]
    workbook = ET.fromstring(z.read('xl/workbook.xml'))
    sheets = []
    for ix, sheet in enumerate(workbook.findall('s:sheets/s:sheet',ns),1):
        content = ET.fromstring(z.read(f'xl/worksheets/sheet{ix}.xml'))
        values = []
        for row in content.findall('s:sheetData/s:row',ns):
            cells = []
            for c in row.findall('s:c',ns):
                v = c.find('s:v',ns)
                if c.get('t') == 's' and v is not None:
                    val = shared[int(v.text)]
                elif c.get('t') == 'inlineStr':
                    val = ''.join(c.find('s:is',ns).itertext())
                else:
                    val = v.text if v is not None else ''
                cells.append([c.get('r'),val])
            if cells:
                values.append(cells)
        ids = [v for row in values for _,v in row if v in {d['id'] for d in directions} or v in {l['id'] for l in literature}]
        sheets.append({'name':sheet.get('name'),'nonempty_rows':len(values),'first_rows':values[:4], 'matched_ids':ids})
snapshot = {'source':str(book),'sha256':hashlib.sha256(book.read_bytes()).hexdigest(),'read_method':'read-only ZIP/XML; no workbook edits or re-export','sheets':sheets,'direction_json_rows':len(directions),'literature_json_rows':len(literature)}
(OUT/'input_workbook_verification.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'workbook':str(book),'sheets':[{'name':s['name'],'nonempty_rows':s['nonempty_rows'],'matched_ids':len(s['matched_ids'])} for s in sheets],'direction_rows':len(directions),'literature_rows':len(literature)},ensure_ascii=False))
ids = ['L001','L006','L007','L015','L019','L052','L057','L113','L115','L124','L133','L136','L204','L206','L214','L220','L221','L229','L238','L241','L248','L271']
pool = [{k:l.get(k) for k in ['id','title','urls','pdf_files','pdf_record_numbers','historical_reading_grade','historical_reading_evidence','documented_project_uses','comparison_evidence']} for l in literature if l['id'] in ids]
(OUT/'local_literature_pool.json').write_text(json.dumps(pool,ensure_ascii=False,indent=2),encoding='utf-8')
sources = [AUDIT/'direction_table_data.json',AUDIT/'literature_table_data.json',book,ROOT/'英文小论文/持续寻找研究主线_20261020_v2/373_问题优先寻找创新点_提示词_20261026.txt']
(OUT/'initial_source_hashes.json').write_text(json.dumps([{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],ensure_ascii=False,indent=2),encoding='utf-8')
