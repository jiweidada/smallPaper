"""Read-only verification of delivered tables, references and unchanged evidence."""
import hashlib,json,sys
from pathlib import Path
from collections import Counter
from zipfile import ZipFile
from xml.etree import ElementTree as ET
sys.stdout.reconfigure(encoding='utf-8')
A=Path(__file__).resolve().parent
load=lambda n:json.loads((A/n).read_text(encoding='utf-8'))
D=load('direction_table_data.json');L=load('literature_table_data.json');S=load('assembled_summary.json')
manifest=load('delivery_manifest.json');f=Path(manifest['workbook'])
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with ZipFile(f) as z:
    strings=[]
    if 'xl/sharedStrings.xml' in z.namelist():
        strings=[''.join(s.itertext()) for s in ET.fromstring(z.read('xl/sharedStrings.xml'))]
    sheets=[]
    for number in [1,2]:
        xml=ET.fromstring(z.read(f'xl/worksheets/sheet{number}.xml'))
        cells={}
        for c in xml.findall('.//s:sheetData/s:row/s:c',ns):
            v=c.find('s:v',ns)
            if c.attrib.get('t')=='s':val=strings[int(v.text)]
            elif c.attrib.get('t')=='inlineStr':val=''.join(c.find('s:is',ns).itertext())
            else:val=v.text if v is not None else ''
            assert c.attrib.get('t')!='e',(c.attrib,val)
            cells[c.attrib['r']]=val
        pane=xml.find('s:sheetViews/s:sheetView/s:pane',ns).attrib
        assert pane.get('xSplit')=='3' and pane.get('ySplit')=='1',pane
        sheets.append(cells)
    assert len([k for k in sheets[0] if k.startswith('A') and k[1:].isdigit()])==74
    assert len([k for k in sheets[1] if k.startswith('A') and k[1:].isdigit()])==313
    for i,d in enumerate(D,2):
        for col,key in [('A','id'),('C','name'),('D','status'),('F','result'),('G','stop_reason'),('H','negative_scope')]:
            assert sheets[0][f'{col}{i}']==d[key],(i,col)
    for i,p in enumerate(L,2):
        for col,key in [('A','id'),('B','title'),('C','theme_name'),('D','availability'),('E','title_verification'),('F','historical_reading_grade')]:
            assert sheets[1][f'{col}{i}']==p[key],(i,col)
    table_ranges=[]
    for n,expected in [(1,'A1:S74'),(2,'A1:T313')]:
        t=ET.fromstring(z.read(f'xl/tables/table{n}.xml'))
        assert t.attrib['ref']==expected
        assert t.find('s:autoFilter',ns) is not None
        table_ranges.append(expected)

assert len({d['id'] for d in D})==73 and len({p['id'] for p in L})==312
assert sum(S['direction_status_counts'].values())==73
assert Counter(d['status'] for d in D)==S['direction_status_counts']
assert sum(S['pdf_work_themes'].values())==252
assert sum(S['pdf_work_reading_grades'].values())==252
assert S['main_library_pdf_files']==260 and S['unique_pdf_sha256']==255
assert sum(len(p['pdf_files']) for p in L)==260
assert len({v for p in L for v in p['pdf_sha256']})==255
assert all(Path(p).is_file() for r in L for p in r['pdf_files'])
missing_read_sources=sorted({r['source'] for p in L for r in p['historical_reading_evidence'] if not Path(r['source']).is_file()})
assert not missing_read_sources,missing_read_sources
hashes=load('document_evidence_hashes.json')
changed=[]
for name,sha in hashes.items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:changed.append(name)
assert not changed,changed
report=Path(manifest['conclusion']).read_text(encoding='utf-8')
assert ':57' in report and '未知158' in report and '75%' in report
result={'direction_rows':len(D),'literature_rows':len(L),'table_ranges':table_ranges,
        'native_filters':True,'frozen_rows':1,'frozen_columns':3,
        'all_record_ids_unique':True,'displayed_key_fields_match_json':True,
        'old_evidence_files_rehashed_and_unchanged':len(hashes),'missing_reading_source_files':missing_read_sources,
        'excel_error_cells':0,'workbook_bytes':f.stat().st_size,
        'workbook_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),
        'conclusion_sha256':hashlib.sha256(Path(manifest['conclusion']).read_bytes()).hexdigest(),
        'no_experiments_performed':True}
(A/'delivery_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
