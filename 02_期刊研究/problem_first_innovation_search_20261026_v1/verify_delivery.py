import json, hashlib, sys, zipfile, xml.etree.ElementTree as ET
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent
load=lambda name:json.loads((OUT/name).read_text(encoding='utf-8-sig'))
tables=load('research_tables_data.json'); registry=load('evidence_registry.json'); cards=load('problem_cards_resolved.json')
for s in registry['sources']:
    assert hashlib.sha256(Path(s['path']).read_bytes()).hexdigest()==s['sha256'],s['path']
for s in load('initial_source_hashes.json'):
    assert hashlib.sha256(Path(s['path']).read_bytes()).hexdigest()==s['sha256'],s['path']
assert [c['id'] for c in cards if c['priority'].split()[0] in ['1','2','3']]==['Q01','Q02','Q03']
assert len(cards)==10
legacy=next(t for t in tables['tables'] if t['name']=='旧方案适用范围')
assert [r[0] for r in legacy['rows']]==[f'D{i:03d}' for i in range(1,74)]
assert '未完成' in next(r for r in legacy['rows'] if r[0]=='D035')[3]
refs={s['id'] for s in load('literature_comparison.json')['sources']}
assert len(refs)==22 and 'R41' in refs and 'R62' not in refs and 'R64' in refs
for c in cards:assert set(c['literature_ids'])<=refs

book=OUT/'373_问题优先研究_证据与文献对照.xlsx'
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
compared=0; sheet_results=[]
with zipfile.ZipFile(book) as z:
    shared=[]
    if 'xl/sharedStrings.xml' in z.namelist():
        shared=[''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)]
    for ix,t in enumerate(tables['tables'],1):
        root=ET.fromstring(z.read(f'xl/worksheets/sheet{ix}.xml'))
        found={}
        for c in root.findall('s:sheetData/s:row/s:c',ns):
            v=c.find('s:v',ns)
            if c.get('t')=='s':value=shared[int(v.text)]
            elif c.get('t')=='inlineStr':value=''.join(c.find('s:is',ns).itertext())
            elif v is not None:value=v.text
            else:value=''
            found[c.get('r')]=value
        def col(j):
            s='';n=j+1
            while n:s=chr(65+(n-1)%26)+s;n=(n-1)//26
            return s
        for ri,row in enumerate([t['headers'],*t['rows']],4):
            for ci,value in enumerate(row):
                address=f'{col(ci)}{ri}'
                assert found.get(address,'')==str(value), (t['name'],address,str(value)[:80],str(found.get(address))[:80])
                compared+=1
        panes=root.find('s:sheetViews/s:sheetView/s:pane',ns)
        tableparts=root.findall('s:tableParts/s:tablePart',ns)
        assert panes is not None and len(tableparts)==1,t['name']
        sheet_results.append({'sheet':t['name'],'data_rows':len(t['rows']),'values_verified':True,'freeze_panes':True,'native_table':True})

# Arithmetic only on already-published aggregate metrics; no prediction replay/evaluation.
root_path=OUT.parent.parent
nms=json.loads((root_path/'英文小论文/code/postprocess_nms_control_20261026_v1/result.json').read_text(encoding='utf-8'))
checks=[]
for seed,tiny_delta in [('1',4),('2',-2)]:
    p=nms['per_seed'][seed];b,a=p['B0'],p['B1']
    assert a['tiny_TP50']-b['tiny_TP50']==tiny_delta
    assert abs((a['stats12']['AP']-b['stats12']['AP'])-a['delta_stats12']['AP'])<1e-14
    checks.append({'seed':seed,'NMS_AP_pp':a['delta_stats12']['AP']*100,'tiny_TP_delta':tiny_delta,'FP_delta':a['FP50']-b['FP50']})
report=(OUT/'02_综合判断与可证伪设想.txt').read_text(encoding='utf-8')
assert '四臂共同没学会补框' not in report
assert '第94–96轮' in report and '不是独立.75 TP' in report
assert '尚不能证明M1不只是已有组件拼接' in report
inspection=load('workbook_inspection.json');assert inspection['qa']==[]
assert 'matched 0 entries' in inspection['errors']
outputs=['00_简洁结论.txt','02_综合判断与可证伪设想.txt','03_文献与已有办法_逐项说明.txt','04_证据补充与口径更正.txt',book.name]
for f in outputs:assert (OUT/f).is_file() and (OUT/f).stat().st_size>0
manifest={'scope':'373全部三项交付；仅元数据、文件和已有聚合数值核对，不作方法有效性证明','requirements':{'ten_families_and_73_directions':True,'three_ordered_stages':True,'ten_problem_cards_three_deep':True,'observations_competing_explanations_and_boundaries':True,'primary_sources_and_reading_scope':True,'known_solutions_local_three_statuses':True,'search_records_and_failures':True,'conditional_falsifiable_method_with_counterevidence':True,'no_experiments_no_phase_redirect':True},'content_counts':tables['metadata'],'workbook_checks':{'cells_compared_to_saved_source_matrices':compared,'sheets':sheet_results,'formula_error_matches':0,'preview_files':[f'preview_{i}.png' for i in range(1,6)],'visual_review':'all five sheets and selected right-side/detail ranges reviewed in parent conversation'},'local_source_hashes_unchanged_since_assembly':len(registry['sources']),'earlier_read_hash_covered_sources':len([s for s in registry['sources'] if s['read_hashes']]),'aggregate_arithmetic':checks,'review_resolutions':{'F01':'原TP零值不识别共同学习失败，已更正','F02':'零基epoch明确，已更正','F03':'固定配对.75非独立TP，已更正'},'limitations':['没有重新复现73条训练或重放历史预测；逐条保留阅读强度','21个论文身份中部分仅摘要/局部原文，不声称全部全文精读','M1不是已经成立的新方法，无新颖性/收益证明','定向检索非穷尽系统综述'],'outputs':[{'path':str(OUT/f),'bytes':(OUT/f).stat().st_size,'sha256':hashlib.sha256((OUT/f).read_bytes()).hexdigest()} for f in outputs]}
(OUT/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
state=load('run_state.json');state['stage']='complete';state['coverage']={k:'complete; see partition evidence and 73-row boundary table' for k in state['coverage']};state['checkpoints']['synthesis']='Completed; read_scope and competing explanations retained';state['checkpoints']['final']='Completed; F01/F02/F03 resolved, workbook export/readback/visual verification completed';state['delivery']='delivery_manifest.json';state['method_status']='M1 conditional, not a validated innovation; no execution authorized by delivery';
(OUT/'run_state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
verification=load('content_verification.json');verification['workbook']='exported, reopened, exact cell readback verified, visually reviewed';verification['earlier_read_hash_covered_sources']=manifest['earlier_read_hash_covered_sources'];
(OUT/'content_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'completed':True,'workbook_cells_verified':compared,'counts':tables['metadata'],'outputs':[f for f in outputs]},ensure_ascii=False))
