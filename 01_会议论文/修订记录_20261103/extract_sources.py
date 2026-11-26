from pathlib import Path
from docx import Document
from zipfile import ZipFile
import json, hashlib

root=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent
for name in ['MPSMA-DEIM_conference_EN','MPSMA-DEIM_修订稿_真实数据版']:
    p=root/'01_会议论文'/'稿件'/f'{name}.docx'
    d=Document(p)
    blocks=[]
    for i,el in enumerate(d.element.body):
        tag=el.tag.rsplit('}',1)[-1]
        t=''.join(el.itertext()) if False else ''.join(el.xpath('.//w:t | .//m:t')) if False else '\n'.join(''.join(t.itertext()) for t in [])
        if tag=='p':
            texts=el.xpath('.//w:t | .//m:t')
            text=''.join(t.text or '' for t in texts)
            rels=el.xpath('.//a:blip/@r:embed')
            blocks.append({'id':i,'kind':'p','text':text,'images':rels,'xml':el.xml})
        elif tag=='tbl':
            rows=[]
            for row in el.xpath('./w:tr'):
                rows.append([''.join(t.text or '' for t in cell.xpath('.//w:t | .//m:t')) for cell in row.xpath('./w:tc')])
            blocks.append({'id':i,'kind':'table','rows':rows,'xml':el.xml})
    (out/f'{name}.json').write_text(json.dumps(blocks,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=[]
    for b in blocks:
        if b['kind']=='p': lines.append(f"B{b['id']:03d}: {b['text']}"+(' [IMAGE '+','.join(b['images'])+']' if b['images'] else ''))
        else: lines.append(f"B{b['id']:03d} TABLE:\n"+'\n'.join(' | '.join(r) for r in b['rows']))
    (out/f'{name}_extracted.txt').write_text('\n'.join(lines),encoding='utf-8')
    print(name, 'blocks',len(blocks), 'sha256',hashlib.sha256(p.read_bytes()).hexdigest())
