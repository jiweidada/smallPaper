"""Read the current document in body order, including tables and equations."""
from pathlib import Path
from hashlib import sha256
from docx import Document
import json

HERE = Path(__file__).resolve().parent
DOC = HERE.parents[1] / '稿件/MPSMA-DEIM_conference_EN_revised.docx'

def main():
    assert sha256(DOC.read_bytes()).hexdigest() == '617ffd2024d42affb7f6b66f88cdca67204a0de3fef97946ffebf0c90f896984'
    document = Document(DOC)
    blocks = []
    for i, el in enumerate(document.element.body):
        tag = el.tag.rsplit('}', 1)[-1]
        if tag == 'p':
            text = ''.join(t.text or '' for t in el.xpath('.//w:t | .//m:t'))
            blocks.append({'index': i, 'kind': 'p', 'text': text})
        elif tag == 'tbl':
            rows = []
            for row in el.xpath('./w:tr'):
                rows.append([''.join(t.text or '' for t in cell.xpath('.//w:t | .//m:t')) for cell in row.xpath('./w:tc')])
            blocks.append({'index': i, 'kind': 'table', 'rows': rows})
    output = HERE / 'current_blocks.json'
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump({'manuscript_sha256': sha256(DOC.read_bytes()).hexdigest(), 'blocks': blocks}, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    for b in blocks:
        if b['kind'] == 'p' and b['text']:
            print(f"B{b['index']:03}: {b['text']}")
        elif b['kind'] == 'table':
            print(f"B{b['index']:03} TABLE:")
            for row in b['rows']:
                print(' | '.join(row))

if __name__ == '__main__':
    main()
