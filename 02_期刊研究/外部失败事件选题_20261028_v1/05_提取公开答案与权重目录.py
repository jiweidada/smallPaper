from pathlib import Path
from html.parser import HTMLParser
import re
import json
import sys
from pypdf import PdfReader

sys.stdout.reconfigure(encoding='utf-8')
root = Path(__file__).resolve().parent
lib = root.parent / '文献_相关原始论文_20261004'
for name in ['CTAOD_CVPR2024_Yoo_WhatHowWhen.pdf', 'ContinualAdaptation_ICCV2025_Li_EnvironmentConditional.pdf']:
    reader = PdfReader(lib / name)
    extracted = '\n\n'.join(f'--- PAGE {i+1} ---\n' + (p.extract_text() or '') for i, p in enumerate(reader.pages))
    (lib / '_text' / (Path(name).stem + '.txt')).write_text(extracted.encode('utf-8', errors='replace').decode('utf-8'), encoding='utf-8')
    print(json.dumps({'paper': name, 'pages': len(reader.pages)}, ensure_ascii=False))

class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'): self.skip += 1
    def handle_endtag(self, tag):
        if tag in ('script', 'style'): self.skip = max(0, self.skip - 1)
    def handle_data(self, data):
        if not self.skip and data.strip(): self.parts.append(data.strip())

parser = TextParser()
parser.feed((root / 'Constellation_公开权重目录_20261028.html').read_text(encoding='utf-8'))
text = '\n'.join(parser.parts)
(root / 'Constellation_公开权重目录_可见文本.txt').write_text(text, encoding='utf-8')
print(json.dumps({'visible_weight_names': sorted(set(re.findall(r'[^\s<>"\\]+\.(?:pt|pth|onnx|engine)', text))), 'visible_text_preview': text[:300]}, ensure_ascii=False))
