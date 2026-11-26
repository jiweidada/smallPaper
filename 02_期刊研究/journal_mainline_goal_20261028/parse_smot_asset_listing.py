import argparse
import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', default='public_folder')
args = parser.parse_args()
out = Path(__file__).parent / 'smot_source_metadata_20261029'
receipt_file = out / (args.prefix + '_receipt.json')
receipt = json.loads(receipt_file.read_text(encoding='utf-8'))
receipt['headers'] = {k: v for k, v in receipt.get('headers', {}).items() if k in ('Content-Type', 'Content-Length', 'Last-Modified')}
receipt_file.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
listing_path = out / (args.prefix + '_listing.html')
if receipt.get('status') != 'listing_saved_not_yet_interpreted' or not listing_path.is_file():
    print(json.dumps({'status': 'not_parsed_no_verified_listing', 'prefix': args.prefix,
                      'source_status': receipt.get('status')}, ensure_ascii=False))
    raise SystemExit(0)

class Visible(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.rows = []
        self.texts = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ('script', 'style'):
            self.skip += 1
        if a.get('role') == 'row' or 'data-id' in a:
            self.rows.append({k: a[k] for k in ('role', 'data-id', 'aria-label', 'data-tooltip') if k in a})
    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.skip = max(0, self.skip - 1)
    def handle_data(self, data):
        if not self.skip and data.strip():
            self.texts.append(data.strip())

body = listing_path.read_text(encoding='utf-8')
p = Visible()
p.feed(body)
text = '\n'.join(p.texts)
(out / (args.prefix + '_visible.txt')).write_text(text, encoding='utf-8')
(out / (args.prefix + '_rows.json')).write_text(json.dumps(p.rows, ensure_ascii=False, indent=2), encoding='utf-8')
print(text[:6500])
print('\nROWS:', json.dumps(p.rows[:25], ensure_ascii=False))
