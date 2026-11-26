"""Read only the explanation for a public annotation-file response; follow no forms."""
import json
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

out = Path(__file__).parent / 'smot_source_metadata_20261029'
url = 'https://drive.google.com/uc?export=download&id=15bgHpeHdT1AOR7pezWZ7YpMse4zvGnCt'
class Visible(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []
        self.links = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ('script', 'style'):
            self.skip += 1
        if tag == 'a' and a.get('href'):
            self.links.append(a['href'])
    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.skip = max(0, self.skip - 1)
    def handle_data(self, data):
        if not self.skip and data.strip():
            self.parts.append(data.strip())
r = {'url': url, 'form_or_security_confirmation_followed': False}
try:
    with urllib.request.urlopen(url, timeout=12) as resp:
        r['status'] = resp.status
        r['content_type'] = resp.headers.get('Content-Type')
        payload = resp.read(5_000_001)
        r['bytes'] = len(payload)
        if payload[:4] == b'PK\x03\x04' and len(payload) <= 5_000_000:
            (out / 'annotations.zip').write_bytes(payload)
            r['result'] = 'source_zip_saved_no_code_execution'
        elif 'html' in (r['content_type'] or '') and len(payload) < 100_000:
            p = Visible()
            p.feed(payload.decode('utf-8', errors='replace'))
            r['visible_explanation'] = '\n'.join(p.parts)
            r['visible_links'] = p.links
            r['result'] = 'explanation_only_no_form_followed'
        else:
            r['result'] = 'stopped_unexpected_response'
except Exception as exc:
    r['result'] = 'request_failed'
    r['error'] = type(exc).__name__ + ': ' + str(exc)
(out / 'annotation_response_explanation.json').write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(r, ensure_ascii=False, indent=2))
