"""Inspect public download-page text without saving cookies or hidden token values."""
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import HTTPCookieProcessor, Request, build_opener
from urllib.error import HTTPError
import json
import re

HERE = Path(__file__).resolve().parent
OUTPUT = HERE/'official_val_http_diagnostic.json'
PUBLIC_ID = '1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59'
URL = 'https://drive.google.com/uc?export=download&id=' + PUBLIC_ID

class Parser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.forms, self.current, self.visible, self.hidden = [], None, [], 0
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {'script', 'style', 'noscript'}:
            self.hidden += 1
        if tag == 'form':
            self.current = {'action': attrs.get('action', ''), 'method': attrs.get('method', 'get'), 'fields': {}}
            self.forms.append(self.current)
        if tag == 'input' and self.current and attrs.get('name'):
            self.current['fields'][attrs['name']] = attrs.get('value', '')
    def handle_endtag(self, tag):
        if tag in {'script', 'style', 'noscript'} and self.hidden:
            self.hidden -= 1
        if tag == 'form':
            self.current = None
    def handle_data(self, data):
        if not self.hidden:
            self.visible.append(data)

def safe_host(url):
    host = (urlsplit(url).hostname or '').lower()
    return host in {'drive.google.com', 'drive.usercontent.google.com', 'docs.google.com'} or host.endswith('.googleusercontent.com')

def get(opener, url):
    assert safe_host(url)
    with opener.open(Request(url, headers={'User-Agent':'Academic data source provenance audit'}), timeout=20) as response:
        assert safe_host(response.geturl())
        prefix = response.read(65537)
        return {'http_status': response.status, 'content_type': response.headers.get('Content-Type'),
                'prefix_sha256': sha256(prefix).hexdigest(), 'prefix_bytes': len(prefix),
                'is_zip': prefix.startswith(b'PK\x03\x04')}, prefix

def classify(body):
    p = Parser()
    p.feed(body.decode('utf-8', errors='replace'))
    text = re.sub(r'\s+', ' ', ' '.join(p.visible)).strip()
    lower = text.lower()
    indicators = {name: (term in lower) for name, term in [
        ('too_many_users', 'too many users'), ('quota', 'quota'), ('sign_in', 'sign in'),
        ('virus_scan', 'virus scan'), ('access_denied', 'access denied'),
        ('cannot_download', "can't download"), ('cannot_view_or_download', "can't view or download"),
        ('not_found', 'does not exist'), ('download', 'download')]}
    return p, indicators, text[:1200]

def main():
    assert not OUTPUT.exists()
    out = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'source':'https://github.com/VisDrone/VisDrone-Dataset',
           'requests':'public Google Drive file page and at most one normal confirmation form',
           'user_accounts_used':False, 'private_data_uploaded':False, 'images_downloaded':False}
    try:
        opener = build_opener(HTTPCookieProcessor(CookieJar()))
        first_info, first = get(opener, URL)
        out['initial'] = first_info
        if first_info['is_zip']:
            out['status'] = 'ZIP_PREFIX_RETURNED_NO_DOWNLOAD'
        else:
            p, indicators, visible = classify(first)
            forms = [f for f in p.forms if f['method'].lower() == 'get' and f['fields'].get('id') == PUBLIC_ID]
            out['initial'].update({'text_indicators':indicators, 'visible_text':visible,
                                   'confirmation_form_count':len(forms),
                                   'confirmation_field_names': sorted(forms[0]['fields']) if len(forms)==1 else []})
            if len(forms) != 1:
                out['status'] = 'NO_UNIQUE_CONFIRMATION_FORM'
            else:
                form = forms[0]
                action = urljoin(URL, form['action'])
                assert safe_host(action)
                # The target is the public file ID; other form values are opaque server-provided tokens.
                separator = '&' if '?' in action else '?'
                final_info, final_body = get(opener, action+separator+urlencode(form['fields']))
                out['confirmation'] = final_info
                if final_info['is_zip']:
                    out['status'] = 'ZIP_PREFIX_RETURNED_NO_DOWNLOAD'
                else:
                    _, flags, visible = classify(final_body)
                    out['confirmation'].update({'text_indicators':flags, 'visible_text':visible})
                    out['status'] = 'HTML_AFTER_CONFIRMATION'
    except Exception as exc:
        out.update({'status':'REQUEST_FAILED', 'exception_type':type(exc).__name__,
                    'http_status':exc.code if isinstance(exc, HTTPError) else None})
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(out, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
