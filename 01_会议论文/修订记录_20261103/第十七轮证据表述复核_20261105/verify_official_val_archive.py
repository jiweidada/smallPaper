"""Download the official-linked public validation ZIP and compare two members read-only."""
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import HTTPCookieProcessor, Request, build_opener
from urllib.error import HTTPError
from zipfile import BadZipFile, ZipFile
import json
import os

HERE = Path(__file__).resolve().parent
OUTPUT = HERE/'official_val_archive_verification.json'
ARCHIVE_DIR = HERE/'official_validation_release'
ARCHIVE = ARCHIVE_DIR/'VisDrone2019-DET-val.zip'
PARTIAL = ARCHIVE_DIR/'VisDrone2019-DET-val.zip.part'
ID = '1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59'
URL = 'https://drive.google.com/uc?export=download&id=' + ID
DATA = Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val')
NAMES = ['0000022_00000_d_0000004.jpg', '0000023_00000_d_0000008.jpg']
MAX_BYTES = 200*1024*1024

def digest_file(path):
    h = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def allowed(url):
    host = (urlsplit(url).hostname or '').lower()
    return host in {'drive.google.com', 'drive.usercontent.google.com', 'docs.google.com'} or host.endswith('.googleusercontent.com')

class Forms(HTMLParser):
    def __init__(self):
        super().__init__()
        self.forms, self.current = [], None
    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == 'form':
            self.current = {'action': attributes.get('action', ''), 'method': attributes.get('method', 'get'), 'fields': {}}
            self.forms.append(self.current)
        elif tag == 'input' and self.current and attributes.get('name'):
            self.current['fields'][attributes['name']] = attributes.get('value', '')
    def handle_endtag(self, tag):
        if tag == 'form':
            self.current = None

def open_response(opener, url):
    assert allowed(url)
    response = opener.open(Request(url, headers={'User-Agent': 'Academic data provenance audit'}), timeout=25)
    if not allowed(response.geturl()):
        response.close()
        raise ValueError('redirected to a host outside the official Google download service')
    return response

def copy_member_hash(zip_file, name):
    match = [entry for entry in zip_file.infolist() if entry.filename.replace('\\', '/').endswith('/'+name)]
    assert len(match) == 1
    entry = match[0]
    assert entry.file_size <= 5*1024*1024
    h = sha256()
    with zip_file.open(entry) as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return {'member': entry.filename, 'bytes': entry.file_size, 'sha256': h.hexdigest()}

def main():
    assert not OUTPUT.exists() and not ARCHIVE_DIR.exists()
    ARCHIVE_DIR.mkdir()
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(),
              'official_repository': 'https://github.com/VisDrone/VisDrone-Dataset',
              'official_file_page': 'https://drive.google.com/file/d/'+ID+'/view',
              'data_uploaded': False, 'model_runs': 0, 'evaluator_runs': 0, 'optimizer_updates': 0,
              'scope': 'Only the two specified validation JPEGs and their original annotation TXT files are inspected inside the ZIP; no data are extracted or modified.'}
    try:
        opener = build_opener(HTTPCookieProcessor(CookieJar()))
        response = open_response(opener, URL)
        first = response.read(4096)
        result['initial_content_type'] = response.headers.get('Content-Type')
        result['confirmation_followed'] = False
        if not first.startswith(b'PK\x03\x04'):
            content = first + response.read(28672)
            response.close()
            parser = Forms()
            parser.feed(content.decode('utf-8', errors='replace'))
            forms = [form for form in parser.forms if form['method'].lower() == 'get' and form['fields'].get('id') == ID]
            if len(forms) != 1:
                result.update({'status': 'OFFICIAL_DOWNLOAD_NOT_AVAILABLE', 'html_response_sha256': sha256(content).hexdigest(),
                               'matching_confirmation_forms': len(forms)})
                raise StopIteration
            form = forms[0]
            form_url = urljoin(URL, form['action'])
            assert allowed(form_url)
            fields = {k: v for k, v in form['fields'].items() if k in {'id', 'export', 'confirm', 'uuid'}}
            response = open_response(opener, form_url + '?' + urlencode(fields))
            first = response.read(4096)
            result['confirmation_followed'] = True
        if not first.startswith(b'PK\x03\x04'):
            result.update({'status': 'OFFICIAL_DOWNLOAD_NOT_A_ZIP', 'response_prefix_sha256': sha256(first).hexdigest(),
                           'response_content_type': response.headers.get('Content-Type')})
            response.close()
            raise StopIteration
        declared = response.headers.get('Content-Length')
        if declared and int(declared) > MAX_BYTES:
            raise ValueError('declared size exceeds 200 MiB limit')
        byte_count = len(first)
        with PARTIAL.open('xb') as stream:
            stream.write(first)
            while True:
                chunk = response.read(1024*1024)
                if not chunk:
                    break
                byte_count += len(chunk)
                if byte_count > MAX_BYTES:
                    raise ValueError('download exceeded 200 MiB limit')
                stream.write(chunk)
        response.close()
        if declared:
            assert byte_count == int(declared)
        os.replace(PARTIAL, ARCHIVE)
        image_rows, label_rows = [], []
        with ZipFile(ARCHIVE) as z:
            for name in NAMES:
                image = copy_member_hash(z, 'images/'+name)
                label = copy_member_hash(z, 'annotations/'+Path(name).stem+'.txt')
                image['local_sha256'] = digest_file(DATA/'images'/name)
                label['local_sha256'] = digest_file(DATA/'annotations'/(Path(name).stem+'.txt'))
                image['matches_local'] = image['sha256'] == image['local_sha256']
                label['matches_local'] = label['sha256'] == label['local_sha256']
                image_rows.append(image)
                label_rows.append(label)
        result.update({'status': 'OFFICIAL_LINKED_VALIDATION_ARCHIVE_MEMBERS_HASHED',
                       'archive_path': str(ARCHIVE), 'archive_bytes': byte_count,
                       'archive_sha256': digest_file(ARCHIVE), 'images': image_rows, 'annotations': label_rows,
                       'official_validation_image_pair_identical': image_rows[0]['sha256'] == image_rows[1]['sha256'],
                       'limits': ['Only the two validation image/annotation pairs were checked, not the train archive.',
                                  'The link is official-repository-linked, but no separately published archive checksum was available.',
                                  'This does not authenticate historical training data or calculate an AP impact.']})
    except StopIteration:
        pass
    except Exception as exc:
        result.update({'status': 'OFFICIAL_ARCHIVE_CHECK_INCOMPLETE', 'exception_type': type(exc).__name__,
                       'http_status': exc.code if isinstance(exc, HTTPError) else None,
                       'partial_file_retained': str(PARTIAL) if PARTIAL.exists() else None})
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
