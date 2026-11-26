"""Download anonymous files linked by the UAVDT authors, preserving source receipts.

Only follow Google's ordinary public large-file download confirmation form.
Never log in, solve a challenge, change proxy settings, or replace existing files.
"""
import argparse
import datetime
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import time
from urllib.parse import urlsplit
import zipfile

import requests

FILES = {
    'images': ('1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc', 'UAV-benchmark-M.zip'),
    'toolkit': ('19498uJd7T9w4quwnQEy62nibt3uyT9pq', 'UAV-benchmark-MOTD_v1.0.zip'),
    'readme': ('1FV7zNa_Z7ivir7Kjw3Cpp2SmeQSinij_', 'README.txt'),
}


class PublicDownloadForm(HTMLParser):
    def __init__(self):
        super().__init__(); self.action = None; self.fields = {}; self.in_form = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'form' and attrs.get('id') == 'download-form':
            self.action = attrs.get('action'); self.in_form = True
        elif tag == 'input' and self.in_form and attrs.get('name'):
            self.fields[attrs['name']] = attrs.get('value', '')

    def handle_endtag(self, tag):
        if tag == 'form': self.in_form = False


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('label', choices=FILES)
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    file_id, name = FILES[args.label]; target = args.out / name
    receipt_path = args.out / (name + '.receipt.json')
    if target.exists() or receipt_path.exists():
        raise FileExistsError('Refusing to replace an existing author download: ' + str(target))
    partial = target.with_name(target.name + '.part')
    if partial.exists(): raise FileExistsError('A previous partial download needs explicit inspection: ' + str(partial))
    session = requests.Session(); session.headers['User-Agent'] = 'Codex-research-dataset-download'
    url = 'https://drive.usercontent.google.com/download'
    query = {'id': file_id, 'export': 'download', 'confirm': 't'}
    response = session.get(url, params=query, stream=True, timeout=(15, 60))
    response.raise_for_status()
    if 'text/html' in response.headers.get('Content-Type', ''):
        text = response.text; response.close()
        form = PublicDownloadForm(); form.feed(text)
        parsed = urlsplit(form.action or '')
        if (parsed.scheme != 'https' or parsed.netloc != 'drive.usercontent.google.com'
                or form.fields.get('id') != file_id
                or 'virus scan' not in text.lower() or 'captcha' in text.lower()):
            raise RuntimeError('The public download is unavailable or requires user authentication/challenge; no bypass attempted.')
        response = session.get(form.action, params=form.fields, stream=True, timeout=(15, 60))
        response.raise_for_status()
    content_type = response.headers.get('Content-Type', '')
    if 'text/html' in content_type:
        raise RuntimeError('Refusing to save an HTML login, quota, or confirmation page as data.')
    expected = int(response.headers['Content-Length']) if response.headers.get('Content-Length') else None
    started = time.monotonic(); last_print = started; total = 0; digest = hashlib.sha256()
    print(json.dumps({'event': 'download_started', 'label': args.label, 'file': name, 'expected_bytes': expected}), flush=True)
    try:
        with partial.open('xb') as handle:
            for block in response.iter_content(2 * 1024 * 1024):
                if not block: continue
                handle.write(block); digest.update(block); total += len(block)
                now = time.monotonic()
                if now - last_print >= 20:
                    print(json.dumps({'event': 'progress', 'label': args.label, 'bytes': total,
                                      'MiB_per_second': round(total / 2**20 / (now-started), 2)}), flush=True)
                    last_print = now
    finally:
        response.close(); session.close()
    if expected is not None and total != expected:
        raise RuntimeError('Download length does not match HTTP Content-Length; partial retained.')
    members = None
    if name.endswith('.zip'):
        with zipfile.ZipFile(partial) as archive:
            members = archive.namelist()
            bad = archive.testzip()
            if bad: raise RuntimeError('Archive CRC mismatch: ' + bad)
    partial.rename(target)
    receipt = {'collected_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'author_homepage': 'https://sites.google.com/view/daweidu/projects/uavdt',
               'author_download_link': 'https://drive.google.com/open?id=' + file_id,
               'name': name, 'path': str(target.resolve()), 'bytes': total,
               'sha256': digest.hexdigest(), 'elapsed_seconds': time.monotonic() - started,
               'zip_crc_verified': members is not None, 'members': members}
    with receipt_path.open('x', encoding='utf-8') as handle: json.dump(receipt, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in receipt.items() if key != 'members'}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
