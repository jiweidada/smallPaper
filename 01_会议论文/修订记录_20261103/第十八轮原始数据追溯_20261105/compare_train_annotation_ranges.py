"""Read only four TXT members of a public training ZIP via bounded HTTP ranges."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import json
import re
import struct
import zlib

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PROBE = HERE/'train_archive_range_probe.json'
PRIOR = REC/'第十七轮证据表述复核_20261105/duplicate_provenance_verification.json'
DATA = Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\annotations')
OUTPUT = HERE/'train_annotation_range_audit.json'
SNAPSHOT = HERE/'public_train_annotation_snapshot'
REPO = 'HoangTrungNguyen/VisDrone2019_training'
ARCHIVE = 'data/VisDrone2019-DET-train.zip'
NAMES = ['9999950_00000_d_10000080.txt','9999950_00000_d_0000079.txt',
         '0000239_06450_d_0000017.txt','0000239_06950_d_0000018.txt']

def get_range(url, start, end, total):
    assert 0 <= start <= end < total and end-start+1 <= 5*1024*1024
    req = Request(url, headers={'User-Agent':'Academic data provenance audit', 'Range':f'bytes={start}-{end}'})
    with urlopen(req, timeout=30) as response:
        actual = response.headers.get('Content-Range', '')
        expected = f'bytes {start}-{end}/{total}'
        if response.status != 206 or actual != expected:
            raise ValueError('range was not honored exactly')
        payload = response.read(end-start+2)
        assert len(payload) == end-start+1
        return payload

def parse_central(data, expected_count):
    result, offset = {}, 0
    header = struct.Struct('<4s6H3I5H2I')
    count = 0
    while offset < len(data):
        assert len(data)-offset >= header.size
        v = header.unpack_from(data, offset)
        assert v[0] == b'PK\x01\x02'
        flag, method, crc, csize, usize = v[3], v[4], v[7], v[8], v[9]
        fnlen, extralen, commentlen, local_offset = v[10], v[11], v[12], v[16]
        assert csize != 0xffffffff and usize != 0xffffffff and local_offset != 0xffffffff
        end = offset + header.size + fnlen + extralen + commentlen
        assert end <= len(data)
        raw_name = data[offset+header.size:offset+header.size+fnlen]
        name = raw_name.decode('utf-8' if flag & 0x800 else 'cp437')
        for wanted in NAMES:
            if name.replace('\\', '/').endswith('/annotations/'+wanted):
                assert wanted not in result
                result[wanted] = {'member':name, 'flag':flag, 'method':method, 'crc32':crc,
                                  'compressed_bytes':csize, 'uncompressed_bytes':usize,
                                  'local_header_offset':local_offset}
        count += 1
        offset = end
    assert count == expected_count and set(result) == set(NAMES)
    return result

def retrieve(url, total, name, entry, prior):
    offset = entry['local_header_offset']
    local = get_range(url, offset, offset+29, total)
    v = struct.unpack('<4s5H3I2H', local)
    assert v[0] == b'PK\x03\x04' and v[3] == entry['method']
    name_len, extra_len = v[9], v[10]
    data_start = offset+30+name_len+extra_len
    csize = entry['compressed_bytes']
    assert 0 < csize <= 1024*1024 and entry['uncompressed_bytes'] <= 1024*1024
    compressed = get_range(url, data_start, data_start+csize-1, total)
    if entry['method']==8:
        body = zlib.decompress(compressed, -15)
    elif entry['method']==0:
        body = compressed
    else:
        raise ValueError('unsupported compression')
    assert len(body)==entry['uncompressed_bytes'] and (zlib.crc32(body) & 0xffffffff)==entry['crc32']
    current = (DATA/name).read_bytes()
    current_sha = sha256(current).hexdigest()
    assert current_sha == prior[name]
    return {'name':name, **entry, 'sha256':sha256(body).hexdigest(), 'local_sha256':current_sha,
            'same_as_local': body == current, 'body':body}

def main():
    assert not OUTPUT.exists() and not SNAPSHOT.exists()
    probe = json.loads(PROBE.read_text(encoding='utf-8'))
    assert probe['status']=='RANGE_SUPPORTED' and probe['zip_end_record_present']
    prior_audit = json.loads(PRIOR.read_text(encoding='utf-8'))
    prior = {Path(row['original_text']).name:row['original_text_sha256']
             for group in prior_audit['duplicate_groups'] if group['split']=='train' for row in group['labels']}
    assert set(prior)==set(NAMES)
    revision, total = probe['revision'], probe['declared_archive_bytes']
    url = 'https://huggingface.co/datasets/'+REPO+'/resolve/'+revision+'/'+ARCHIVE
    result = {'checked_utc':datetime.now(timezone.utc).isoformat(), 'repository_revision':revision,
              'archive_pointer_sha256_declared':probe['declared_archive_sha256'], 'archive_bytes_declared':total,
              'range_probe_sha256':sha256(PROBE.read_bytes()).hexdigest(), 'prior_audit_sha256':sha256(PRIOR.read_bytes()).hexdigest(),
              'full_archive_downloaded':False,'private_data_uploaded':False,'models_run':0,'evaluator_runs':0,'optimizer_updates':0}
    try:
        tail_len = min(65536,total)
        tail = get_range(url, total-tail_len, total-1, total)
        assert sha256(tail).hexdigest()==probe['suffix_sha256']
        offset = tail.rfind(b'PK\x05\x06')
        assert offset>=0 and len(tail)-offset>=22
        eocd = struct.unpack_from('<4s4H2IH',tail,offset)
        _, disk, cd_disk, entries_disk, entry_count, cd_size, cd_offset, comment_len = eocd
        assert disk == cd_disk == 0 and entries_disk == entry_count and offset+22+comment_len == len(tail)
        assert cd_size<=5*1024*1024 and cd_offset+cd_size<=total
        directory = get_range(url, cd_offset, cd_offset+cd_size-1, total)
        entries = parse_central(directory,entry_count)
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(lambda name: retrieve(url,total,name,entries[name],prior), NAMES))
        SNAPSHOT.mkdir()
        for row in rows:
            (SNAPSHOT/row['name']).write_bytes(row.pop('body'))
        result.update({'status':'FOUR_TRAIN_ANNOTATIONS_COMPARED_BY_HTTP_RANGE',
                       'directory_bytes':cd_size,'directory_sha256':sha256(directory).hexdigest(),
                       'zip_entry_count':entry_count,'annotations':rows,
                       'all_four_match_local':all(row['same_as_local'] for row in rows),
                       'limits':['This is a third-party mirror, not an authenticated official archive.',
                                 'Only the directory and four original TXT members were read; the full ZIP SHA256 was not recomputed.',
                                 'No training images, models, or evaluator were run, and local data were not modified.']})
    except Exception as exc:
        result.update({'status':'RANGE_ANNOTATION_CHECK_INCOMPLETE','exception_type':type(exc).__name__,
                       'http_status':exc.code if isinstance(exc,HTTPError) else None})
    with OUTPUT.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(json.dumps({k:result.get(k) for k in ['status','directory_bytes','zip_entry_count','all_four_match_local','exception_type','http_status']},ensure_ascii=False))

if __name__=='__main__':
    main()
