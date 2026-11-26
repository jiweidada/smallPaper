"""Prepare a separate bounded-transfer bridge; never mutate training or old receipts."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
old=ROOT/'remote_A49.py'
text=old.read_text(encoding='utf-8-sig')
helper='''def bounded_get(sftp, remote, path, resume=False):
    total = sftp.stat(str(remote)).st_size
    start = path.stat().st_size if resume and path.exists() else 0
    if path.exists() and not resume:
        raise FileExistsError('Inspect existing target rather than overwrite: ' + str(path))
    assert 0 <= start <= total
    offset = start
    last_report = start
    with sftp.open(str(remote), 'rb') as src, path.open('ab' if start else 'xb') as dst:
        src.seek(start)
        while offset < total:
            end = min(offset + 16 * src.MAX_REQUEST_SIZE, total)
            # The installed Paramiko version has no max_concurrent_prefetch_requests.
            # Its inspected prefetch implementation queues only [current position, file_size).
            # Bound each window to at most16 requests and fully consume it before the next.
            src.prefetch(file_size=end)
            data = src.read(end - offset)
            if len(data) != end - offset:
                raise EOFError('Partial bounded read')
            dst.write(data)
            offset = end
            if offset - last_report >= 32 << 20 or offset == total:
                dst.flush()
                print(json.dumps({'status':'TRANSFER_PROGRESS','file':path.name,'bytes':offset,
                                  'total':total,'resumed_from':start}),flush=True)
                last_report = offset
    assert path.stat().st_size == total
    return {'resumed_from':start,'max_prefetch_requests_per_window':16,'remote_bytes':total}

'''
text=text.replace("password = getpass.getpass",helper+"password = getpass.getpass",1)
old_call="                        sftp.get(str(dest), str(path))"
assert old_call in text
text=text.replace(old_call,"                        bounded_get(sftp, dest, path, resume=req.get('resume', False))",1)
ast.parse(text)
new=ROOT/'remote_A49_bounded_transfers.py'
assert not new.exists()
new.write_text(text,encoding='utf-8')
receipt={'reason':'Original unlimited SFTP prefetch connection dropped after65536bytes. Same M PID/start confirmed live afterwards; no experiment restarted.',
         'old_bridge_source_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),
         'new_bridge_source_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),
         'old_download_error_receipt':'B_completed_bundle_download_v1.json',
         'preserve_original_download_error_receipt':True,'bounded_windows_max_requests':16,
         'partial_bytes_before_recovery':(ROOT/'B_completed_bundle_v1.zip').stat().st_size,
         'new_detector_operations':0,'new_optimizer_jobs':0}
with (ROOT/'bounded_transfer_recovery_preparation_v1.json').open('x',encoding='utf-8') as f:
    json.dump(receipt,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':'PREPARED','script':str(new),'partial_bytes':receipt['partial_bytes_before_recovery']}))
