"""Read cache field names/scalar geometry only;no fits or hypothesis outcome analysis."""
from pathlib import Path
import io
import json
import tarfile
import numpy as np

root=Path(__file__).resolve().parent
archive=root/'A0_冻结模型诊断结果_v1/backup/A0_A0b_bounded_diagnostics.tar.gz'
with tarfile.open(archive) as tar:
    manifest=json.load(tar.extractfile('backup/file_manifest.json'))
    first=next(row['path'] for row in manifest if row['path'].startswith('traces/') and row['path'].endswith('.npz'))
    with np.load(io.BytesIO(tar.extractfile(first).read()),allow_pickle=False) as z:
        print(json.dumps(dict(member=first,fields=[dict(key=k,shape=list(z[k].shape),dtype=str(z[k].dtype)) for k in z.files]),indent=2))
        for key in z.files:
            if z[key].size<=4 and any(word in key.lower() for word in ('size','shape','height','width','orig')):
                print(json.dumps(dict(key=key,value=z[key].tolist())))
