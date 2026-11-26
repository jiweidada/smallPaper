"""Read-only OS-cache warmup of Transformers source files during cold imports."""
import concurrent.futures
import json
import pathlib
import sys
import time

base=pathlib.Path(sys.prefix)/'lib'/('python'+str(sys.version_info.major)+'.'+str(sys.version_info.minor))/'site-packages/transformers/models'
assert base.is_dir()
files=list(base.rglob('*.py'))
assert 0<len(files)<20000
started=time.monotonic()


def read(path):
    with path.open('rb') as handle:
        total=0
        for block in iter(lambda:handle.read(1024**2),b''):
            total+=len(block)
    return total


total=0
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for index,size in enumerate(pool.map(read,files),1):
        total+=size
        if index%300==0:
            print(json.dumps({'read_files':index,'total_files':len(files)}),file=sys.stderr,flush=True)
print(json.dumps({'runtime_source':str(base),'files_read':len(files),'bytes_read':total,
                  'elapsed_seconds':time.monotonic()-started,'read_only':True}))
