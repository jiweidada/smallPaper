"""Anonymous metadata and bounded ZIP inventory checks for an original-format mirror."""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import subprocess
import zipfile

REPO='vanthanh/UAVDT-Benchmark-M'
API='https://huggingface.co/api/datasets/'+REPO+'/tree/main'

class HTTPRangeFile(io.RawIOBase):
    def __init__(self,url,size):
        self.url=url;self.size=size;self.position=0;self.transferred=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if self.position<0:raise ValueError('Negative seek')
        return self.position
    def read(self,length=-1):
        if length<0:length=self.size-self.position
        length=min(length,self.size-self.position)
        if length<=0:return b''
        if length>16*1024**2 or self.transferred+length>40*1024**2:raise RuntimeError('Bounded mirror probe budget exceeded')
        start=self.position;end=start+length-1
        result=subprocess.run(['curl.exe','--fail','--silent','--show-error','--location','--connect-timeout','10',
            '--max-time','40','--max-filesize',str(length),'--range',f'{start}-{end}',self.url],capture_output=True,timeout=45)
        if result.returncode:raise RuntimeError('Bounded curl range failed: '+result.stderr.decode(errors='replace')[-300:])
        raw=result.stdout
        if len(raw)!=length:raise RuntimeError('Partial range or oversized response')
        self.position+=length;self.transferred+=length;return raw

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path);parser.add_argument('--metadata-only',action='store_true');args=parser.parse_args()
    response=subprocess.run(['curl.exe','--fail','--silent','--show-error','--location','--connect-timeout','10',
        '--max-time','40',API+'?expand=true'],capture_output=True,timeout=45)
    if response.returncode:raise RuntimeError(response.stderr.decode(errors='replace')[-400:])
    files=json.loads(response.stdout)
    result={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repository':REPO,'metadata':files}
    by_name={x['path']:x for x in files}
    toolkit=by_name['UAV-benchmark-MOTD_v1.0.zip']
    expected='1e565da8c2a035bf1e56b1322de76c74b73702ea3b0f918f45debf69c2b26799'
    result['toolkit_matches_downloaded_author_bytes']=(toolkit.get('lfs',{}).get('oid')==expected)
    assert result['toolkit_matches_downloaded_author_bytes'],'Mirror toolkit differs from author download'
    print(json.dumps({'event':'metadata_verified','official_toolkit_sha_matches':True}),flush=True)
    if not args.metadata_only:
        image=by_name['UAV-benchmark-M.zip'];url='https://huggingface.co/datasets/'+REPO+'/resolve/main/'+image['path']+'?download=true'
        remote=HTTPRangeFile(url,image['size'])
        with zipfile.ZipFile(remote) as archive:
            names=archive.namelist();groups={};other=[]
            for name in names:
                m=re.search(r'(?:^|/)(M\d{4})/img(\d{6})\.jpg$',name)
                if m:groups.setdefault(m[1],[]).append(int(m[2]))
                elif not name.endswith('/'):other.append(name)
            result['zip_inventory']={'members':len(names),'sequences':len(groups),'image_count':sum(map(len,groups.values())),
                'per_sequence':{k:{'images':len(v),'minimum_frame':min(v),'maximum_frame':max(v),
                    'missing_internal_frames':sorted(set(range(min(v),max(v)+1))-set(v))} for k,v in sorted(groups.items())},
                'other_files':other[:100],'first_members':names[:12],'range_bytes':remote.transferred}
            assert len(groups)==50 and not other,'Unexpected original detection archive structure'
        result['image_package_sha256_from_mirror_metadata']=image['lfs']['oid']
    if args.out:
        with args.out.open('x',encoding='utf-8') as handle:json.dump(result,handle,ensure_ascii=False,indent=2)
    print(json.dumps(result if not args.out else {'receipt':str(args.out),'toolkit_author_sha_matches':True,'inventory':result.get('zip_inventory')},ensure_ascii=False))

if __name__=='__main__':main()
