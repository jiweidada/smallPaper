from pathlib import Path
from email.parser import Parser
import hashlib,json,re,shutil,tarfile,zipfile
from packaging.requirements import Requirement
from packaging.markers import default_environment
from packaging.version import Version
from packaging.specifiers import SpecifierSet
ROOT=Path(__file__).resolve().parent
normalize=lambda s:re.sub(r'[-_.]+','-',s).lower()
folder=ROOT/'offline_wheelhouse';folder.mkdir(exist_ok=True)
accelerated=ROOT/'accelerated_download_manifest.json'
if accelerated.exists():
    assert json.loads(accelerated.read_bytes())['status']=='VERIFIED_ALL'
    paths=list((ROOT/'accelerated_wheels').glob('*.whl'))
else:
    paths=list((ROOT/'linux_wheels').glob('*.whl'))+list((ROOT/'linux_cuda_wheels').glob('*.whl'))
paths.append(ROOT.parent/'本机CPU短测_20261029_v1/wheels/ultralytics-8.4.191-py3-none-any.whl')
extras=list((ROOT/'linux_extra_wheels').glob('*.whl'))
if extras:
    # The supplemental batch explicitly satisfies Linux/Python3.9-only requirements and supersedes the old thop pin.
    paths=[p for p in paths if not p.name.startswith('ultralytics_thop-')]+extras
records={};metadata={}
for path in paths:
    assert 'win_amd64' not in path.name and 'macosx' not in path.name
    with zipfile.ZipFile(path) as z:
        matches=[n for n in z.namelist() if n.endswith('.dist-info/METADATA') and n.count('/')==1]
        assert len(matches)==1,(path.name,matches)
        m=Parser().parsestr(z.read(matches[0]).decode('utf-8'))
    assert not m['Requires-Python'] or Version('3.9.7') in SpecifierSet(m['Requires-Python']),(path.name,m['Requires-Python'])
    name=normalize(m['Name']);version=m['Version']
    if name in records:assert version==records[name]['version'],(name,version,records[name]['version'])
    else:
        target=folder/path.name
        if not target.exists():shutil.copyfile(path,target)
        with target.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        records[name]={'name':m['Name'],'version':version,'filename':path.name,'sha256':sha,'bytes':target.stat().st_size}
        metadata[name]=m
env=default_environment();env.update(python_version='3.9',python_full_version='3.9.7',platform_system='Linux',platform_machine='x86_64',sys_platform='linux',os_name='posix',extra='')
missing=[]
for name,m in metadata.items():
    for text in m.get_all('Requires-Dist') or []:
        r=Requirement(text)
        if r.marker and not r.marker.evaluate(env):continue
        target=records.get(normalize(r.name))
        if target is None or Version(target['version']) not in r.specifier:missing.append({'parent':name,'requirement':text,'found':target})
json_obj={'target':env,'packages':list(records.values()),'missing_or_incompatible_dependencies':missing,
          'bytes':sum(r['bytes'] for r in records.values()),'user_instruction':'download locally then upload and install without network'}
(ROOT/'offline_bundle_manifest.json').write_text(json.dumps(json_obj,indent=2),encoding='utf-8')
assert not missing,missing
(ROOT/'offline_all_lock.txt').write_text('\n'.join(r['name']+'=='+r['version'] for r in sorted(records.values(),key=lambda r:r['name'].lower()))+'\n',encoding='utf-8')
archive=ROOT/'linux_offline_wheels.tar'
with tarfile.open(archive,'w') as tar:
    for record in sorted(records.values(),key=lambda x:x['filename']):
        path=folder/record['filename'];tar.add(path,arcname='wheelhouse/'+path.name,recursive=False)
    for fn in ['offline_bundle_manifest.json','offline_all_lock.txt']:tar.add(ROOT/fn,arcname=fn,recursive=False)
with archive.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
(ROOT/'offline_bundle_archive.json').write_text(json.dumps({'filename':archive.name,'bytes':archive.stat().st_size,'sha256':sha},indent=2),encoding='utf-8')
pin=[]
source=ROOT.parent/'深入核查_20261028_v1/source_code/v8.4.191_2004cba021c6/ultralytics'
for rel in ['cfg/models/26/yolo26.yaml','cfg/default.yaml','nn/modules/head.py','nn/tasks.py','utils/loss.py','utils/tal.py']:
    pin.append({'relative':rel,'normalized_sha256':hashlib.sha256((source/rel).read_bytes().replace(b'\r\n',b'\n')).hexdigest()})
(ROOT/'source_pin.json').write_text(json.dumps(pin,indent=2),encoding='utf-8')
print(json.dumps({'packages':len(records),'total_bytes':json_obj['bytes'],'archive_sha256':sha,'dependency_graph':'PASS'}),flush=True)
