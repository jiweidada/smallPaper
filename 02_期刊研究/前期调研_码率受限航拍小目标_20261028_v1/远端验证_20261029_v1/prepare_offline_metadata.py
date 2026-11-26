from pathlib import Path
import json
from packaging.requirements import Requirement
from packaging.markers import default_environment
ROOT=Path(__file__).resolve().parent
metadata=json.loads((ROOT/'torch_pypi_metadata.json').read_bytes())
env=default_environment()
env.update(python_version='3.9',python_full_version='3.9.7',platform_system='Linux',platform_machine='x86_64',sys_platform='linux',os_name='posix',extra='')
requirements=[]
for text in metadata['info']['requires_dist']:
    r=Requirement(text)
    if r.name.startswith('nvidia-') or r.name=='triton':
        if r.marker is None or r.marker.evaluate(env):requirements.append(r.name+str(r.specifier))
(ROOT/'linux_cuda_requirements.txt').write_text('\n'.join(requirements)+'\n',encoding='utf-8')
assets=[]
for filename,wanted in [('ultralytics_release_api.json',{'yolo26s.pt','yolo26n.pt'}),('fbcnn_release_api.json',{'fbcnn_color.pth'})]:
    release=json.loads((ROOT/filename).read_bytes())
    for a in release.get('assets',[]):
        if a['name'] in wanted:
            assets.append({'name':a['name'],'size':a['size'],'url':a['browser_download_url'],
                           'release':release.get('html_url'),'expected_digest':a.get('digest')})
assert {a['name'] for a in assets}=={'yolo26s.pt','yolo26n.pt','fbcnn_color.pth'}
(ROOT/'public_assets_to_download.json').write_text(json.dumps(assets,indent=2),encoding='utf-8')
print(json.dumps({'cuda_requirements':requirements,'assets':assets}))
