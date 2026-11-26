"""Read interpreter/package metadata only. Run with -S; never import ML packages or initialize devices."""
from pathlib import Path
import sys,json,ast,re,site as site_module
from importlib import metadata
sys.stdout.reconfigure(encoding='utf-8')
site=Path(sys.executable).parent/'Lib'/'site-packages'
if not site.is_dir():site=Path(sys.prefix)/'Lib'/'site-packages'
wanted={'torch','torchvision','torchaudio','ultralytics','torch-directml','onnxruntime','onnxruntime-gpu','numpy','pillow','opencv-python','opencv-python-headless','scipy','pycocotools'}
scan_paths=list(dict.fromkeys([str(site),site_module.getusersitepackages()]))
packages={}
for d in metadata.distributions(path=scan_paths):
    name=(d.metadata.get('Name') or '').lower().replace('_','-')
    if name in wanted:packages[name]=d.version
build={}
f=next((Path(s)/'torch'/'version.py' for s in scan_paths if (Path(s)/'torch'/'version.py').is_file()),None)
if f is not None:
    for name,rhs in re.findall(r'^(__version__|cuda|hip|xpu)\s*(?::[^=\n]+)?=\s*([^\n]+)',f.read_text(encoding='utf-8'),re.M):
        try:build[name]=ast.literal_eval(rhs)
        except (ValueError,SyntaxError):build[name]='unparsed metadata'
result={'python_executable':sys.executable,'python_version':sys.version.split()[0],
        'site_packages':str(site),'site_packages_exists':site.is_dir(),
        'metadata_scan_paths':scan_paths,
        'module_roots':{name:[str(Path(s)/name) for s in scan_paths if (Path(s)/name).exists()] for name in ['torch','ultralytics','cv2']},
        'packages':packages,'not_found_in_this_environment':sorted(wanted-packages.keys()),
        'torch_build_metadata':build,
        'execution':'Standard-library metadata reading only, -S disables site initialization. No torch/ultralytics import, model, GPU initialization, training or inference.'}
print(json.dumps(result,ensure_ascii=False))
