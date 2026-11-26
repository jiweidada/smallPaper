import pathlib,json,sys,importlib.util
root=pathlib.Path(CONFIG['root']);store=pathlib.Path(CONFIG['storage'])
def read(p):return json.loads(p.read_text()) if p.is_file() else None
pause=root/'pause_user_20261019_v1'
r={'account':CONFIG['account'],'pause':read(pause/'receipt.json'),'pause_failure':read(pause/'failure.json'),'backup_manifest':read(pause/'backup_manifest.json'),'watcher_launch':read(root/'pause_control_20261019_v1/launch.json'),'datasets':[],'old_model_assets':[],'dependencies':{}}
for p in [store/'DEIM/work/DEIM_1572862469/dataset_visdrone',store/'DEIM/dataset_visdrone',store/'dataset_visdrone']:
 for split in ['train','val']:
  d=p/f'VisDrone2019-DET-{split}';ann=d/'annotations'/(split+'.json')
  if ann.is_file():
   a=read(ann);r['datasets'].append({'root':str(d),'split':split,'images':len(a['images']),'first_image_exists':(d/'images'/a['images'][0]['file_name']).is_file()})
for base in [store/'DEIM/experiments/fine_residual_screen24_20261009_v2',store/'experiments/fine_residual_screen24_20261009_v2']:
 if base.is_dir():r['old_model_assets'].append({'root':str(base),'paths':[str(p.relative_to(base)) for p in base.glob('runs/*/last.pth')]})
for name in ['torch','torchvision','numpy','scipy','networkx','pycocotools','timm','yaml']:
 try:s=importlib.util.find_spec(name);r['dependencies'][name]=None if s is None else s.origin
 except Exception as e:r['dependencies'][name]={'error':str(e)}
print(json.dumps(r),flush=True)
