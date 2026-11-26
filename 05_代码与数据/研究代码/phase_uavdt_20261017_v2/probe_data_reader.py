"""Read only processes holding this task's native image files; never enumerate command secrets."""
import json,os
from pathlib import Path
base=Path(CONFIG['storage'])/'datasets_verified_20261017';native=base/'UAVDT_official/native_frames_v2'
manifest=json.loads((base/'UAVDT_v1/native_image_manifest.json').read_text())
lookup={entry['relative']:i+1 for i,entry in enumerate(manifest)};found=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or proc.name in ['1',str(os.getpid())]:continue
 try:
  if 'python' not in (proc/'comm').read_text().lower():continue
  for fd in (proc/'fd').iterdir():
   try:path=Path(os.readlink(fd))
   except OSError:continue
   if not path.is_relative_to(native):continue
   relative=str(path.relative_to(native));stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
   args=[a.decode(errors='replace') for a in (proc/'cmdline').read_bytes().split(b'\0') if a]
   if args!=[CONFIG['python'],'-u','-']:continue
   found.append({'pid':int(proc.name),'start_ticks':int(stat[19]),'state':stat[0],'open_image':relative,'approximate_manifest_index':lookup.get(relative),'total':len(manifest),'owned_data_reader_argv':args});break
 except (OSError,ValueError):continue
print(json.dumps({'account':CONFIG['account'],'readers':found,'ready':(base/'UAVDT_v1/data_ready.json').exists()}),flush=True)
