import contextlib,io,json,os,runpy,sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
sys.stderr.reconfigure(encoding='utf-8',errors='backslashreplace')
root=Path.cwd()/'01_会议论文'
site=root/'本地CPU核验_20261103/.runtime/site-packages'
handles=[os.add_dll_directory(str(p)) for p in (Path(sys.base_prefix),Path(sys.base_prefix)/'DLLs',site/'torch/lib') if p.is_dir()]
remote=root/'远程公平对照_20261103'
folder=remote/'断点续训_20261105/FULL_30848_20261106'
sys.path=[str(site),str(folder),str(folder.parent),str(remote)]+[p for p in sys.path if 'site-packages' not in p.replace('\\','/').split('/')]
sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']=''


bundle=remote/'模块消融'
sys.path.insert(0,str(bundle))
script=bundle/'audit_module_shared_conditions.py'
sys.argv=[str(script),'--output',str(bundle/'module_shared_conditions_verified_20261108.json')]
runpy.run_path(str(script),run_name='__main__')

