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
import hashlib
script=bundle/'audit_ablation_lineage.py'
sys.argv=[str(script),'--arm-root',str(bundle/'NO_MPS'),'--epoch','308','--protocol-sha256',hashlib.sha256((bundle/'NO_MPS/execution_protocol.json').read_bytes()).hexdigest(),'--backup-root',r'D:\MPSMA_DEIM_experiment_backups\20261107\NO_MPS\backups','--output',str(bundle/'NO_MPS/lineage_through_round309.json')]
with contextlib.redirect_stdout(io.StringIO()):
    runpy.run_path(str(script),run_name='__main__')
data=json.loads((bundle/'NO_MPS/lineage_through_round309.json').read_text(encoding='utf-8'))
print(json.dumps({key:data[key] for key in ['status','completed_rounds','rolling_state_restored_after_evaluation','restore_event_count','learning_rates']}))

