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
protocol=hashlib.sha256((bundle/'NO_MSCA/execution_protocol.json').read_bytes()).hexdigest()
for name,filename in [('verify_ablation_backup.py','round040_verified_from_D.json'),('audit_ablation_lineage.py','lineage_through_round040.json')]:
    script=bundle/name
    sys.argv=[str(script),'--arm-root',str(bundle/'NO_MSCA'),'--epoch','39','--protocol-sha256',protocol,'--backup-root',r'D:\MPSMA_DEIM_experiment_backups\20261107\NO_MSCA\backups','--output',str(bundle/'NO_MSCA'/filename)]
    with contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(script),run_name='__main__')
    data=json.loads((bundle/'NO_MSCA'/filename).read_text(encoding='utf-8'))
    print(json.dumps({'file':filename,'status':data['status'],'learning_rates':data.get('learning_rates',data.get('learning_rate_records')),'completed_rounds':data.get('completed_rounds')}))

