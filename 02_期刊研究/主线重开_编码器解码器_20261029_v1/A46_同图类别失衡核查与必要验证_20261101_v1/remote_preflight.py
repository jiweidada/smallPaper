from pathlib import Path
import json,time,getpass,hashlib,shlex
import paramiko
ROOT=Path(__file__).resolve().parent
password=getpass.getpass("SSH password (not recorded): ")
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    client.connect("172.17.120.160",port=30890,username="root",password=password,
                   timeout=12,banner_timeout=12,auth_timeout=12,look_for_keys=False,allow_agent=False)
    password=None
    code = r"""
import sys,json,pathlib,subprocess,os
base=pathlib.Path('/root/c10_region_restore_20261029_v1')
old=base/'architecture_probe_v1/a43_ecdet_distill_three_arm_20261031_v1'
items=['runtime_common.py','arm_B/epoch12.pth','common_init_10class.pth']
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
jobs=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
print(json.dumps({'python':sys.executable,'old_root':str(old),'old_root_exists':old.is_dir(),'old_assets':{i:(old/i).stat().st_size if (old/i).is_file() else None for i in items},'train_images':(base/'dataset/train/images').is_dir(),'train_annotations':(base/'dataset/train/annotations').is_dir(),'gpu':gpu.stdout.strip(),'gpu_jobs':jobs.stdout.strip(),'env_pythons':[str(p) for p in pathlib.Path('/root/miniconda3/envs').glob('*/bin/python')],'new_model_forwards':0,'new_optimizer_updates':0}))
"""
    stdin,stdout,stderr=client.exec_command("python -c "+shlex.quote(code),timeout=30)
    raw=stdout.read().decode("utf-8",errors="replace")
    err=stderr.read().decode("utf-8",errors="replace")
    status=stdout.channel.recv_exit_status()
    receipt={'status':'PASS' if status==0 else 'COMMAND_ERROR','remote_exit':status,
             'stdout':raw,'stderr':err,'remote_access':True,'new_model_forwards':0,'new_optimizer_updates':0,
             'server_key_sha256':hashlib.sha256(client.get_transport().get_remote_server_key().asbytes()).hexdigest()}
    (ROOT/'D1_remote_preflight.json').write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(json.dumps(receipt))
except Exception as e:
    (ROOT/'D1_remote_preflight.json').write_text(json.dumps({'status':'CONNECTION_FAILED','type':type(e).__name__,'error':str(e),'new_model_forwards':0,'new_optimizer_updates':0},indent=2),encoding="utf-8")
    print(json.dumps({'status':'CONNECTION_FAILED','type':type(e).__name__,'error':str(e)}))
finally:
    client.close()

