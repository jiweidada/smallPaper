from pathlib import Path
import json,getpass,hashlib,time,warnings
warnings.filterwarnings("ignore")
import paramiko
ROOT=Path(__file__).resolve().parent
REMOTE="/root/uav_a47_20261101_v1"
client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
pw=getpass.getpass("SSH password (not recorded): ")
client.connect("172.17.120.160",port=30172,username="root",password=pw,timeout=12,banner_timeout=12,auth_timeout=12,allow_agent=False,look_for_keys=False)
pw=None
assert hashlib.sha256(client.get_transport().get_remote_server_key().asbytes()).hexdigest()==json.loads((ROOT/"server_30172_observed_host_key.json").read_text())["server_key_sha256"]
client.get_transport().set_keepalive(10)
sftp=client.open_sftp();sftp.get_channel().settimeout(30)
try:sftp.stat(REMOTE)
except OSError as e:
    assert getattr(e,"errno",None)==2;sftp.mkdir(REMOTE)
paths=[ROOT/"A47_minimal_assets_30172_v1.tar"]+sorted((ROOT/"deploy_wheels_pypi_v1").glob("pycocotools*.whl"))
assert len(paths)==2
receipts=[]
for p in paths:
    dest=REMOTE+"/"+p.name
    try:sftp.stat(dest);raise RuntimeError("Refusing to overwrite existing "+dest)
    except OSError as e:assert getattr(e,"errno",None)==2
    partial=dest+".uploading"
    try:sftp.stat(partial);raise RuntimeError("Existing partial preserved: "+partial)
    except OSError as e:assert getattr(e,"errno",None)==2
    last=[0.]
    def progress(sent,total):
        if time.monotonic()-last[0]>=5 or sent==total:
            print(json.dumps({"file":p.name,"sent":sent,"total":total}),flush=True);last[0]=time.monotonic()
    sftp.put(str(p),partial,callback=progress,confirm=True)
    assert sftp.stat(partial).st_size==p.stat().st_size
    sftp.rename(partial,dest)
    receipts.append({"local":str(p),"remote":dest,"bytes":p.stat().st_size,"local_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"remote_size_confirmed":True,"remote_SHA_pending":True})
sftp.close();client.close()
(ROOT/"minimal_upload_receipt.json").write_text(json.dumps({"status":"UPLOADED_SIZE_CONFIRMED","files":receipts},indent=2),encoding="utf-8")
print('{"status":"UPLOADED_SIZE_CONFIRMED_REMOTE_HASH_PENDING"}',flush=True)

