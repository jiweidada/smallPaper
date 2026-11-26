from pathlib import Path
import json,getpass,hashlib,zipfile,warnings
warnings.filterwarnings("ignore")
import paramiko
ROOT=Path(__file__).resolve().parent;REMOTE="/root/uav_a47_20261101_v1"
client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
pw=getpass.getpass("SSH password (not recorded): ")
client.connect("172.17.120.160",port=30172,username="root",password=pw,timeout=12,banner_timeout=12,auth_timeout=12,allow_agent=False,look_for_keys=False);pw=None
assert hashlib.sha256(client.get_transport().get_remote_server_key().asbytes()).hexdigest()==json.loads((ROOT/"server_30172_observed_host_key.json").read_text())["server_key_sha256"]
sftp=client.open_sftp();sftp.get_channel().settimeout(30)
def get(name,local):
    local=Path(local)
    assert not local.exists()
    with sftp.open(REMOTE+"/"+name,"rb") as f:data=f.read()
    local.write_bytes(data);return data
receipt=json.loads(get("result_archive_receipt.json",ROOT/"remote_result_archive_receipt.json"))
data=get("A47_remote_diagnostic_results_v1.zip",ROOT/"A47_remote_diagnostic_results_v1.zip")
assert len(data)==receipt["bytes"] and hashlib.sha256(data).hexdigest()==receipt["sha256"]
sftp.close();client.close()
out=ROOT/"remote_downloaded_30172";out.mkdir(exist_ok=False)
with zipfile.ZipFile(ROOT/"A47_remote_diagnostic_results_v1.zip") as z:
    assert z.testzip() is None
    names=z.namelist();assert len(names)==receipt["files"]
    for name in names:
        p=(out/name).resolve();assert p.is_relative_to(out.resolve())
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
manifest_path=out/"result_archive_manifest.json"
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==receipt["manifest_sha256"]
manifest=json.loads(manifest_path.read_text())
for name,m in manifest.items():
    p=out/name;assert p.stat().st_size==m["bytes"] and hashlib.sha256(p.read_bytes()).hexdigest()==m["sha256"],name
result={"status":"PASS","archive_sha256":receipt["sha256"],"archive_bytes":receipt["bytes"],
        "ZIP_CRC":"PASS","manifest_files_verified":len(manifest),"total_downloaded_files":len(names),
        "new_model_forwards":0,"optimizer_updates":0}
(ROOT/"remote_download_verified.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result))

