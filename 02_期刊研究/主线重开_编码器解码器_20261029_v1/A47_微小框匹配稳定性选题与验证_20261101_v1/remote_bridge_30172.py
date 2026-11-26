from pathlib import Path,PurePosixPath
import json,getpass,hashlib,shlex,sys,warnings,time
warnings.filterwarnings("ignore")
import paramiko
ROOT=Path(__file__).resolve().parent
REMOTE="/root/c10_region_restore_20261029_v1/architecture_probe_v1/a47_matching_sensitivity_diagnostic_20261101_v1"
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
password=getpass.getpass("SSH password (not recorded): ")
client.connect("172.17.120.160",port=30172,username="root",password=password,timeout=12,
               banner_timeout=12,auth_timeout=12,allow_agent=False,look_for_keys=False)
password=None
key=hashlib.sha256(client.get_transport().get_remote_server_key().asbytes()).hexdigest()
(ROOT/"server_30172_observed_host_key.json").write_text(json.dumps({"host":"172.17.120.160","port":30172,"server_key_sha256":key,"provenance":"new endpoint explicitly supplied by user; first successful session key recorded; system known_hosts not modified"},indent=2),encoding="utf-8")
sftp=client.open_sftp()
print(json.dumps({"status":"CONNECTED","server_key_sha256":key}),flush=True)
for line in sys.stdin:
    try:
        request=json.loads(line)
        op=request["op"]
        if op=="close":break
        if op=="exec":
            timeout=request.get("timeout",30)
            a,b,c=client.exec_command(request["cmd"],timeout=timeout)
            out=b.read().decode("utf-8",errors="replace")
            err=c.read().decode("utf-8",errors="replace")
            result={"status":"EXECUTED","exit":b.channel.recv_exit_status(),"stdout":out,"stderr":err}
        elif op=="inventory":
            paths=request["paths"];items={}
            for p in paths:
                try:
                    st=sftp.stat(p)
                    import stat
                    items[p]={"bytes":st.st_size,"directory":stat.S_ISDIR(st.st_mode)}
                    if request.get("list_dirs") and stat.S_ISDIR(st.st_mode):items[p]["files"]=sftp.listdir(p)
                except OSError as e:items[p]={"error":str(e)}
            result={"status":"INVENTORY","items":items}
        elif op=="put":
            local=Path(request["local"]).resolve()
            assert local.is_relative_to(ROOT)
            remote=str(PurePosixPath(request["remote"]))
            assert remote.startswith(REMOTE+"/") and ".." not in PurePosixPath(remote).parts
            parts=PurePosixPath(remote).parent.parts
            current=""
            for part in parts:
                current=current+"/"+part if current else part
                if current=="/":continue
                try:sftp.stat(current)
                except OSError:sftp.mkdir(current)
            try:sftp.stat(remote);raise RuntimeError("Refusing to overwrite existing remote file")
            except FileNotFoundError:pass
            except OSError as e:
                if getattr(e,"errno",None)!=2:raise
            sftp.put(str(local),remote)
            with sftp.open(remote,"rb") as f:got=f.read()
            expect=local.read_bytes()
            assert hashlib.sha256(got).digest()==hashlib.sha256(expect).digest()
            result={"status":"UPLOADED_VERIFIED","remote":remote,"bytes":len(got),"sha256":hashlib.sha256(got).hexdigest()}
        elif op=="get":
            remote=str(PurePosixPath(request["remote"]))
            assert remote.startswith(REMOTE+"/") and ".." not in PurePosixPath(remote).parts
            local=Path(request["local"]).resolve();assert local.is_relative_to(ROOT)
            local.parent.mkdir(parents=True,exist_ok=True)
            with sftp.open(remote,"rb") as f:data=f.read()
            if local.exists():assert local.read_bytes()==data
            else:local.write_bytes(data)
            result={"status":"DOWNLOADED","local":str(local),"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
        else:raise ValueError("unknown op")
        if request.get("receipt"):
            p=(ROOT/request["receipt"]).resolve();assert p.is_relative_to(ROOT)
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except Exception as e:print(json.dumps({"status":"BRIDGE_ERROR","type":type(e).__name__,"error":str(e)}),flush=True)
sftp.close();client.close()

