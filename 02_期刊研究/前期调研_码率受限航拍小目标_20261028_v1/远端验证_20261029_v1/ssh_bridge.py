"""Authenticated SSH/SFTP controller. Password arrives on stdin, never in argv/files/logs."""
from pathlib import Path, PurePosixPath
import json
import getpass
import os
import socket
import sys
import time
import paramiko

ROOT=Path(__file__).resolve().parent
REMOTE_ROOT="/root/c10_region_restore_20261029_v1"
WORK=ROOT.parents[2]

def local_path(value):
    p=Path(value).resolve()
    if not p.is_relative_to(WORK):
        raise ValueError("local file must remain inside the authorized workspace")
    return p

def remote_write_path(value):
    p=PurePosixPath(value)
    if not p.is_absolute() or ".." in p.parts or not (str(p)==REMOTE_ROOT or str(p).startswith(REMOTE_ROOT+"/")):
        raise ValueError("remote writes must remain inside the new experiment directory")
    return str(p)

client=paramiko.SSHClient()
client.load_host_keys(str(ROOT/"known_hosts"))
client.set_missing_host_key_policy(paramiko.RejectPolicy())
print("AUTH_INPUT_REQUIRED",flush=True)
password=getpass.getpass("SSH password: ")
def connect():
    client.connect("172.17.120.160",port=30890,username="root",password=password,
                   timeout=15,banner_timeout=20,auth_timeout=20,allow_agent=False,look_for_keys=False)
    client.get_transport().set_keepalive(20)
connect()  # password is retained only in this process's memory for the authorized session
client.get_transport().set_keepalive(30)
print(json.dumps({"status":"CONNECTED","host_key_verified":True,"password_persisted":False}),flush=True)
if os.name=="nt":
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.windll.kernel32
    kernel.GetStdHandle.restype=wintypes.HANDLE
    console=kernel.GetStdHandle(-10)
    mode=wintypes.DWORD()
    if kernel.GetConsoleMode(console,ctypes.byref(mode)):
        kernel.SetConsoleMode(console,mode.value & ~4)  # suppress JSON command echo in this dedicated console

def mkdir(sftp,path):
    path=remote_write_path(path)
    prefix=""
    for part in PurePosixPath(path).parts[1:]:
        prefix+="/"+part
        try:
            sftp.stat(prefix)
        except FileNotFoundError:
            sftp.mkdir(prefix)

for line in sys.stdin:
    if not line.strip():
        continue
    request=json.loads(line)
    started=time.monotonic()
    result={"id":request.get("id"),"action":request["action"]}
    try:
        action=request["action"]
        if action!="close" and (client.get_transport() is None or not client.get_transport().is_active()):
            connect()  # reconnect transport before the operation, never rerun a submitted experiment
        if action=="exec":
            chan=client.get_transport().open_session(timeout=15)
            chan.exec_command(request["command"])
            stdout,stderr=bytearray(),bytearray()
            limit=int(request.get("output_limit",32000))
            timeout=int(request.get("timeout",120))
            while True:
                if chan.recv_ready():
                    block=chan.recv(65536)
                    if len(stdout)<limit: stdout.extend(block[:limit-len(stdout)])
                if chan.recv_stderr_ready():
                    block=chan.recv_stderr(65536)
                    if len(stderr)<limit: stderr.extend(block[:limit-len(stderr)])
                if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
                    break
                if time.monotonic()-started>timeout:
                    chan.close()
                    raise TimeoutError("SSH channel timeout; channel closed; no automatic experiment restart")
                time.sleep(.05)
            result.update(exit_code=chan.recv_exit_status(),stdout=stdout.decode("utf-8",errors="replace"),stderr=stderr.decode("utf-8",errors="replace"))
            chan.close()
        elif action=="upload":
            source=local_path(request["local"])
            destination=remote_write_path(request["remote"])
            progress_time=[time.monotonic()]
            def progress(sent,total):
                if time.monotonic()-progress_time[0]>=20 or sent==total:
                    info={"id":request.get("id"),"status":"TRANSFER_PROGRESS","sent":sent,"total":total}
                    (ROOT/"transfer_progress.json").write_text(json.dumps(info),encoding="utf-8")
                    print(json.dumps(info),flush=True)
                    progress_time[0]=time.monotonic()
            with paramiko.SFTPClient.from_transport(client.get_transport(),window_size=33554432) as sftp:
                mkdir(sftp,str(PurePosixPath(destination).parent))
                sftp.put(str(source),destination,callback=progress,confirm=True)
            result.update(status="UPLOADED",bytes=source.stat().st_size,remote=destination)
        elif action=="download":
            destination=local_path(request["local"])
            destination.parent.mkdir(parents=True,exist_ok=True)
            with client.open_sftp() as sftp:
                sftp.get(request["remote"],str(destination))
            result.update(status="DOWNLOADED",bytes=destination.stat().st_size,local=str(destination))
        elif action=="close":
            client.close()
            print(json.dumps({"id":request.get("id"),"status":"CLOSED"}),flush=True)
            break
        else:
            raise ValueError("unsupported action")
    except Exception as error:
        result.update(status="ERROR",error_type=type(error).__name__,detail=str(error))
    result["seconds"]=round(time.monotonic()-started,3)
    with (ROOT/"ssh_operation_receipts.jsonl").open("a",encoding="utf-8") as f:
        f.write(json.dumps(result,ensure_ascii=False)+"\n")
    print(json.dumps(result,ensure_ascii=True),flush=True)
client.close()
