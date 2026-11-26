"""SSH for user-supplied endpoints; credentials stay DPAPI encrypted locally."""
from pathlib import Path
import argparse
import ctypes
from ctypes import wintypes
import getpass
import hashlib
import json
import shlex
import sys
import paramiko

HERE=Path(__file__).resolve().parent
HOST="172.17.120.160"
PORTS=(30715,30048,30344,30009)

class Blob(ctypes.Structure):
    _fields_=[("size",wintypes.DWORD),("data",ctypes.POINTER(ctypes.c_ubyte))]

def protect(data,decode=False):
    buffer=ctypes.create_string_buffer(data)
    source=Blob(len(data),ctypes.cast(buffer,ctypes.POINTER(ctypes.c_ubyte)))
    target=Blob()
    function=ctypes.windll.crypt32.CryptUnprotectData if decode else ctypes.windll.crypt32.CryptProtectData
    description=None if decode else "MPSMA-DEIM authorized server credential"
    if not function(ctypes.byref(source),description,None,None,None,1,ctypes.byref(target)):
        raise ctypes.WinError()
    try:return ctypes.string_at(target.data,target.size)
    finally:
        ctypes.windll.kernel32.LocalFree.argtypes=[ctypes.c_void_p]
        ctypes.windll.kernel32.LocalFree.restype=ctypes.c_void_p
        ctypes.windll.kernel32.LocalFree(target.data)

def credential_path(port,slot="control"):
    assert port in PORTS
    assert slot in ("control","backup")
    suffix="" if slot=="control" else "_backup"
    return HERE/".auth"/f"root_{port}{suffix}.dpapi"

def connect(port):
    # The interactive sandbox and the approved background client previously
    # used different Windows identities. Select an existing decryptable local
    # blob; never copy plaintext between identities or retry training here.
    password=None
    error=None
    for slot in ("control","backup"):
        try:
            encrypted=credential_path(port,slot).read_bytes()
            password=protect(encrypted,decode=True).decode("utf-8")
            break
        except OSError as caught:error=caught
    if password is None:raise error or RuntimeError("No credential cache available")
    client=paramiko.SSHClient()
    hosts=HERE/"known_hosts"
    hosts.touch(exist_ok=True)
    client.load_host_keys(str(hosts))
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST,port=port,username="root",password=password,allow_agent=False,look_for_keys=False,
                       timeout=15,banner_timeout=20,auth_timeout=20)
        client.get_transport().set_keepalive(20)
        return client
    finally:password=None

def run_script(client,script,python="python3",timeout=180):
    stdin,stdout,stderr=client.exec_command(shlex.quote(python)+" -",timeout=timeout)
    stdin.write(script)
    stdin.channel.shutdown_write()
    output=stdout.read().decode("utf-8",errors="replace")
    error=stderr.read().decode("utf-8",errors="replace")
    return {"exit_code":stdout.channel.recv_exit_status(),"stdout":output,"stderr":error}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--port",type=int,choices=PORTS,required=True)
    parser.add_argument("--cache-secret",action="store_true")
    parser.add_argument("--credential-slot",choices=["control","backup"],default="control")
    parser.add_argument("--script",type=Path)
    parser.add_argument("--python",default="python3")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--timeout-seconds",type=int,default=180)
    args=parser.parse_args()
    if args.cache_secret:
        value=getpass.getpass(f"SSH password for port{args.port}: ")
        path=credential_path(args.port,args.credential_slot)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(protect(value.encode("utf-8")))
        value=None
        print(json.dumps({"port":args.port,"credential_cached":"Windows user-bound DPAPI; no plaintext file"}))
        return
    if not args.script:parser.error("--script is required unless caching a credential")
    client=connect(args.port)
    try:
        result=run_script(client,args.script.read_text(encoding="utf-8-sig"),args.python,args.timeout_seconds)
        result["port"]=args.port
        result["server_key_sha256"]=hashlib.sha256(client.get_transport().get_remote_server_key().asbytes()).hexdigest()
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        if args.output:
            print(json.dumps({"port":args.port,"exit_code":result["exit_code"],"result_file":str(args.output),"stdout_bytes":len(result["stdout"].encode("utf-8")),"stderr":result["stderr"][-2000:]},ensure_ascii=False))
        else:print(json.dumps(result,ensure_ascii=False))
    finally:client.close()

if __name__=="__main__":main()
