from pathlib import Path
import getpass,json,hashlib,sys,warnings
warnings.filterwarnings("ignore")
import paramiko
ROOT=Path(__file__).resolve().parent
PASSWORD=getpass.getpass("SSH password (memory only): ")
EXPECTED=json.loads((ROOT/"server_30172_observed_host_key.json").read_text(encoding="utf-8"))["server_key_sha256"]
print('{"status":"READY_FOR_FRESH_CONNECTION_QUERIES"}',flush=True)
for line in sys.stdin:
    request=json.loads(line)
    if request["op"]=="close":break
    client=paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect("172.17.120.160",port=30172,username="root",password=PASSWORD,timeout=12,
                       banner_timeout=12,auth_timeout=12,allow_agent=False,look_for_keys=False)
        client.get_transport().set_keepalive(10)
        key=hashlib.sha256(client.get_transport().get_remote_server_key().asbytes()).hexdigest()
        assert key==EXPECTED
        if request["op"]=="exec":
            a,b,c=client.exec_command(request["cmd"],timeout=request.get("timeout",25))
            out=b.read().decode("utf-8",errors="replace");err=c.read().decode("utf-8",errors="replace")
            result={"status":"EXECUTED","exit":b.channel.recv_exit_status(),"stdout":out,"stderr":err,"fresh_connection":True}
            a.close();b.close();c.close();b.channel.close()
        else:raise ValueError("read-only exec supported")
    except Exception as e:
        result={"status":"CONNECTION_OR_COMMAND_ERROR","type":type(e).__name__,"error":str(e),"fresh_connection":True}
    finally:client.close()
    if request.get("receipt"):
        p=(ROOT/request["receipt"]).resolve();assert p.is_relative_to(ROOT)
        p.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False),flush=True)
PASSWORD=None

