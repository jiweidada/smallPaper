"""Start the one bounded diagnostic once;no training or followup queue."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol-sha',required=True);ap.add_argument('--runner-sha',required=True);args=ap.parse_args()
    assert sha(R/'native_dual_head_protocol_FROZEN.json')==args.protocol_sha
    assert sha(R/'native_dual_head_worker.py')==args.runner_sha
    assert not any((R/n).exists() for n in ['launch.json','status.json','complete.json','failure.json','captures'])
    fd=os.open(str(R/'one_shot.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,json.dumps({'launcher_pid':os.getpid(),'created_unix':time.time()}).encode());os.close(fd)
    log=(R/'stdout.log').open('xb')
    p=subprocess.Popen([sys.executable,str(R/'native_dual_head_worker.py')],cwd=str(R),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    info={'status':'LAUNCHED_ONCE','pid':p.pid,'start_ticks':Path('/proc/%d/stat'%p.pid).read_text().split()[21],'protocol_sha256':args.protocol_sha,'runner_sha256':args.runner_sha,'automatic_followups':False}
    with (R/'launch.json').open('x') as f:json.dump(info,f,indent=2);f.write('\n')
    print(json.dumps(info),flush=True)
if __name__=='__main__':main()
