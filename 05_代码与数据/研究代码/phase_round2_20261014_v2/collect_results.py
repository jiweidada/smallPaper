"""Bounded local collector; copies only logs/args/evaluation JSON/analysis CSV.

Runs until both authorized queues complete or fail. Never sends commands that
start, stop, modify, or retry training. A failed transfer is retried next poll.
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
DESTINATION = WORKSPACE/'英文小论文/data/06_第二轮_20261014'
EXECUTION = DESTINATION/'execution_v1'
STATE_PATH = EXECUTION/'collector_state_v2.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    data = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {'collected':{'A':[],'B':[]},'terminal':{}}
    data['collector_pid'] = __import__('os').getpid()
    while True:
        for account in ['A','B']:
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            source = (HERE/'collection_remote.py').read_text().replace('COLLECTED = []  # INSERT_COLLECTED_IDS', 'COLLECTED = '+repr(data['collected'][account]))
            request = EXECUTION/('collection_request_'+account+'.py')
            request.write_text(source,encoding='utf-8',newline='\n')
            command = [sys.executable,'-X','utf8',str(HERE/'ssh_control.py'),account,'--script',str(request),'--timeout','90']
            try:
                completed = subprocess.run(command,capture_output=True,timeout=100)
                if completed.returncode:
                    raise RuntimeError('Read-only SSH query failed: '+completed.stderr.decode(errors='replace')[-1000:])
                response = json.loads(completed.stdout)
                received = []
                for item in response['files']:
                    relative = pathlib.PurePosixPath(item['local_relative'])
                    if relative.is_absolute() or '..' in relative.parts:
                        raise ValueError('Invalid destination in source manifest')
                    target = DESTINATION.joinpath(*relative.parts)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists():
                        if hashlib.sha256(target.read_bytes()).hexdigest()!=item['sha256']:
                            raise RuntimeError('Existing local evidence has different bytes: '+str(target))
                    else:
                        transfer = subprocess.run([sys.executable,'-X','utf8',str(HERE/'ssh_control.py'),account,
                                                   '--get',item['remote'],str(target),'--timeout','180'],capture_output=True,timeout=190)
                        if transfer.returncode:
                            raise RuntimeError('Transfer failed: '+str(target))
                        if hashlib.sha256(target.read_bytes()).hexdigest()!=item['sha256']:
                            raise RuntimeError('Downloaded evidence failed SHA256 verification')
                    received.append({'local':str(target),'remote':item['remote'],'sha256':item['sha256']})
                receipt = EXECUTION/('collection_'+account+'_'+stamp+'.json')
                with receipt.open('x',encoding='utf-8') as f:
                    json.dump({'queue':response['queue'],'received':received,'remote_time':response['remote_time']},f,ensure_ascii=False,indent=2)
                data['collected'][account] += [name for name in response['new_collection_ids'] if name not in data['collected'][account]]
                data['terminal'][account] = response['queue']['state'] in ['complete','failed']
                data[account+'_queue'] = response['queue']
                print(json.dumps({'account':account,'queue_state':response['queue']['state'],'new_ids':response['new_collection_ids'],'copied_files':len(received)}),flush=True)
                data.pop(account+'_error',None)
            except Exception as exc:
                data[account+'_error'] = type(exc).__name__+': '+str(exc)
                print(json.dumps({'account':account,'error':data[account+'_error']}),flush=True)
        data['observed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        temporary = STATE_PATH.with_suffix('.tmp')
        temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.replace(STATE_PATH)
        if args.once or all(data['terminal'].get(account,False) for account in ['A','B']):
            return
        time.sleep(600)


if __name__=='__main__':
    main()
