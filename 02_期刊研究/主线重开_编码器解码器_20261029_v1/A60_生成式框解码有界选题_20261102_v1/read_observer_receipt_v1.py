"""Read the exact saved SSH response, independent of terminal wrapping."""
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,time

P=Path(__file__).resolve().parent
D=P.parent/'A54_场景尺度约束候选_20261101_v1'
parser=argparse.ArgumentParser()
parser.add_argument('--number',type=int,required=True)
parser.add_argument('--wait',type=float,default=5)
args=parser.parse_args()
assert 0<=args.number<100000 and 0<=args.wait<=30
path=D/f'A60_formal_observe_{args.number:03d}.json'
deadline=time.monotonic()+args.wait
while True:
    try:
        response=json.loads(path.read_text(encoding='utf-8-sig'))
        if response.get('status')!='EXECUTED' or response.get('exit')!=0:
            result={'status':'REMOTE_OBSERVATION_ERROR','receipt':str(path),'response':response}
        else:
            observation=json.loads(response['stdout'])
            result={'status':'SAVED_OBSERVATION_READ','receipt':str(path),
                    'recorded_client_utc':datetime.now(timezone.utc).isoformat(),
                    'observation':observation}
        break
    except (FileNotFoundError,json.JSONDecodeError) as exc:
        if time.monotonic()>=deadline:
            result={'status':'SAVED_OBSERVATION_NOT_READY_NOT_TRAINING_TERMINAL',
                    'receipt':str(path),'error':str(exc)}
            break
        time.sleep(.1)
print(json.dumps(result,ensure_ascii=True,allow_nan=False),flush=True)
