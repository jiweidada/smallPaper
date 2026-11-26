"""Read-only evidence collection in distinct timestamped local directories."""
import base64
import concurrent.futures
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
DESTINATION = WORKSPACE/'英文小论文/data/06_第二轮_20261014/execution_v3'
HELPER = HERE.with_name('phase_round2_20261014_v2')/'ssh_control.py'
SOURCE = r'''
import base64, hashlib, json, pathlib
phase = pathlib.Path('/25085400120/DEIM/experiments/phase_mechanism_20261012' if pathlib.Path('/25085400120/DEIM').is_dir() else '/24085403014/experiments/phase_mechanism_20261012')
stage = phase/'artifacts/round2_20261014_v1/execution_v3'
items=[]
for name in ['preparation_receipt.json','scheduler_tests.log','branch_tests.log','branch_acceptance.log',
             'analysis_tests.log','soep_tests.log','gpu_preflight.log','handoff_committed.json',
             'launch_receipt.json','previous_v2_state.json','queue_state.json']:
    p=stage/name
    data=p.read_bytes()
    items.append({'name':name,'remote':str(p),'sha256':hashlib.sha256(data).hexdigest(),
                  'base64':base64.b64encode(data).decode()})
print(json.dumps(items))
'''


def collect(account):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root = DESTINATION/(account+'_'+stamp)
    root.mkdir(parents=True, exist_ok=False)
    request = root/'read_only_request.py'
    request.write_text(SOURCE, encoding='utf-8')
    process = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(HELPER), account,
                              '--script', str(request), '--timeout', '60'], capture_output=True, timeout=70)
    if process.returncode:
        raise RuntimeError(account+': '+process.stderr.decode(errors='replace')[-1500:])
    manifest=[]
    for item in json.loads(process.stdout):
        raw=base64.b64decode(item.pop('base64'))
        assert hashlib.sha256(raw).hexdigest()==item['sha256']
        with (root/item['name']).open('xb') as output:
            output.write(raw)
        manifest.append(item)
    receipt={'account':account,'collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':manifest}
    (root/'collection_manifest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'account':account,'files':len(manifest),'destination':str(root)}


if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(collect,['A','B']):
            print(json.dumps(result,ensure_ascii=False))
