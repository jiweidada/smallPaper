import concurrent.futures
import base64
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
DESTINATION = HERE.parents[2]/'英文小论文/data/06_第二轮_20261014/结果评估_20261014'
HELPER = HERE.with_name('phase_round2_20261014_v2')/'ssh_control.py'
ENVIRONMENTS = {'A': '/25085400120/envs/deim/bin/python', 'B': '/24085403014/deim_migration_runtime/envs/deim/bin/python'}


def collect(account):
    completed = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(HELPER), account,
        '--python', ENVIRONMENTS[account], '--script', str(HERE/'probe_completed_for_analysis.py'), '--timeout', '180'],
        capture_output=True, timeout=190)
    if completed.returncode:
        raise RuntimeError(account+': '+completed.stderr.decode(errors='replace')[-2000:])
    data = json.loads(completed.stdout)
    folder = DESTINATION/(data['arm'].replace('-','_')+'_seed'+str(data['seed']))
    folder.mkdir(parents=True, exist_ok=False)
    for item in data['files']:
        raw = base64.b64decode(item.pop('base64'))
        assert hashlib.sha256(raw).hexdigest() == item['sha256']
        with (folder/item['name']).open('xb') as handle:
            handle.write(raw)
    data['collected_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (folder/'checkpoint_log_binding.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'account':account, 'folder':str(folder), 'best_round':data['checkpoint_epoch_zero_based']+1,
            'AP':data['metrics_raw_0_to_1'][0]*100, 'APs':data['metrics_raw_0_to_1'][3]*100,
            'raw_epoch_count':len(data['epochs_in_raw_log'])}


if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(collect,['A','B']):
            print(json.dumps(result,ensure_ascii=False))
