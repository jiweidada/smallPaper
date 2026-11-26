"""Collect logs/metadata only, preserving older result collections."""
import base64
import concurrent.futures
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys
from zoneinfo import ZoneInfo

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
now = datetime.datetime.now(datetime.timezone.utc)
stamp = now.strftime('%Y%m%dT%H%M%S_%fZ')
DEST = ROOT/'英文小论文/data/06_第二轮_20261014'/('best_stg2整理_20261015_'+stamp)
DEST.mkdir(exist_ok=False)
HELPER = HERE.with_name('phase_round2_20261014_v2')/'ssh_control.py'
ENVS = {'A': '/25085400120/envs/deim/bin/python',
        'B': '/24085403014/deim_migration_runtime/envs/deim/bin/python'}
INIT = '85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91'


def collect(account):
    command = [sys.executable, '-X', 'utf8', '-B', str(HELPER), account, '--python', ENVS[account],
               '--script', str(HERE/'probe_completed_round2_20261015.py'), '--timeout', '180']
    process = subprocess.run(command, capture_output=True, timeout=190)
    (DEST/(account+'.stdout.json')).write_bytes(process.stdout)
    (DEST/(account+'.stderr.txt')).write_bytes(process.stderr)
    if process.returncode:
        raise RuntimeError(account+': '+process.stderr.decode(errors='replace')[-1500:])
    result = json.loads(process.stdout)
    assert result['common_init_sha256'] == INIT, account
    for row in result['runs']:
        folder = DEST/row['job']
        folder.mkdir(exist_ok=False)
        for item in row['files']:
            raw = base64.b64decode(item.pop('base64'))
            assert hashlib.sha256(raw).hexdigest() == item['sha256']
            with (folder/item['name']).open('xb') as handle:
                handle.write(raw)
        row['common_init_sha256'] = result['common_init_sha256']
        row['collected_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        row['collected_los_angeles'] = now.astimezone(ZoneInfo('America/Los_Angeles')).isoformat()
        with (folder/'checkpoint_log_binding.json').open('x', encoding='utf-8') as handle:
            json.dump(row, handle, ensure_ascii=False, indent=2)
    with (DEST/(account+'_collection.json')).open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return {'account': account, 'completed': [r['job'] for r in result['runs']],
            'unfinished': result['unfinished'],
            'metrics': [{'job': r['job'], 'best_round': r['best_round_one_based'],
                         'AP': r['metrics_raw_0_to_1'][0]*100, 'APs': r['metrics_raw_0_to_1'][3]*100}
                        for r in result['runs']]}


if __name__ == '__main__':
    print(json.dumps({'destination': str(DEST), 'checked_los_angeles': now.astimezone(ZoneInfo('America/Los_Angeles')).isoformat()}, ensure_ascii=False), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(collect, ['A', 'B']))
    for result in results:
        print(json.dumps(result, ensure_ascii=False), flush=True)
