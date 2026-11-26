"""Observe genuine worker identities and export the next accepted epoch."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import pair
from collect_metrics import export


def observe(server):
    with pair.ssh.connect(server.port) as client:
        live = pair.live_status(client, server)
        segments = pair.recovery_segments(server.arm)
        directory = segments[-1] if segments else HERE / server.arm
        plan = json.loads((directory / 'recovery_plan.json').read_text())
        first = directory / 'first_resumed_live.json'
        if (not first.exists() and live['worker_live'] and live['receipt_worker_bound']
                and live['optimizer_attempts'] > plan['accepted_rounds'] * 1617):
            with client.open_sftp() as sftp:
                def read(remote):
                    with sftp.open(str(server.control / remote), 'rb') as stream:
                        return json.loads(stream.read())
                supervisor = read('supervisor.json')
                receipt = read('run/receipt.json')
            assert receipt['pid'] == supervisor['pid'] == live['worker_pid']
            assert receipt['resumed_from_checkpoint_sha256'] == plan['checkpoint_sha256']
            assert receipt['recovery_plan_sha256'] == supervisor['recovery_plan_sha256']
            pair.save(first, {'supervisor': supervisor, 'receipt': receipt, 'observed_live': live})
        return live


def main():
    prior = json.loads((pair.HERE / '逐轮导出核对.json').read_text(encoding='utf-8-sig'))
    targets = {arm: count + 1 for arm, count in prior['accepted_rounds'].items() if count < 320}
    deadline = time.monotonic() + 570
    with ThreadPoolExecutor(max_workers=2) as pool:
        while time.monotonic() < deadline:
            records = list(pool.map(observe, pair.SERVERS))
            print(json.dumps([{k: record[k] for k in ('arm', 'phase', 'batch', 'optimizer_attempts',
                                                      'worker_live', 'backup_client_live', 'backup')}
                              for record in records]), flush=True)
            for record in records:
                training = record['worker_live'] and record['backup_client_live']
                successful_exit = (record['worker_status'] == 'COMPLETED'
                                   and record['completed_rounds'] == 320
                                   and record['supervisor_status'] in ('RUNNING', 'COMPLETED')
                                   and record['exit_code'] in (None, 0)
                                   and (record['supervisor_live'] or record['supervisor_status'] == 'COMPLETED'))
                finishing = successful_exit and (record['backup_client_live']
                             or record['backup']['status'] == 'COMPLETED_AND_BACKED_UP')
                if not (training or finishing):
                    raise RuntimeError(json.dumps(record))
            if not targets or any(r['backup'].get('epoch', -1) + 1 >= targets[r['arm']]
                                  for r in records if r['arm'] in targets):
                export()
                return
            time.sleep(30)
    print('Observation window ended while the last confirmed workers were live.', flush=True)


if __name__ == '__main__':
    main()
