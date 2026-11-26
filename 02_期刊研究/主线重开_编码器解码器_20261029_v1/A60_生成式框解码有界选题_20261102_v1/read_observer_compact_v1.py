"""Read one existing SSH receipt and print bounded progress; no remote actions."""
from pathlib import Path
import json
import subprocess
import sys

p = Path(__file__).resolve().parent
r = subprocess.run([sys.executable, str(p/'read_observer_receipt_v1.py'),
                    '--number', str(int(sys.argv[1])), '--wait', '5'],
                   capture_output=True, text=True, check=True)
a = json.loads(r.stdout)
o = a['observation']
assert o['source_bound'] and o['protocol_sha256'] == '9243a7b198bc23eead5c5e32ad680f4ed91a2cd0daa5e6f4f05ebd2892f8c555'
q = o['arms']['Q']
terminal = o.get('supervisor_exit')
if terminal is None:
    assert o['supervisor']['live'] and o['supervisor']['same_start'], o['supervisor']
    if q.get('exit') is None:
        assert q['process']['live'] and q['process']['same_start'], q['process']
        assert q['process']['pid'] == 39733 and q['process']['actual_start_ticks'] == '33850461'
else:
    assert terminal['status'] == 'ALL_THREE_FIXED12_COMPLETED' and terminal['exit'] == 0, terminal
    assert terminal['remaining_queue'] == [] and terminal['completed_arms'] == ['B','M','Q']
completed = q['epochs'] if isinstance(q['epochs'], int) else q['completed_epochs']
print(json.dumps({'receipt':int(sys.argv[1]),'client_utc':a['recorded_client_utc'],
    'remote_utc':o['utc'],'source_bound':True,'Q_process':q['process'],
    'Q_progress':q['progress'],'Q_completed_epochs':completed,'Q_evaluations':q['evaluations'],
    'Q_exit':q['exit'],'supervisor_exit':terminal,'terminal_complete':terminal is not None},
    ensure_ascii=True))
