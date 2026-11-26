"""Inspect current kernel uptime and old process identities without changing the server."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import platform

def identity(pid, expected):
    try:
        data = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'state': data[0], 'start_ticks': data[19],
                'same_as_previous': data[19] == expected,
                'live': data[0] not in ['Z', 'X']}
    except (FileNotFoundError, ProcessLookupError):
        return {'pid': pid, 'present': False, 'same_as_previous': False, 'live': False}

ticks_per_second = os.sysconf('SC_CLK_TCK')
uptime_seconds = float(Path('/proc/uptime').read_text().split()[0])
boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
start_times = [(1193, '39931198'), (1194, '39931668'), (564, '42790973'), (567, '42791005')]
out = {'observed_remote_utc': datetime.now(timezone.utc).isoformat(), 'hostname': platform.node(),
       'boot_id': boot_id, 'uptime_seconds': uptime_seconds, 'clock_ticks_per_second': ticks_per_second,
       'current_uptime_ticks_approx': int(uptime_seconds * ticks_per_second),
       'prior_process_start_ticks': {str(pid): tick for pid, tick in start_times},
       'prior_starts_exceed_current_uptime': {str(pid): int(tick) > int(uptime_seconds * ticks_per_second)
                                               for pid, tick in start_times},
       'previous_processes': [identity(pid, tick) for pid, tick in start_times],
       'scope': 'Current kernel and process identities only. A reset or migration must not be inferred solely from wall-clock offsets.'}
print(json.dumps(out))
