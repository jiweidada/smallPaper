"""Stop only the orphaned read-only preflight from a disconnected SSH session."""
import json
import os
import pathlib
import signal
import sys
import time

pid=int(sys.argv[1]);expected_ticks=int(sys.argv[2]);path=pathlib.Path('/proc')/str(pid)


def identity():
    try:
        stat=(path/'stat').read_text().rsplit(')',1)[1].split()
        args=[x.decode(errors='replace') for x in (path/'cmdline').read_bytes().split(b'\0') if x]
        return {'pid':pid,'ppid':int(stat[1]),'start_ticks':int(stat[19]),'state':stat[0],'argv':args}
    except FileNotFoundError:
        return None


row=identity()
if row is None or row['state']=='Z':
    print(json.dumps({'target_pid':pid,'already_exited':True}));raise SystemExit(0)
assert row['start_ticks']==expected_ticks and row['ppid']==1
assert row['argv']==[sys.executable,'-u','-'], 'Not the orphaned stdin preflight process'
os.kill(pid,signal.SIGTERM)
deadline=time.monotonic()+15
while time.monotonic()<deadline:
    current=identity()
    if current is None or current['state']=='Z':
        print(json.dumps({'target':row,'terminated':True}));raise SystemExit(0)
    assert current['start_ticks']==expected_ticks
    time.sleep(.2)
print(json.dumps({'target':row,'termination_requested':True,'still_waiting':identity()}))
raise SystemExit(2)
