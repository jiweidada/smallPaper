"""Inject the local encrypted credential into the reviewed backup client."""
from pathlib import Path
import sys
from ssh_control import HOST, PORTS, credential_path, protect

control=Path(sys.argv[sys.argv.index('--local-root')+1]).resolve(strict=True)
allowed=Path('D:/MPSMA_DEIM_experiment_backups/BASE_FULL_FRESH_20261112_v1').resolve(strict=True)
assert control.parent==allowed and control.name in ['BASE_FRESH','FULL_FRESH']
sys.path.insert(0,str(control))
import backup_monitor

def password_provider(host,port,user):
    assert host==HOST and port in PORTS and user=='root'
    return protect(credential_path(port,"backup").read_bytes(),decode=True).decode('utf-8')

backup_monitor.main(password_provider=password_provider)
