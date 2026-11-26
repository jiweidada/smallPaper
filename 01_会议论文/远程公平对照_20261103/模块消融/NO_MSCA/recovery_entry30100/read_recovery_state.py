"""Read the supplied server identity and the original MSCA run, without mutation."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess

root = Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1/no_msca_20261107_v1')
keys = ('arm', 'status', 'phase', 'pid', 'start_ticks', 'actual_loop_epoch', 'completed_epochs',
        'next_epoch', 'batch', 'optimizer_step_attempts', 'last_loss', 'updated_utc', 'ended_utc',
        'error', 'error_type', 'traceback', 'exit_code', 'protocol_sha256', 'configuration_sha256')
result = {'observed_server_utc': datetime.now(timezone.utc).isoformat(),
          'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'original_run_path': str(root), 'original_run_exists': root.is_dir(),
          'environment_python_exists': Path('/25085400120/envs/deim/bin/python').is_file(),
          'project_processes': []}
if root.is_dir():
    result['root_files'] = [{'name': p.name, 'bytes': p.stat().st_size, 'directory': p.is_dir()}
                            for p in sorted(root.iterdir())]
    for label, path in [('receipt', root / 'run/receipt.json'), ('supervisor', root / 'supervisor.json')]:
        if path.is_file():
            data = json.loads(path.read_text())
            result[label] = {k: data[k] for k in keys if k in data}
            if label == 'supervisor':
                result[label].update({k: data[k] for k in ['supervisor_pid', 'supervisor_start_ticks', 'boot_id'] if k in data})
                if 'final_receipt' in data:
                    result[label]['final_receipt'] = {k: data['final_receipt'][k] for k in keys if k in data['final_receipt']}
    protocol = root / 'execution_protocol.json'
    if protocol.is_file():
        result['execution_protocol_sha256'] = sha256(protocol.read_bytes()).hexdigest()
    gate = root / 'backup_gate'
    if gate.is_dir():
        result['backup_gate_entries'] = [p.name for p in sorted(gate.iterdir())]
    run = root / 'run'
    if run.is_dir():
        result['run_files'] = [{'name': p.name, 'bytes': p.stat().st_size, 'directory': p.is_dir()}
                               for p in sorted(run.iterdir())]
    for config in [root / 'execution_configs/NO_MSCA.yml']:
        if config.is_file():
            data = json.loads(config.read_text())
            result['dataset_paths'] = {}
            for split in ['train', 'val']:
                dataset = data[split + '_dataloader']['dataset']
                result['dataset_paths'][split] = {k: {'path': dataset[k], 'exists': Path(dataset[k]).exists()}
                                                   for k in ['img_folder', 'ann_file']}
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        raw = (proc / 'cmdline').read_bytes()
        if str(root).encode() not in raw:
            continue
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        result['project_processes'].append({'pid': int(proc.name), 'state': fields[0],
                                            'start_ticks': int(fields[19]),
                                            'command': raw.replace(b'\x00', b' ').decode(errors='replace').strip()})
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
result['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], text=True).strip()
result['gpu_processes'] = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True).strip()
print(json.dumps(result))
