"""Export accepted 320-round MSCA capacity comparison rows and provenance."""

import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path

from pair import BACKUP, HERE, SERVERS


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path, payload):
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as stream:
        stream.write(payload)
    temporary.replace(path)


def export():
    rows, sources, counts = [], [], {}
    trained_status = {}
    for server in SERVERS:
        control = BACKUP / server.arm
        archive = control / 'backups'
        spec = importlib.util.spec_from_file_location('capacity_protocol_' + server.arm, control / 'backup_protocol.py')
        protocol = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol)
        assert protocol.ARM == server.arm
        validator_sha = sha(control / 'checkpoint_validator.py')
        launch = read(HERE / f'launch_{server.arm}.json')
        assert protocol.PROTOCOL == launch['protocol_sha256']
        accepted = []
        for stage in sorted((archive / 'epochs').glob('epoch[0-9][0-9][0-9][0-9]')):
            if not (stage / 'accepted.json').is_file():
                continue
            epoch = int(stage.name[5:])
            assert epoch == len(accepted) and epoch <= protocol.FINAL_EPOCH
            manifest_data = (stage / 'manifest.json').read_bytes()
            manifest = protocol.parse_manifest(manifest_data, server.arm, epoch)
            assert (stage / 'ack.json').read_bytes() == protocol.ack_bytes(manifest_data, manifest)
            assert (stage / 'accepted.json').read_bytes() == protocol.accepted_bytes(manifest_data, manifest)
            protocol.check_metric_files(manifest, lambda name: protocol.local_metadata(stage / name))
            proof = read(stage / 'payload_validation.json')
            assert proof['status'] == 'CHECKPOINT_PAYLOADS_VERIFIED_CPU_ONLY' and proof['epoch'] == epoch
            assert proof['arm'] == server.arm and proof['protocol_sha256'] == protocol.PROTOCOL
            assert proof['manifest_sha256'] == hashlib.sha256(manifest_data).hexdigest()
            assert proof['validator_sha256'] == validator_sha
            assert any(p.get('complete_state_and_rng_fields_verified') for p in proof['checked_payloads'])
            row_path = stage / 'epoch.metrics.json'
            row = read(row_path)
            assert row['epoch'] == epoch and row['optimizer_step_attempts'] == (epoch + 1) * protocol.BATCHES
            assert set(row['metrics']) == protocol.METRICS and all(math.isfinite(v) for v in row['metrics'].values())
            rows.append({'arm':server.arm, 'round':epoch + 1, 'optimizer_step_attempts':row['optimizer_step_attempts'],
                         **row['metrics'], 'train_loss_mean':row['train_loss_mean'],
                         'evaluated_checkpoint_sha256':row['evaluated_checkpoint_sha256'],
                         'evaluated_ema_tensor_sha256':row['ema_tensor_sha256'], 'protocol_sha256':protocol.PROTOCOL})
            sources.append({'arm':server.arm, 'round':epoch + 1, 'metrics_file':str(row_path),
                            'metrics_sha256':sha(row_path), 'manifest_sha256':proof['manifest_sha256'],
                            'validation_sha256':sha(stage/'payload_validation.json'), 'acceptance_sha256':sha(stage/'accepted.json')})
            accepted.append(epoch)
        counts[server.arm] = len(accepted)
        trained_status[server.arm] = read(archive/'monitor_status.json')['status']

    csv_path = HERE / '逐轮指标_MSCA容量匹配.csv'
    report = {'accepted_rounds':counts,'backup_status':trained_status,'records':len(rows),'sources':sources,
              'primary_endpoint_audit_pending':True}
    if rows:
        if csv_path.exists():
            old_rows = {(r['arm'], int(r['round'])): r for r in
                        csv.DictReader(io.StringIO(csv_path.read_text(encoding='utf-8-sig')))}
            new_rows = {(r['arm'], r['round']): {key:str(value) for key,value in r.items()} for r in rows}
            assert all(new_rows.get(k) == r for k,r in old_rows.items()), 'Previously exported row changed'
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        atomic_write(csv_path, stream.getvalue().encode('utf-8-sig'))
        report['csv_sha256'] = sha(csv_path)
    atomic_write(HERE/'逐轮导出核对.json', (json.dumps(report, ensure_ascii=False, indent=2)+'\n').encode())
    print(json.dumps({'accepted_rounds':counts,'backup_status':trained_status,'records':len(rows)},ensure_ascii=False))


if __name__ == '__main__':
    export()
