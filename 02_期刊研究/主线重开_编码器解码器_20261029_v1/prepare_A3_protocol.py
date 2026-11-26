from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import py_compile
import tarfile

BASE=Path(__file__).resolve().parent
ROOT=BASE/'A3_中心修正一对一约束核验_v1'
assert not (ROOT/'freeze_manifest.json').exists()
archive=BASE/'A0_冻结模型诊断结果_v1/backup/A0_A0b_bounded_diagnostics.tar.gz'
receipt=json.loads((archive.parent/'archive_receipt.json').read_text(encoding='utf-8'))
assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['archive_sha256']
with tarfile.open(archive) as source:
    manifest=json.load(source.extractfile('backup/file_manifest.json'))
inputs=[r for r in manifest if r['path'].startswith('traces/') and r['path'].endswith('.npz')]
assert len(inputs)==100
a2=BASE/'A2_已有预测框误差分解_v2/per_gt.jsonl'
records=[json.loads(line) for line in a2.read_text(encoding='utf-8').splitlines()]
uids=sorted({r['uid'] for r in records if r['primary']})
assert len(uids)==470
now=datetime.now(timezone.utc).isoformat()
protocol=dict(material_passport=dict(origin_skill='academic-research-suite',origin_mode='conditional existing-prediction capacity audit',
    origin_date=now,verification_status='FROZEN_NOT_EXECUTED',version='A3_capacity_v1'),
    status='FROZEN_BEFORE_CAPACITY_RESULTS',reason='A2 per-GT center oracle may reuse queries;check preservation and one-to-one capacity before proposing ownership/routing mechanisms.',
    input_A0_archive_sha256=receipt['archive_sha256'],input_A2_gt_table_sha256=hashlib.sha256(a2.read_bytes()).hexdigest(),
    primary_uids=uids,stage='final decoder3',iou_threshold=.5,
    primary_class_mode='native_post300:hold original flatTop300 query/class pairs and scores fixed;query itself may be used only once',
    secondary_modes=['argmax class','geometry only'],
    candidate_restriction='For the unchanged470 A2 persistent-miss GT,only selected native stride8 query anchors within the GT-local3x3 grid neighborhood receive GT-center oracle edges;native width/height are unchanged.',
    baseline='Canonical maximum-cardinality native GT-query coverage,then maximum IoU;dummy columns allow no match. Its matched GT set is the protected set.',
    constraints=['Unconstrained per-GT oracle potential,which may reuse queries',
                 'Keep exact baseline query-GT pairs;only previously unused queries may rescue new GT',
                 'Allow reassigning queries,but retain all protected baseline GT and maximize rescued primary GT'],
    primary_endpoint='Pooled fraction of unconstrained native-post300 oracle-rescuable primary GT recovered using previously unused queries;2000 filename-group cluster bootstrap,seed20261030',
    decisions=dict(enough='at least100 unconstrained rescues and15 groups',
                   low_conflict='ratio>=0.80 and95% lower>=0.65:do not justify a complex ownership module from oracle query-capacity conflict',
                   material_conflict='ratio<=0.50 and95% upper<=0.70:prioritize correspondence diagnosis,not yet any new method',
                   otherwise='inconclusive;no routing module approved'),
    limitations=['GT-oracle bounds,not deployable corrections,AP gains or an original training-owner reconstruction.',
                 'Native scores and selection stay fixed;classification can still limit precision/recall.',
                 'The author checkpoint saw training images;filename groups are a heuristic.',
                 'Protecting a canonical maximum matching is weaker than preserving every GT covered in many-to-one geometry.'],
    budget=dict(new_full_model_forwards=0,new_readout_fits=0,new_model_updates=0,existing_global_forward_count=185,
                existing_global_forward_cap=200,CPU_hard_timeout_seconds=180,automatic_retry=False))
(ROOT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(ROOT/'input_manifest.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8')
py_compile.compile(str(ROOT/'analyze_capacity.py'),doraise=True)
files=[dict(file=name,sha256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),bytes=(ROOT/name).stat().st_size)
       for name in ('analyze_capacity.py','protocol.json','input_manifest.json')]
(ROOT/'freeze_manifest.json').write_text(json.dumps(dict(frozen_utc=now,files=files),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='A3_CAPACITY_PROTOCOL_FROZEN',images=100,primary_gt=470,new_forwards=0,new_fits=0,
                     freeze_sha256=hashlib.sha256((ROOT/'freeze_manifest.json').read_bytes()).hexdigest())))
