from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parent
path=ROOT/'13_架构检索增补记录.json'
log=json.loads(path.read_text(encoding='utf-8'))
groups=[dict(name='A1_A2 center correspondence and parameterization clearance',queries=[
    '"DETR" "soft-argmax" "center"','"DETR" "weighted coordinates" detection',
    '"Slim-DETR" "Gaussian"','"DETR" "grid-relative"',
    'site:arxiv.org "Slim-DETR"','site:ecva.net "Slim-DETR"',
    '"DETR" "sigmoid" "translation equivariance"','"DETR" "inverse sigmoid" "bias"']),
    dict(name='A1_A2 center and geometry discovery',queries=[
    '"DETR" "boundary" "queries" "small"','"DETR" "center" "size" "decoupled" regression',
    '"tiny object" "box" "refinement" decoder']),
    dict(name='A1 actual positional source cross-check',queries=[
    '"ShareDAB" "position" decoder','"DETR" "query positional" "reuse"',
    '"DEIMv2" "query positional" shared','"DETR" "stale" "positional"'])]
existing={r['name'] for r in log['query_groups']}
log['query_groups'].extend(r for r in groups if r['name'] not in existing)
log['raw_architecture_excerpt_inventory']=[dict(file=str(p.relative_to(ROOT)),bytes=p.stat().st_size,
    sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted((ROOT/'source_audit_v1').glob('arch_*.json'))]
log['latest_discovery_limits']=dict(Slim_DETR='Third-party discovery only;primary paper not verified;excluded as confirmed novelty evidence',
    QCAR='Primary publisher abstract/highlights/contribution excerpts only',CLSC_DETR='Primary indexed abstract only',
    No_search_absence_claim='No exact-match result is not proof of novelty')
path.write_text(json.dumps(log,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
state=json.loads((ROOT/'15_当前状态与结果校验.json').read_text(encoding='utf-8'))
board=json.loads((ROOT/'01_候选判别表.json').read_text(encoding='utf-8'))
index=json.loads((ROOT/'19_A1实际路径与冻结协议索引.json').read_text(encoding='utf-8'))
assert state['model_forwards']==185 and state['CPU_diagnostic_readout_fits']==32 and not board['mainline_validated']
assert index['status']=='COMPLETED_WITH_MODEL_STATE_PRESERVED'
receipt=json.loads((ROOT/'21_A1A2完成与核验回执.json').read_text(encoding='utf-8'))
assert hashlib.sha256((ROOT/'20_A1A2结果与当前候选主线.txt').read_bytes()).hexdigest()==receipt['report_sha256']
print(json.dumps(dict(status='CURRENT_STATE_AND_REPORT_HASH_CONSISTENT',full_model_forwards=185,
                     diagnostic_readout_fits=32,mainline_validated=False,query_groups=len(log['query_groups']))))
