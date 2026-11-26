from pathlib import Path
import json
src=Path('英文小论文/code/stem_joint_error_analysis_20261011_v1/analyze.py')
dst=Path('英文小论文/code/stem_joint_error_analysis_20261011_v2')
assert not dst.exists();dst.mkdir()
s=src.read_bytes()
assert b"old['uncertain_ranking_reconstruction']" in s
s=s.replace(b"old['uncertain_ranking_reconstruction']",b"old.get('uncertain_ranking_reconstruction',0)")
(dst/'analyze.py').write_bytes(s)
out=Path('outputs/stem_joint_error_analysis_20261011_v1');assert not out.exists();out.mkdir()
(out/'failure.json').write_text(json.dumps(dict(status='READ_ONLY_CHECK_STOPPED_BEFORE_RESULT_EXPORT',
    error='Prior ranking group reports omit zero-valued cells; direct key lookup failed.',
    repair='v2 reader treats omitted uncertain cell as zero, without changing any saved array or learned model.',
    model_forwards=0,optimizer_updates=0,training_epochs=0),indent=2),encoding='utf8')
print('READER_V2_CREATED_ORIGINAL_FAILURE_PRESERVED')
