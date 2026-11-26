"""Preserve failed validation artifacts; append corrected queue tooling as v2."""
import base64
import hashlib
import pathlib

here = pathlib.Path(__file__).resolve().parent
names = ['activate_remote.py','dump_preds.py','analyze_preds.py','test_prediction_analysis.py',
         'run_translation_suite.py','soep_lite_round2.py','test_soep_lite.py','train_round2.py',
         'benchmark_soep_lite.py','run_round2_queue.py']
payload = {name: {'base64': base64.b64encode((here/name).read_bytes()).decode(),
                  'sha256': hashlib.sha256((here/name).read_bytes()).hexdigest()} for name in names}
source = (here/'upgrade_v2_template.py').read_text().replace('PAYLOAD = {}  # INSERT', 'PAYLOAD = '+repr(payload))
with (here/'upgrade_v2_remote.py').open('x', encoding='utf-8', newline='\n') as f:
    f.write(source)
print('v2 append-only upgrade prepared')
