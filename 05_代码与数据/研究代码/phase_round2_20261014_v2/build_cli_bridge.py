import base64
import hashlib
import pathlib
here=pathlib.Path(__file__).resolve().parent
names=['dump_preds.py','export_cli_predictions.py','run_translation_suite.py']
payload={name:{'base64':base64.b64encode((here/name).read_bytes()).decode(),'sha256':hashlib.sha256((here/name).read_bytes()).hexdigest()} for name in names}
source=(here/'cli_bridge_template.py').read_text().replace('PAYLOAD = {}  # INSERT','PAYLOAD = '+repr(payload))
with (here/'cli_bridge_remote.py').open('x',encoding='utf-8',newline='\n') as f:f.write(source)
