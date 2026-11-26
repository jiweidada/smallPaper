import base64
import hashlib
import pathlib

here = pathlib.Path(__file__).resolve().parent
raw = (here/'dump_preds.py').read_bytes()
payload = {'base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest()}
source = (here/'fusion_fix_template.py').read_text().replace('PAYLOAD = {}  # INSERT','PAYLOAD = '+repr(payload))
with (here/'fusion_fix_remote.py').open('x',encoding='utf-8',newline='\n') as f:f.write(source)
