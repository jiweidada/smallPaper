from pathlib import Path
import json,hashlib
from ultralytics import YOLO
from ultralytics.utils import SETTINGS,events
SETTINGS.update({'sync':False});events.enabled=False
root=Path(__file__).resolve().parent
path=root/'public_assets/yolo26s.pt'
assert hashlib.sha256(path.read_bytes()).hexdigest()=='646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b'
wrapper=YOLO(str(path));head=wrapper.model.model[-1]
r={'status':'PASS','source':'official Ultralytics v8.4.0 release','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
   'public_nc':head.nc,'one2many_retained':head.cv2 is not None,'one2one_retained':head.one2one_cv2 is not None,
   'weights_loaded':1,'explicit_forwards':0,'training_updates':0,'class_adaptation_pending':True}
assert r['one2many_retained'] and r['one2one_retained']
(root/'public_weight_inspection.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r))
