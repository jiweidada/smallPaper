from pathlib import Path
import datetime,hashlib,json,sys
import torch,torchvision,ultralytics,yaml
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT
from study import ROOT,state_hash,json_write
assert ultralytics.__version__=='8.4.191'
pins=json.loads((ROOT/'assets/source_pin.json').read_text())
package=Path(ultralytics.__file__).parent
for row in pins:
    raw=(package/row['relative']).read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(raw).hexdigest()==row['normalized_sha256'],row['relative']
assert torch.cuda.is_available()
source_manifest=json.loads((ROOT/'assets/dataset_source_manifest.json').read_text())
for row in source_manifest['files']:
    image_path=ROOT/'dataset/train/images'/row['file_name']
    with image_path.open('rb') as stream:stream.seek(-2,2);assert stream.read()==b'\xff\xd9',('JPEG repair would be required',row['file_name'])
    from PIL import Image
    with Image.open(image_path) as image:assert image.getexif().get(274,1)==1,('Unexpected EXIF orientation',row['file_name'])
_ = torchvision.ops.nms(torch.tensor([[0.,0.,4.,4.]],device='cuda'),torch.tensor([.9],device='cuda'),.5)
public_manifest=json.loads((ROOT/'assets/public_asset_manifest.json').read_text())
for a in public_manifest:
    path=ROOT/'assets'/a['name']
    with path.open('rb') as stream:h=hashlib.sha256(stream.read()).hexdigest()
    assert h==a['sha256'] and path.stat().st_size==a['bytes']
torch.manual_seed(0)
public=YOLO(str(ROOT/'assets/yolo26s.pt'))
source_head=public.model.model[-1]
assert source_head.cv2 is not None and source_head.one2one_cv2 is not None,'Public initialization must retain both unfused heads'
cfg=yaml.safe_load((package/'cfg/models/26/yolo26.yaml').read_text());cfg.update(nc=10,scale='s')
torch.manual_seed(0)
model=DetectionModel(cfg,ch=3,nc=10,verbose=False)
model.load(public.model)
classes=json.loads((ROOT/'receipts/data_preparation.json').read_text())['classes']
model.names=dict(enumerate(classes));model.args=dict(DEFAULT_CFG_DICT)
path=ROOT/'assets/common_init_10class.pt'
assert not path.exists(),'Do not overwrite an existing initialization'
ckpt={'model':model.float().cpu(),'ema':None,'epoch':-1,'optimizer':None,'updates':0,
      'train_args':dict(DEFAULT_CFG_DICT,model=str(path),task='detect',data=str(ROOT/'dataset/data.yaml'),imgsz=640),
      'version':ultralytics.__version__,'date':datetime.datetime.now(datetime.timezone.utc).isoformat()}
torch.save(ckpt,path)
loaded=YOLO(str(path)).model.float().cpu()
assert state_hash(loaded)==state_hash(model)
with path.open('rb') as stream:digest=hashlib.sha256(stream.read()).hexdigest()
json_write(ROOT/'receipts/common_initialization.json',{'status':'PASS','checkpoint':str(path),'checkpoint_sha256':digest,
    'state_sha256':state_hash(loaded),'public_checkpoint_sha256':next(a['sha256'] for a in public_manifest if a['name']=='yolo26s.pt'),
    'public_heads_retained':True,'classes':10,'random_head_seed':0,'transfer':'native DetectionModel.load, shape-compatible pretrained tensors',
    'source_pin_files':len(pins),'cuda_nms_check':'PASS','scientific_hypothesis_validated':False})
print((ROOT/'receipts/common_initialization.json').read_text())
