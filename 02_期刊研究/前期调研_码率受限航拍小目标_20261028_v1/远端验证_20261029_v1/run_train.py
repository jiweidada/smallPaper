from pathlib import Path
import argparse,json,sys,traceback
import torch
from ultralytics import YOLO
from study import ROOT,StudyTrainer,PreflightComplete,training_args,json_write
parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['D0','DA'],required=True);parser.add_argument('--batch',type=int,required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
StudyTrainer.ARM=args.arm;StudyTrainer.PREFLIGHT=args.preflight
kw=training_args(args.arm,args.batch,args.preflight)
assert not (ROOT/'results'/kw['name']).exists(),'Refusing to overwrite/reuse a previous run'
model=YOLO(str(ROOT/'assets/common_init_10class.pt'))
try:
    model.train(trainer=StudyTrainer,**kw)
except PreflightComplete:
    print(json.dumps({'status':'PREFLIGHT_PASS','arm':args.arm,'batch':args.batch}),flush=True)
    sys.exit(0)
except Exception as error:
    traceback.print_exc()
    folder=ROOT/'results'/kw['name'];folder.mkdir(parents=True,exist_ok=True)
    json_write(folder/'failure.json',{'status':'FAILED','type':type(error).__name__,'message':str(error),'preflight':args.preflight,
               'oom':isinstance(error,torch.cuda.OutOfMemoryError) or 'out of memory' in str(error).lower(),
               'automatic_restart':False})
    sys.exit(1)
print(json.dumps({'status':'TRAINING_COMPLETED','arm':args.arm}),flush=True)
