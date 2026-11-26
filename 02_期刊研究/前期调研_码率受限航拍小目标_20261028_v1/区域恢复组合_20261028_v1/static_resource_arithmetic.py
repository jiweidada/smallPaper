"""Deterministic arithmetic from existing train metadata and read source dimensions. No model import/construction."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
p=Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文\相位机制论文_20261018\修订_20261018_v2\data\annotations\train\converted.json')
digest=hashlib.sha256(p.read_bytes()).hexdigest()
assert digest=='32c1cae852e084430ea74503eb660f0707ec6b5e65d0998ce3ec8b448265c883'
images=json.loads(p.read_text(encoding='utf-8'))['images']
largest=max(images,key=lambda x:x['width']*x['height'])
conv=lambda a,b,k:a*b*k*k+b
res=lambda c:2*conv(c,c,3)
linear=lambda a,b:a*b+b
parts={
 'head':conv(3,64,3),
 'down1':4*res(64)+conv(64,128,2),
 'down2':4*res(128)+conv(128,256,2),
 'down3':4*res(256)+conv(256,512,2),
 'body_encoder':4*res(512),'body_decoder':4*res(512),
 'up3':conv(512,256,2)+4*res(256),
 'up2':conv(256,128,2)+4*res(128),
 'up1':conv(128,64,2)+4*res(64),'tail':conv(64,3,3),
 'qf_predictor':4*res(512)+2*linear(512,512)+linear(512,1),
 'qf_embed':linear(1,512)+2*linear(512,512),
 'modulations':2*sum(linear(512,c) for c in [256,128,64])}
params=sum(parts.values())
maxw=((largest['width']+7)//8)*8;maxh=((largest['height']+7)//8)*8
skip_floats=sum((maxw//(2**i))*(maxh//(2**i))*c for i,c in enumerate([64,128,256,512]))
receipt={'purpose':'Static resource planning only, not profiling or measured capacity.',
 'source_train_metadata':str(p),'source_sha256':digest,'images':len(images),
 'max_width':max(x['width'] for x in images),'max_height':max(x['height'] for x in images),
 'largest_by_pixels':{k:largest[k] for k in ['file_name','width','height']},
 'largest_pixels':largest['width']*largest['height'],
 'FBCNN_configuration':{'in_nc':3,'out_nc':3,'nc':[64,128,256,512],'nb':4,'downsample':'strideconv','upsample':'convtranspose'},
 'FBCNN_source':'https://raw.githubusercontent.com/jiaxi-jiang/FBCNN/main/models/network_fbcnn.py',
 'FBCNN_source_pin':'Main-branch text read at research date; not a fully pinned repository. Must pin source before future use.',
 'FBCNN_parameter_count_by_symbolic_arithmetic':params,'parameter_parts':parts,
 'FBCNN_FP32_weights_MiB':params*4/(2**20),
 'largest_train_encoder_skip_arrays_FP32_MiB':skip_floats*4/(2**20),
 'memory_limits':['Skip arrays plus weights are only partial storage, not a peak-memory bound.','No convolution workspace, allocator, current tensors, library overhead or detector memory measured.','No feasibility guarantee based on these numbers; sequential batch1 proposed for future pilot.'],
 'pilot_future_counts':{'sources':100,'regions':4,'subsets_per_source':16,'detectors':2,'restorer_forwards':100,'detector_forwards_subsets':3200,'optional_source_reference_detector_forwards':200,'new_utility_model_fits':0},
 'execution':{'metadata_JSON_read':True,'dataset_images_read':0,'model_imports':0,'model_constructed':0,'model_forwards':0,'training':0,'profiling':0,'GPU_jobs':0}}
(root/'static_resource_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:receipt[k] for k in ['images','largest_by_pixels','FBCNN_parameter_count_by_symbolic_arithmetic','FBCNN_FP32_weights_MiB','largest_train_encoder_skip_arrays_FP32_MiB','pilot_future_counts']},ensure_ascii=False))
