"""Freeze one geometry-decomposition contrast before reading the derived outcomes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import py_compile

BASE=Path(__file__).resolve().parent
ROOT=BASE/'A2_已有预测框误差分解_v1'
assert not (ROOT/'freeze_manifest.json').exists()
py_compile.compile(str(ROOT/'decompose_boxes.py'),doraise=True)
protocol=dict(material_passport=dict(origin_skill='academic-research-suite',origin_mode='existing-prediction conditional diagnostic',
    origin_date=datetime.now(timezone.utc).isoformat(),verification_status='FROZEN_NOT_EXECUTED',version='A2_geometry_v1'),
    status='FROZEN_BEFORE_DERIVED_GEOMETRY_RESULTS',scope='All100 existing A0 trace files,4491 GT;zero new forwards or fits',
    research_question='Within persistently missed tiny GT with a nearby native selected query,is initial candidate geometry more limited by center or width/height?',
    primary_cohort='sqrt input area<16;nativeTop300 maxIoU<0.5 and finaldecoder maxIoU<0.5;at least one native selected stride8 anchor in GT-nearest3x3 grid neighborhood',
    candidates='Same native selected query IDs for every counterfactual and stage;GT-local anchor association is diagnostic,not object identity',
    oracles=dict(center='Set center toGT,keep width/height;optionally keep unmodified box',
                 size='Set width/height toGT,keep center;optionally keep unmodified box',
                 template='Oracle choose among15 fixed width/height templates: sqrt-area4,8,16 times aspect ratios0.25,0.5,1,2,4;native center unchanged;optionally keep original'),
    stages=['encoder_topk300','decoder_0','decoder_3'],modes=['geometry','class_argmax'],
    primary_contrast='Size-oracle recovered fraction minus center-oracle recovered fraction atIoU0.5,encoder_topk300,geometry mode;average within source image then across images',
    interval='2000 filename-prefix group cluster bootstraps,seed20261030,percentile95%',
    necessary_screen_conditions=['at least100 cohort GT and15 groups','mean size-minus-center >=0.10 and95% lower>0',
       'size-minus-template-bank mean>=0.05','class-aware size-minus-center>0','both <8 and8-16 size-minus-center point contrasts>0'],
    decision='If not supported,do not treat tiny-box size-specific architecture as a mainline from this evidence. No threshold,template,subset or stage substitution after reading results.',
    limitations=['GT oracles are not deployable algorithms orAP gains.','Template oracle expands candidates and is only an explanatory control.',
                 'Geometry ignores confidence and may share one query among GT.','Author checkpoint saw training images.','Initial-and-final-failure conditioning is exploratory.'],
    budgets=dict(new_model_forwards=0,new_fits=0,new_main_model_updates=0,analysis_hard_timeout_seconds=180,automatic_retry=False))
protocol_file=ROOT/'protocol.json'
protocol_file.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
manifest=dict(script_sha256=hashlib.sha256((ROOT/'decompose_boxes.py').read_bytes()).hexdigest(),
              protocol_sha256=hashlib.sha256(protocol_file.read_bytes()).hexdigest(),
              frozen_utc=datetime.now(timezone.utc).isoformat())
(ROOT/'freeze_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='A2_GEOMETRY_ANALYSIS_FROZEN',**manifest)))
