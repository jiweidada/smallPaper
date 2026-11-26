from pathlib import Path
import ast,json
D=Path(__file__).resolve().parent;OLD=D.parent/'A38_细尺度与查询语义成对诊断_v1'
for name in ['run_and_record.py','launch_collector.py']:
    text=(OLD/name).read_text(encoding='utf-8')
    if name=='run_and_record.py':text=text.replace('==114','==113')
    ast.parse(text,feature_version=(3,9))
    with (D/name).open('x',encoding='utf-8') as f:f.write(text)
print(json.dumps(dict(status='BOUNDED_RUNTIME_WRAPPERS_READY',cap=113,worker_timeout=900)))
