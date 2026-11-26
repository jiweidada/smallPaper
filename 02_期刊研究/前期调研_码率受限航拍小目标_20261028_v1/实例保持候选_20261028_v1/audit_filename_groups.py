"""Existing train-annotation filename metadata only; no image/model/simulation."""
from pathlib import Path
from collections import Counter
import hashlib, json
root=Path(__file__).resolve().parent
source=Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文\相位机制论文_20261018\修订_20261018_v2\data\annotations\train\converted.json')
raw=source.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='32c1cae852e084430ea74503eb660f0707ec6b5e65d0998ce3ec8b448265c883'
images=json.loads(raw)['images']
counts=Counter(Path(x['file_name']).name.split('_')[0] for x in images)
result={'date':'2026-10-28','source':str(source),'source_sha256':hashlib.sha256(raw).hexdigest(),
        'images':len(images),'unique_filename_first_tokens':len(counts),
        'largest_token_counts':counts.most_common(20),
        'counts_histogram':dict(sorted(Counter(counts.values()).items())),
        'limitation':'Filename first token is a provisional grouping key only. Not verified to equal independent scenes/flights; image independence is not established by unique tokens.',
        'execution':{'dataset_image_reads':0,'image_encodes':0,'augmentations':0,'random_simulations':0,'model_imports':0,'model_forwards':0,'training':0,'test_dev_reads':0}}
(root/'filename_group_metadata.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=True))
