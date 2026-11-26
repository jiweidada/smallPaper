"""Read only the authorized new UAVDT data directory and its legacy image roots."""
import datetime
import json
import os
from pathlib import Path
import shutil

root=Path('/24085403014/datasets_verified_20261017/UAVDT_official')
record={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'root':str(root),'free_bytes':shutil.disk_usage(root).free,
        'new_assets':{name:{'exists':(root/name).exists(),
                            'bytes':(root/name).stat().st_size if (root/name).is_file() else None}
                      for name in ['raw/UAV-benchmark-M.zip','raw/UAV-benchmark-M.zip.part',
                                   'raw/M_attr.zip','raw/UAVDT_complete_image_manifest_v1.json',
                                   'native_frames_v2','native_attributes_v2','complete_image_replica_v2.json']},
        'legacy_image_counts':{},'new_training_started':False}
for split in ['train','test']:
    path=Path('/24085403014/DEIM/UAVDT_Datasets')/('UAVDT-'+split)/'images'
    if path.is_dir():
        with os.scandir(path) as items:record['legacy_image_counts'][split]=sum(p.is_file() for p in items)
    else:record['legacy_image_counts'][split]=None
print(json.dumps(record))
