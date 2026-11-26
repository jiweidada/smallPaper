"""Add verified native frames to B; preserve old converted datasets and all training artifacts."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT=Path('/24085403014/datasets_verified_20261017/UAVDT_official')
archive=ROOT/'raw/UAV-benchmark-M.zip';attributes=ROOT/'raw/M_attr.zip';manifest_path=ROOT/'raw/UAVDT_complete_image_manifest_v1.json'
dest=ROOT/'native_frames_v2';attr_dest=ROOT/'native_attributes_v2';receipt=ROOT/'complete_image_replica_v2.json'
assert archive.is_file() and attributes.is_file() and manifest_path.is_file()
assert not dest.exists() and not attr_dest.exists() and not receipt.exists(),'Do not redeploy or overwrite a previous extraction'
entries=json.loads(manifest_path.read_text());assert len(entries)==40735
assert hashlib.sha256(attributes.read_bytes()).hexdigest()=='ac6f85e355db3f4808cd64148b1f3c33933906ebd47c63939bb7eba6974824d1'
expected='e82675c938e4e6bd65b4533dba2fb6a028e3e60bd3577cc2f3031504bd585b08'
h=hashlib.sha256()
with archive.open('rb') as f:
    for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
assert h.hexdigest()==expected
assert shutil.disk_usage(ROOT).free>sum(e['bytes'] for e in entries)*2
dest.mkdir();attr_dest.mkdir()
with zipfile.ZipFile(archive) as source:
    for item in entries:
        path=dest/item['relative'];assert path.is_relative_to(dest) and '..' not in Path(item['relative']).parts
        raw=source.read(item['archive_member']);assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(raw)
with zipfile.ZipFile(attributes) as source:
    for item in source.infolist():
        path=attr_dest/item.filename;assert path.is_relative_to(attr_dest) and '..' not in Path(item.filename).parts
        if item.is_dir():path.mkdir(parents=True,exist_ok=True)
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:f.write(source.read(item))
legacy_comparison={}
with zipfile.ZipFile(attributes) as source:
    for split in ['train','test']:
        sequences={re.search(r'(M\d{4})\s*_attr\.txt$',name)[1] for name in source.namelist()
                   if '/'+split+'/' in name and re.search(r'(M\d{4})\s*_attr\.txt$',name)}
        expected_frames={(item['relative'].split('/')[0],int(Path(item['relative']).stem[3:]))
                         for item in entries if item['relative'].split('/')[0] in sequences}
        legacy_root=Path('/24085403014/DEIM/UAVDT_Datasets')/('UAVDT-'+split)/'images'
        if not legacy_root.is_dir():
            legacy_comparison[split]={'state':'legacy_images_unavailable'};continue
        old_frames=set();unrecognized=[]
        for path in legacy_root.iterdir():
            if not path.is_file():continue
            matched=re.fullmatch(r'(M\d{4})_(\d+)\.(?:jpg|jpeg|png)',path.name,re.IGNORECASE)
            if matched:old_frames.add((matched[1],int(matched[2])))
            else:unrecognized.append(path.name)
        missing=sorted(expected_frames-old_frames);positive_frames=set();ignore_frames=set()
        for sequence in sequences:
            for suffix,accumulator in [('_gt_whole.txt',positive_frames),('_gt_ignore.txt',ignore_frames)]:
                gt_path=ROOT/'author_toolkit/UAV-benchmark-MOTD_v1.0/GT'/(sequence+suffix)
                assert gt_path.is_file()
                for row in gt_path.read_text().splitlines():
                    if row.strip():accumulator.add((sequence,int(row.split(',')[0])))
        legacy_comparison[split]={'state':'compared_to_complete_native_frame_set','legacy_images':len(old_frames),
            'native_images':len(expected_frames),'missing_native_frames':len(missing),
            'missing_with_positive_GT':sum(frame in positive_frames for frame in missing),
            'missing_with_ignore_GT':sum(frame in ignore_frames for frame in missing),
            'missing_frames':[{'sequence':s,'frame':f} for s,f in missing],
            'unexpected_native_frame_keys':[{'sequence':s,'frame':f} for s,f in sorted(old_frames-expected_frames)],
            'unrecognized_files':unrecognized}
record={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'complete_native_images_and_attributes_verified',
        'images':len(entries),'archive_sha256':expected,'file_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'native_frames':str(dest),'native_attributes':str(attr_dest),'old_converted_data_preserved':True,'new_training_started':False,
        'dataset_selection_pending':True,'legacy_converted_coverage_comparison':legacy_comparison}
with receipt.open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
