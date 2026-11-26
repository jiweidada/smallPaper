"""Run only after full download: all JPEG headers/hashes, official split, GT coverage and size statistics."""
import collections
import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent;DATA=HERE.parents[1]/'data/07_第二数据集_20261017'
RAW=DATA/'原始下载/UAVDT';PREFIX='UAV-benchmark-MOTD_v1.0/'

def main():
    image_zip=RAW/'UAV-benchmark-M.zip';receipt=json.loads((RAW/'UAV-benchmark-M.zip.receipt.json').read_text(encoding='utf-8'))
    assert receipt['zip_crc_verified'] is True and receipt['sha256']=='e82675c938e4e6bd65b4533dba2fb6a028e3e60bd3577cc2f3031504bd585b08'
    assert image_zip.stat().st_size==receipt['bytes']==6790452113
    source_hashes={}
    for name,expected in [('M_attr.zip','ac6f85e355db3f4808cd64148b1f3c33933906ebd47c63939bb7eba6974824d1'),
                          ('UAV-benchmark-MOTD_v1.0.zip','1e565da8c2a035bf1e56b1322de76c74b73702ea3b0f918f45debf69c2b26799')]:
        digest=hashlib.sha256()
        with (RAW/name).open('rb') as source:
            for block in iter(lambda:source.read(8*1024**2),b''):digest.update(block)
        assert digest.hexdigest()==expected
        source_hashes[name]=expected
    destination=DATA/'协议核查_v1/UAVDT完整原始包验收_v1.json'
    manifest_path=DATA/'协议核查_v1/UAVDT完整图像文件清单_v1.json'
    assert not destination.exists() and not manifest_path.exists(),'Do not overwrite a previous complete audit'
    with zipfile.ZipFile(RAW/'M_attr.zip') as attributes:
        splits={split:sorted({re.search(r'(M\d{4})\s*_attr\.txt$',n)[1] for n in attributes.namelist()
            if '/'+split+'/' in n and re.search(r'(M\d{4})\s*_attr\.txt$',n)}) for split in ['train','test']}
        attribute_name_notes=[{'original_member':n,'sequence':re.search(r'(M\d{4})\s*_attr\.txt$',n)[1],
                              'handling':'Whitespace accepted in logical sequence mapping; original archive/member unchanged'}
            for n in attributes.namelist() if re.search(r'M\d{4}\s+_attr\.txt$',n)]
    assert len(splits['train'])==30 and len(splits['test'])==20 and not set(splits['train'])&set(splits['test'])
    frames=collections.defaultdict(dict);manifest=[];unexpected=[]
    with zipfile.ZipFile(image_zip) as archive:
        for item in archive.infolist():
            if item.is_dir():continue
            matched=re.search(r'(?:^|/)(M\d{4})/img(\d{6})\.jpg$',item.filename)
            if not matched:unexpected.append(item.filename);continue
            sequence,frame=matched[1],int(matched[2]);assert frame not in frames[sequence]
            raw=archive.read(item);image=Image.open(io.BytesIO(raw));width,height=image.size
            assert image.format=='JPEG' and width>0 and height>0
            record={'relative':sequence+'/img'+str(frame).zfill(6)+'.jpg','archive_member':item.filename,
                    'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'width':width,'height':height}
            frames[sequence][frame]=record;manifest.append(record)
    assert not unexpected,'Unexpected non-image files in original detection archive'
    assert set(frames)==set(splits['train']+splits['test'])
    counts={split:sum(len(frames[s]) for s in splits[split]) for split in splits}
    assert counts=={'train':24143,'test':16592},'Released image coverage differs from official protocol; inspect before use'
    per_sequence=[];size_values={split:[] for split in splits};class_counts={split:collections.Counter() for split in splits}
    with zipfile.ZipFile(RAW/'UAV-benchmark-MOTD_v1.0.zip') as gt:
        for split,sequences in splits.items():
            for sequence in sequences:
                native=gt.read(PREFIX+'GT/'+sequence+'_gt_whole.txt')
                positive=np.loadtxt(io.BytesIO(native),delimiter=',',ndmin=2)
                ignore_raw=gt.read(PREFIX+'GT/'+sequence+'_gt_ignore.txt')
                ignored=np.loadtxt(io.BytesIO(ignore_raw),delimiter=',',ndmin=2) if ignore_raw.strip() else np.empty((0,9))
                present=set(frames[sequence]);missing=sorted(set(range(1,max(present)+1))-present)
                assert not missing,'Internal frame gaps in '+sequence
                assert set(positive[:,0].astype(int))<=present and set(ignored[:,0].astype(int))<=present
                for row in positive:
                    image=frames[sequence][int(row[0])];area=float(row[4]*row[5]);assert area>0
                    size_values[split].append((area**.5,(area*640/image['width']*640/image['height'])**.5))
                    class_counts[split][int(row[8])]+=1
                per_sequence.append({'split':split,'sequence':sequence,'images':len(present),'positive_boxes':len(positive),
                    'ignore_boxes':len(ignored),'empty_positive_frames':len(present-set(positive[:,0].astype(int))),
                    'GT_sha256':hashlib.sha256(native).hexdigest(),'ignore_sha256':hashlib.sha256(ignore_raw).hexdigest()})
    size_summary={}
    for split,values in size_values.items():
        array=np.asarray(values)
        size_summary[split]={'GT_count':len(array),'original_sqrt_area_quantiles_10_25_50_75_90':np.percentile(array[:,0],[10,25,50,75,90]).tolist(),
            'input640_stretch_sqrt_area_quantiles_10_25_50_75_90':np.percentile(array[:,1],[10,25,50,75,90]).tolist(),
            'input640_bins':{label:int(((array[:,1]>=lo)&(array[:,1]<hi)).sum()) for label,lo,hi in
                [('<4',0,4),('4-8',4,8),('8-16',8,16),('16-32',16,32),('>=32',32,float('inf'))]}}
    with manifest_path.open('x',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
    record={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'complete_original_detection_archive_verified',
            'scope':'Complete UAVDT-Benchmark-M DET/MOT: 50 sequences; separate SOT package is not part of the detection experiment.',
            'archive':str(image_zip.resolve()),'archive_sha256':receipt['sha256'],'archive_bytes':receipt['bytes'],
            'full_zip_crc_verified':True,'all_images_sha_and_JPEG_headers_verified':True,'image_counts':counts,
            'splits':splits,'attribute_filename_notes':attribute_name_notes,
            'source_asset_sha256':source_hashes,'all_sequence_frame_ranges_contiguous_from_1':True,
            'all_positive_and_ignore_GT_frames_present':True,
            'image_dimensions_and_counts':dict(collections.Counter(str(item['width'])+'x'+str(item['height']) for item in manifest)),
            'per_sequence':per_sequence,'positive_class_counts':{s:dict(c) for s,c in class_counts.items()},
            'size_summary':size_summary,'manifest':str(manifest_path.resolve()),
            'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'selected_dataset':None,'new_training_started':False}
    with destination.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
    print(json.dumps({'audit':str(destination),'counts':counts,'size_summary':size_summary},ensure_ascii=False))

if __name__=='__main__':main()
