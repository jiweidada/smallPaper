"""Build exact native GT/ignore metadata without changing images or official test split."""
import collections
import datetime
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]/'data/07_第二数据集_20261017'
OUTPUT=ROOT/'执行_v2/data_packet'
TEST_SEQUENCES=['M0203','M0205','M0208','M0209','M0403','M0601','M0602','M0606','M0701','M0801','M0802','M1001','M1004','M1007','M1009','M1101','M1301','M1302','M1303','M1401']

def dump(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,separators=(',',':'))

def main():
    OUTPUT.mkdir(parents=True,exist_ok=False);(OUTPUT/'annotations').mkdir();(OUTPUT/'protocol_reference').mkdir()
    plan=json.loads((ROOT/'UAVDT_96轮待执行队列_20261017_v2.json').read_text(encoding='utf-8'))
    assert plan['jobs_total']==6
    manifest_path=ROOT/'协议核查_v1/UAVDT完整图像文件清单_v1.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    frame_entries=sorted(manifest,key=lambda x:x['relative']);split_for={s:k for k in ['train','val','test'] for s in plan['split'][k+'_sequences']}
    grouped=collections.defaultdict(list)
    for index,entry in enumerate(frame_entries,1):
        sequence=entry['relative'].split('/')[0];frame=int(Path(entry['relative']).stem[3:])
        grouped[sequence].append((index,frame,entry))
    prefix='UAV-benchmark-MOTD_v1.0/'
    archive_path=ROOT/'原始下载/UAVDT/UAV-benchmark-MOTD_v1.0.zip'
    source_hash=hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert source_hash=='1e565da8c2a035bf1e56b1322de76c74b73702ea3b0f918f45debf69c2b26799'
    summary={};ann_id=0
    with zipfile.ZipFile(archive_path) as source:
        for split in ['train','val','test']:
            images=[];annotations=[];overlapping=0;fully_inside=0;ignore_count=0;empty=0
            for sequence in plan['split'][split+'_sequences']:
                positive_raw=source.read(prefix+'GT/'+sequence+'_gt_whole.txt')
                gt=np.loadtxt(io.BytesIO(positive_raw),delimiter=',',ndmin=2)
                raw=source.read(prefix+'GT/'+sequence+'_gt_ignore.txt')
                ignored=np.loadtxt(io.BytesIO(raw),delimiter=',',ndmin=2) if raw.strip() else np.empty((0,9))
                gt_by=collections.defaultdict(list);ign_by=collections.defaultdict(list)
                for row in gt:gt_by[int(row[0])].append(row)
                for row in ignored:ign_by[int(row[0])].append(row[2:6].tolist())
                for image_id,frame,entry in grouped[sequence]:
                    positives=gt_by[frame];regions=ign_by[frame];ignore_count+=len(regions);empty+=int(not positives)
                    images.append({'id':image_id,'file_name':entry['relative'],'width':entry['width'],'height':entry['height'],
                                   'uav_sequence':sequence,'uav_frame':frame,'uav_ignore_boxes':regions,
                                   'uav_apply_native_ignore':sequence not in ['M0209','M0801'] if split=='test' else True})
                    for row in positives:
                        box=row[2:6].tolist();assert box[2]>0 and box[3]>0
                        ann_id+=1
                        annotations.append({'id':ann_id,'image_id':image_id,'category_id':0,'bbox':box,
                                            'area':box[2]*box[3],'iscrowd':0,'uav_original_class':int(row[8]),'uav_track_id':int(row[1])})
                        x,y,w,h=box
                        overlapping+=int(any(min(x+w,a+c)>max(x,a) and min(y+h,b+d)>max(y,b) for a,b,c,d in regions))
                        fully_inside+=int(any(x>a and y>b and x+w<a+c and y+h<b+d for a,b,c,d in regions))
            annotation={'info':{'dataset':'UAVDT_v1','split':split,'protocol':'native overall vehicle AP70',
                                'split_tag':plan['split']['tag'],'source_toolkit_sha256':source_hash},
                        'uavdt_split':split,'images':images,'annotations':annotations,'categories':[{'id':0,'name':'vehicle'}]}
            assert len(images)==plan['split'][split+'_images']
            dump(OUTPUT/'annotations'/(split+'.json'),annotation)
            summary[split]={'images':len(images),'positive_GT':len(annotations),'ignore_boxes':ignore_count,
                            'empty_positive_frames':empty,'positive_boxes_intersecting_ignore':overlapping,
                            'positive_boxes_strictly_inside_ignore':fully_inside}
        reference=OUTPUT/'protocol_reference'
        for sequence in TEST_SEQUENCES:
            path=reference/(sequence+'.txt');path.write_bytes(source.read(prefix+'RES_DET/det_FRCNN/'+sequence+'.txt'))
        (reference/'det_FRCNN_overall.mat').write_bytes(source.read(prefix+'det_EVA/det_FRCNN_overall.mat'))
        for name in ['utils/CalculateDetectionPR_overall.m','utils/evalRes.m','compOas.m','utils/VOCap.m']:
            (reference/Path(name).name).write_bytes(source.read(prefix+name))
    (OUTPUT/'native_image_manifest.json').write_bytes(manifest_path.read_bytes())
    metadata={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dataset':'UAVDT_v1',
              'split':plan['split'],'counts':summary,'training_ignore_treatment':'Mean-fill ignored rectangles before augmentation, preserving every positive GT rectangle pixel; same for all arms. Validation/test images unchanged.',
              'image_archive_sha256':'e82675c938e4e6bd65b4533dba2fb6a028e3e60bd3577cc2f3031504bd585b08',
              'original_image_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'metadata_files':[{'relative':str(p.relative_to(OUTPUT)).replace('\\','/'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(OUTPUT.rglob('*')) if p.is_file()]}
    dump(OUTPUT/'data_metadata.json',metadata)
    packet=ROOT/'执行_v2/UAVDT_metadata_v2.tar.gz'
    with tarfile.open(packet,'x:gz') as out:
        for path in sorted(OUTPUT.rglob('*')):
            if path.is_file():out.add(path,arcname=str(path.relative_to(OUTPUT)).replace('\\','/'),recursive=False)
    receipt={'packet':str(packet.resolve()),'bytes':packet.stat().st_size,'sha256':hashlib.sha256(packet.read_bytes()).hexdigest(),'counts':summary}
    dump(ROOT/'执行_v2/UAVDT_metadata_v2.receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
