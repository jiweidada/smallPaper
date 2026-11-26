"""Prepare and verify only archived bytes. No model imports or GPU work."""
from pathlib import Path, PurePosixPath
import datetime
import hashlib
import json
import os
import shutil
import stat
import tarfile
import time
import zipfile

VOLUME=Path('/25085400120') if Path('/25085400120').is_mount() else Path('/24085403014')
WORK=VOLUME/'mpsma_fresh_pair_20261112_v1'
WORK.mkdir(exist_ok=True)
PREP=WORK/'preparation'
PREP.mkdir(exist_ok=True)
EXPECTED_PROTOCOL='8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'
EXPECTED_MANIFEST='ba463a04fe688d5c3005cb3cf9ca6e13aa699b0b11016e15e7fa9b76b52e2a56'
EXPECTED_ZIP='fdde141711867fa1bc2de3a1e6342cca257eb056e2a1700795eb04068aa90266'
EXPECTED_TAR='e127bf14dacd999ca3794c90dc16d778f34f22ff135c227d599ad43ac243c36d'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while True:
            b=f.read(4*1024**2)
            if not b:break
            h.update(b)
    return h.hexdigest()
def emit(status,**details):
    d={'status':status,'observed_utc':now(),'training_started':False,**details}
    tmp=PREP/'file_preparation_status.json.partial'
    tmp.write_text(json.dumps(d,indent=2)+'\n')
    os.replace(str(tmp),str(PREP/'file_preparation_status.json'))
def inside(path,root):return path==root or root in path.parents
def verify(path,item):
    assert path.is_file() and not path.is_symlink(),str(path)
    if 'bytes' in item:assert path.stat().st_size==item['bytes'],str(path)
    assert sha(path)==item['sha256'],str(path)

try:
    emit('VERIFYING_PERSISTENT_STORAGE')
    assert VOLUME.is_mount() and shutil.disk_usage(str(VOLUME)).free>=70*2**30
    if VOLUME.name=='25085400120':
        frozen=VOLUME/'mpsma_fair_20261103_v1'
        dataset=Path('/tmp/mpsma_module_ablation_20261107_data')
    else:
        inputs=WORK/'inputs'
        archive=inputs/'secondary_frozen_inputs.tar.gz'
        emit('VERIFYING_FROZEN_SOURCE_ARCHIVE')
        assert sha(archive)==EXPECTED_TAR
        frozen=WORK/'frozen_inputs'
        assert not frozen.exists(),'Refusing to replace existing frozen source extraction'
        frozen.mkdir()
        with tarfile.open(str(archive),'r:gz') as tar:
            members=tar.getmembers()
            for member in members:
                rel=PurePosixPath(member.name)
                assert not rel.is_absolute() and '..' not in rel.parts and (member.isdir() or member.isfile())
                assert inside((frozen/member.name).resolve(),frozen.resolve())
            tar.extractall(str(frozen),members=members)
        dataset=Path('/tmp/mpsma_fresh_pair_20261112_data')
        zip_path=inputs/'verified_dataset.zip'
        manifest_path=inputs/'dataset_bundle_manifest.json'
        emit('VERIFYING_CANONICAL_DATA_ARCHIVE')
        assert sha(manifest_path)==EXPECTED_MANIFEST and sha(zip_path)==EXPECTED_ZIP
        manifest=json.loads(manifest_path.read_text())
        assert manifest['zip_sha256']==EXPECTED_ZIP and zip_path.stat().st_size==manifest['bytes']
        assert len(manifest['files'])==7021
        expected={x['path']:x for x in manifest['files']}
        assert not dataset.exists(),'Refusing to replace an existing data cache'
        dataset.mkdir()
        with zipfile.ZipFile(str(zip_path)) as z:
            entries=z.infolist()
            assert len(entries)==len(expected) and {i.filename for i in entries}==set(expected)
            for index,entry in enumerate(entries):
                rel=PurePosixPath(entry.filename)
                assert not rel.is_absolute() and '..' not in rel.parts and not stat.S_ISLNK(entry.external_attr>>16)
                assert not entry.is_dir() and entry.file_size==expected[entry.filename]['bytes']
                target=(dataset/entry.filename).resolve()
                assert inside(target,dataset.resolve())
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(entry) as source,target.open('xb') as output:shutil.copyfileobj(source,output,4*1024**2)
                verify(target,expected[entry.filename])
                if index%500==0:emit('RESTORING_AND_VERIFYING7021_DATA_FILES',verified_files=index+1)
    emit('VERIFYING247_FROZEN_INPUT_FILES')
    protocol_path=frozen/'execution_protocol.json'
    assert sha(protocol_path)==EXPECTED_PROTOCOL
    protocol=json.loads(protocol_path.read_text())
    assert len(protocol['frozen_files'])==247
    for i,item in enumerate(protocol['frozen_files']):
        p=(frozen/item['path']).resolve()
        assert inside(p,frozen.resolve())
        verify(p,item)
        if i%50==0:emit('VERIFYING247_FROZEN_INPUT_FILES',verified_files=i+1)
    manifest_path=frozen/'dataset_bundle_manifest.json'
    assert sha(manifest_path)==EXPECTED_MANIFEST
    manifest=json.loads(manifest_path.read_text())
    assert len(manifest['files'])==7021
    emit('VERIFYING7021_CURRENT_DATA_FILES')
    for i,item in enumerate(manifest['files']):
        p=(dataset/item['path']).resolve()
        assert inside(p,dataset.resolve())
        verify(p,item)
        if i%500==0:emit('VERIFYING7021_CURRENT_DATA_FILES',verified_files=i+1)
    emit('ALL_FROZEN_INPUT_BYTES_VERIFIED',frozen_root=str(frozen),data_root=str(dataset),
         frozen_files=247,data_files=7021,protocol_sha256=EXPECTED_PROTOCOL,dataset_manifest_sha256=EXPECTED_MANIFEST,
         persistent_volume=str(VOLUME),persistent_free_bytes=shutil.disk_usage(str(VOLUME)).free,
         scope='Byte preparation and provenance only. No model import, forward, evaluator or training.')
    print((PREP/'file_preparation_status.json').read_text())
except Exception as error:
    emit('FILE_PREPARATION_FAILED_PRESERVED',error_type=type(error).__name__,error=str(error))
    raise
