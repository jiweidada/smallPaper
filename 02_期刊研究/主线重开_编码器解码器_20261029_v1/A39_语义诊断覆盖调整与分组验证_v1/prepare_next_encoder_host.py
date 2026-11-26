from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,urllib.request
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;B=D.parent;P=B/'A32_轻量新编码器与专家路由原文筛查_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=read(P/'selected_source_manifest.json')['rows']
selected=[r for r in rows if r['repo']=='Intellindust-AI-Lab/EdgeCrafter']
assert len(selected)==5
for r in selected:assert sha(r['local'])==r['sha256']
readme=next(r for r in selected if r['path']=='README.md')
text=Path(readme['local']).read_text(encoding='utf-8')
asset_url='https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_s.pth'
assert asset_url in text
release_url='https://api.github.com/repos/capsule2077/edgecrafter/releases/tags/edgecrafterv1'
asset=dict(status='METADATA_NOT_YET_VERIFIED',url=asset_url)
try:
    request=urllib.request.Request(release_url,headers={'User-Agent':'UAV-architecture-primary-source-review'})
    with urllib.request.urlopen(request,timeout=30) as response:raw=response.read()
    release=json.loads(raw)
    with (D/'ECDet_release_metadata.json').open('xb') as output:output.write(raw)
    a=next(a for a in release['assets'] if a['name']=='ecdet_s.pth')
    assert a['browser_download_url']==asset_url
    asset.update(status='PRIMARY_RELEASE_ASSET_METADATA_VERIFIED_NOT_DOWNLOADED',release_id=release['id'],asset_id=a['id'],
        bytes=a['size'],api_digest=a.get('digest'),updated_at=a['updated_at'],metadata_sha256=hashlib.sha256(raw).hexdigest())
except Exception as error:asset.update(status='RELEASE_METADATA_UNAVAILABLE_NO_WEIGHT_DOWNLOADED',error=repr(error))
plan=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),status='NEXT_ENCODER_HOST_ASSET_AND_CAPACITY_DRAFT_NO_NEW_MODEL_OR_TRAINING',
    reason='A39pairedscreenfailedtheFine-over-QueryadvancementgatesinCNN-basedDome;no fine-readoutmoduleorrescuegridisjustified. ItdoesnotmeasurecompactViTencoding.',
    host='OfficialECDet-S/EdgeCrafter,asnewcompactViTdiagnostichost,notaclaimedalgorithmcontribution',
    repo='Intellindust-AI-Lab/EdgeCrafter',source_commit='d8752b75c9beaecfcdd0e5cd207bae3b6946ae23',
    pinned_selected_source_rehashed=5,config='ecdetseg/configs/ecdet/ecdet_s.yml',
    known_source='Overlappingstride16ConvStem,meanlateViTblocks10/11,resized8/16/32pyramid,and4layer300querydecoder;SprojectionandFPNdetailsretainedinA32audit.',
    primary_readme=str(readme['local']),primary_readme_sha256=readme['sha256'],asset=asset,
    variant='COCO-onlydetectionweightswithoutObjects365extradetectionpretraining;thisdoesnotmeanabsenceofexternalteacher/backbonepretraining.',
    first_required_steps=['DownloadlocallyandrecordactualSHA256beforeremoteupload',
        'Inspectcheckpointwithsafeweights-onlyloadingandstrictoriginalCOCOmodelidentity',
        'Tracecompleteauthorentry/config/dependenciesandactualselectedoperators',
        'SpecifyVisDrone10classmapping,ignore0/other11handlingandheadinitialization;COCO80classoutputsarenotaVisDronebaseline',
        'Freezeisolatedsingle24GBforward/backwardcapacitypreflightwithboundedcalls/updatesbeforeanylongrun',
        'Establishtask-adaptedbaselinebeforeclaimingsemantic/spatialfailureinspecificViTstages'],
    DeepSeek_scope='Usecompletedprimary-mechanism/operatorauditonlyasadesignreferenceafteraspecificbaselinefailureisobserved;neithername-substitutionnorsharedstatealoneisanincrement.',
    closest_guidance_warning='TSC-DETRtitle/DOIisadiscoverylead;fullprimarymethodandrelatedrepositoryidentitynotclosed. GenericTransformer-to-CNNsemanticguidancecannotbeassumednovel.',
    no_new_model_call_allocation=0,no_new_optimizer_update_allocation=0,weights_downloaded=False,
    training_on_24GB_verified=False,new_algorithm_approved=None,mainline_validated=False,
    old_A22_DEIM_or_compression_restarted=False)
with (D/'next_compact_ViT_host_DRAFT.json').open('x',encoding='utf-8') as output:json.dump(plan,output,ensure_ascii=False,indent=2)
print(json.dumps(dict(status=plan['status'],source_files_reverified=5,asset_metadata_status=asset['status'],asset_bytes=asset.get('bytes'),
    new_model_calls=0,new_optimizer_updates=0),ensure_ascii=False))
