from pathlib import Path
import hashlib,json,sys
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parent
def main():
    p=R/'author_report_archive.json';d=json.loads(p.read_text(encoding='utf-8'))
    scope='SelectedSec3.2.1,Sec3.3.2-3.3.3,selected4.1/4.3.3-4.3.4/4.4 andTables5-7. Eq2-5 visually checkedPDFpage7;Tables5/6 andheadtradeoff visually checkedPDFpage13. Not full31-page paper,supplementary,MuSGD experiments or COCO reproduction.'
    d.update(read_scope=scope,version_scope_correction=str(R/'version_scope_correction.json'),read_by='assistant',human_read_attested=False)
    assert hashlib.sha256(Path(d['file']).read_bytes()).hexdigest()==d['sha256']
    assert hashlib.sha256(Path(d['text_file']).read_bytes()).hexdigest()==d['text_sha256']
    for out in [p,Path(d['file']).with_suffix('.source.json')]:out.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    record={'id':'Ultralytics_YOLO26_Author_A30','title':d['title'],'authors':d['authors'],'url':d['url'],'tier':d['version'],'read_scope':scope,'author_method':'Dual-head real-time detector,DFL-free regression,epochwiseProgressiveLoss,STAL andMuSGD;published dual-head benchmark andSTALreference-size ablations.','our_consequence':'Nativeone2one/one2many tradeoffs and generic tiny-positive coverage are already published. Pinnedsource/runtime determine this diagnostic;currentdocs andpapertrigger differ from frozenimplementation.','unresolved':'No our novel architecture or mechanism follows from dual-head comparison;our D0recipe is not authorCOCOrecipe.','read_by':'assistant','human_read_attested':False,'local_pdf':d['file'],'pdf_sha256':d['sha256'],'access_record':str(Path(d['file']).with_suffix('.source.json'))}
    with (R/'new_core_records.json').open('x',encoding='utf-8') as f:json.dump({'rows':[record]},f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'new_core_records':1,'reading_scope_saved':True,'official_author_report_not_third_party_overview':True}))
if __name__=='__main__':main()
