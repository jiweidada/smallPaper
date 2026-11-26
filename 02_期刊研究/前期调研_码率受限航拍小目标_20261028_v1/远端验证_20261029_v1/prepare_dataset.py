"""Archive the explicitly selected public VisDrone train split. No val/test, no private papers."""
from pathlib import Path
from collections import Counter
import hashlib,json,tarfile,time
ROOT=Path(__file__).resolve().parent
SOURCE=Path(r"D:\数据集\visdrone2019\VisDrone2019-DET-train")
ANNOT=ROOT.parents[2]/"英文小论文/相位机制论文_20261018/修订_20261018_v2/data/annotations/train/converted.json"
data=json.loads(ANNOT.read_bytes())
names=sorted(im["file_name"] for im in data["images"])
assert len(names)==len(set(names))==6471
assert set(names)=={p.name for p in (SOURCE/"images").glob("*.jpg")}
archive=ROOT/"visdrone_train_source.tar"
rows=[]
start=time.monotonic()
with tarfile.open(archive,"w") as tar:
    for i,name in enumerate(names):
        item={"file_name":name}
        for folder,fn in (("images",name),("annotations",Path(name).stem+".txt")):
            p=SOURCE/folder/fn
            assert p.is_file() and not p.is_symlink()
            raw=p.read_bytes()
            item[folder]={"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
            tar.add(p,arcname="dataset/train/"+folder+"/"+fn,recursive=False)
        rows.append(item)
        if (i+1)%1000==0:print(json.dumps({"archived_images":i+1,"seconds":round(time.monotonic()-start,2)}),flush=True)
with archive.open("rb") as stream:digest=hashlib.file_digest(stream,"sha256").hexdigest()
record={"source":str(SOURCE),"scope":"official train only; raw images and raw txt annotations; no val/test","images":6471,
        "files":rows,"archive":str(archive),"archive_bytes":archive.stat().st_size,"archive_sha256":digest,
        "source_metadata_sha256":hashlib.sha256(ANNOT.read_bytes()).hexdigest(),"seconds":round(time.monotonic()-start,2)}
(ROOT/"dataset_source_manifest.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:record[k] for k in ('images','archive_bytes','archive_sha256','seconds')}),flush=True)
