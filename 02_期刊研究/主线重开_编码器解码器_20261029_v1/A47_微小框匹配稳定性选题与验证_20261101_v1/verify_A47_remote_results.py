"""Reaggregate saved assignment indices independently of the executed analysis code."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
download=ROOT/"remote_downloaded_30172"
complete=json.loads((download/"D0_results_v1/complete.json").read_text(encoding="utf-8"))
assert complete["status"]=="COMPLETE"
assert complete["counters"]=={"model_forwards":32,"model_backward_calls":0,"optimizer_updates":0}
assert complete["parameter_hash_before"]==complete["parameter_hash_after"]
assert complete["state_hash_before"]==complete["state_hash_after"]
checked=0
manifest=json.loads((download/"result_archive_manifest.json").read_text())
for name,row in manifest.items():
    p=download/name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row["sha256"] and p.stat().st_size==row["bytes"]
    checked+=1
for name,key in [("run_A47_frozen_match_capture_30172_v1.py","capture_script_sha256"),
                 ("run_A47_match_analysis.py","analysis_script_sha256"),("protocol_FROZEN.json","protocol_sha256"),
                 ("deployment_30172_amendment_FROZEN.json","deployment_amendment_sha256")]:
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==complete[key]
assert hashlib.sha256((download/"D0_results_v1/raw_final_outputs.npz").read_bytes()).hexdigest()==complete["raw_outputs_sha256"]
expected=json.loads((ROOT/"asset_expectations.json").read_text())
remote_assets="/root/uav_a47_20261101_v1/assets_A43_baseline"
for rel,key in [("arm_B/epoch12.pth","checkpoint_sha256"),("runtime_common.py","runtime_common_sha256"),
                ("common_init_10class.pth","common_init_sha256"),("train_data_manifest.json","train_data_manifest_sha256")]:
    assert complete["source_input_sha256"][remote_assets+"/"+rel]==expected[key]
for rel,h in expected["native_source_sha256"].items():
    path=remote_assets+"/source/EdgeCrafter-d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecdetseg/"+rel
    assert complete["source_input_sha256"][path]==h
local_dataset=Path("C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-train")
selected=json.loads((download/"D0_results_v1/selected_inputs_FROZEN.json").read_text())["rows"]
local_clone_checks=0
for meta in selected:
    for sub,file in [("images",meta["file"]),("annotations",Path(meta["file"]).stem+".txt")]:
        h=hashlib.sha256((local_dataset/sub/file).read_bytes()).hexdigest()
        assert h==complete["source_input_sha256"][complete["data_root"]+"/"+sub+"/"+file]
        local_clone_checks+=1
results={};permutations=0;scalar_owners=0;row_checks=0;max_iou_error=0
for seed in (1,):
    out=download/"D0_results_v1"
    data=np.load(out/"raw_final_outputs.npz",allow_pickle=False)
    selection=json.loads((out/"selected_inputs_FROZEN.json").read_text(encoding="utf-8"))["rows"]
    owners=json.loads((out/"assignment_owners.json").read_text(encoding="utf-8"))
    summary=json.loads((out/"analysis_summary.json").read_text(encoding="utf-8"))
    rows=list(csv.DictReader((out/"per_gt_results.csv").open(encoding="utf-8",newline="")))
    lookup={(r["image"],int(r["gt_index"]),r["method"],float(r["amplitude"])):r for r in rows}
    independent={}
    for image_index,meta in enumerate(selection):
        file=meta["file"];begin,end=data["offsets"][image_index:image_index+2]
        labels=data["target_labels"][begin:end];area=data["original_area"][begin:end]
        for method in ("B","M"):
            base=np.asarray(owners[file]["clean_"+method],dtype=np.int64)
            assert len(base)==len(labels) and len(np.unique(base))==len(base)
            if method=="B":assert np.array_equal(base,data["native_owners"][begin:end])
            pred=data["boxes"][image_index,base];gt=data["target_boxes"][begin:end]
            pa=np.c_[pred[:,:2]-pred[:,2:]/2,pred[:,:2]+pred[:,2:]/2]
            ga=np.c_[gt[:,:2]-gt[:,2:]/2,gt[:,:2]+gt[:,2:]/2]
            inter=np.maximum(0,np.minimum(pa[:,2:],ga[:,2:])-np.maximum(pa[:,:2],ga[:,:2])).prod(1)
            iou=inter/((pa[:,2:]-pa[:,:2]).prod(1)+(ga[:,2:]-ga[:,:2]).prod(1)-inter)
            for amp in (.5,1.):
                draw=np.asarray(owners[file]["draws"][method+"_"+str(amp)],dtype=np.int64)
                assert draw.shape==(16,len(labels))
                assert np.all((draw>=0)&(draw<300))
                assert all(len(np.unique(x))==len(labels) for x in draw)
                permutations+=16;scalar_owners+=int(draw.size)
                flip=np.count_nonzero(draw!=base,axis=0)/16
                qlab=np.full((16,300),-1,dtype=np.int64)
                qlab[np.arange(16)[:,None],draw]=labels[None,:]
                identity=np.count_nonzero(qlab[:,base]!=labels[None,:],axis=0)/16
                key=method+"_"+str(amp)
                for j in range(len(labels)):
                    r=lookup[(file,j,method,amp)]
                    assert float(r["owner_flip_fraction"])==flip[j]
                    assert float(r["positive_class_identity_change_fraction"])==identity[j]
                    err=abs(float(r["clean_matched_iou"])-iou[j]);max_iou_error=max(max_iou_error,err)
                    assert err<2e-6,(seed,file,j,err)
                    assert float(r["original_area"])==area[j]
                    row_checks+=1
                independent.setdefault(key,[]).extend(
                    {"image":file,"class":int(labels[j]),"area":float(area[j]),"flip":float(flip[j]),
                     "identity":float(identity[j]),"iou":float(iou[j])} for j in range(len(labels)))
    for key,rr in independent.items():
        for cohort,test in [("tiny",lambda a:a<=256),("small",lambda a:256<a<=1024),
                             ("larger",lambda a:a>1024),("area_le_1024",lambda a:a<=1024),("all",lambda a:True)]:
            group=[r for r in rr if test(r["area"])]
            original=summary["cohorts"][key][cohort]
            assert len(group)==original["GT"] and len({r["image"] for r in group})==original["images"]
            for field,expected in [("flip","owner_flip_fraction"),("identity","positive_class_identity_change_fraction"),("iou","clean_matched_iou")]:
                assert abs(np.mean([r[field] for r in group])-original[expected])<2e-6
        cells={}
        for r in rr:
            kind="tiny" if r["area"]<=256 else "larger" if r["area"]>1024 else None
            if kind:cells.setdefault((r["image"],r["class"]),{}).setdefault(kind,[]).append(r["flip"])
        by_image={}
        for (image,cl),v in cells.items():
            if "tiny" in v and "larger" in v:by_image.setdefault(image,[]).append(np.mean(v["tiny"])-np.mean(v["larger"]))
        old=summary["common_support"][key]
        assert len(by_image)==old["images"]
        assert abs(np.mean([np.mean(v) for v in by_image.values()])-old["image_class_macro_tiny_minus_larger"])<1e-12
    b=summary["cohorts"]["B_0.5"]["tiny"];m=summary["cohorts"]["M_0.5"]["tiny"]
    gap=summary["common_support"]["B_0.5"]["image_class_macro_tiny_minus_larger"]
    assert b["GT"]==1909 and b["images"]==100 and summary["common_support"]["B_0.5"]["images"]==75
    assert b["owner_flip_fraction"]<.10 and gap<.05 and b["positive_class_identity_change_fraction"]<.05
    assert summary["prespecified_screen"]["prominent_native_primary_sensitivity"] is False
    results["ECDet_S_seed20261101_train128"]={"tiny_owner_flip_percent":100*b["owner_flip_fraction"],
        "tiny_positive_class_identity_change_percent":100*b["positive_class_identity_change_fraction"],
        "tiny_minus_larger_image_class_macro_pp":100*gap,
        "NWD_tiny_owner_flip_percent":100*m["owner_flip_fraction"],
        "NWD_clean_matched_tiny_mean_iou_change":m["clean_matched_iou"]-b["clean_matched_iou"],
        "screen":"NOT_PROMOTED"}
receipt={"status":"PASS","downloaded_manifest_files_sha256_checked":checked,"local_clone_input_files_verified":local_clone_checks,"native_source_file_digests_bound":71,"permutations_checked":permutations,
         "individual_GT_owner_indices_checked":scalar_owners,"per_GT_CSV_row_checks":row_checks,
         "max_independent_source_FP32_clean_iou_error":max_iou_error,"runs":results,
         "full_model_forwards":0,"optimizer_updates":0,"new_matching_solves":0,
         "scope":"Independent raw-index reaggregation, conservation/uniqueness and source-FP32 clean IoU; not full detector rerun or efficacy verification."}
(ROOT/"remote_independent_verification.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
print(json.dumps(receipt))
