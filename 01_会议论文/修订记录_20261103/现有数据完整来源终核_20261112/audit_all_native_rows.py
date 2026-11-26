from pathlib import Path
import collections
import csv
import datetime as dt
import hashlib
import io
import json
import math
import zipfile
from lxml import etree as ET

ROOT=Path(r"C:\Users\jiwei\Desktop\小论文投稿\01_会议论文")
HERE=Path(__file__).resolve().parent
SNAP=ROOT/"实验数据汇总_20261108"/"verified_snapshots"/"NO_MSCA_round320_verified_20261111"
METRICS=["AP","AP50","AP75","APs","APm","APl","AR1","AR10","AR100","ARs","ARm","ARl"]
NS={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha(p):return sha_bytes(p.read_bytes())
def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def csv_rows(p):return list(csv.DictReader(io.StringIO(p.read_text(encoding="utf-8-sig"))))
def rec(p):return {"file":str(p),"bytes":p.stat().st_size,"sha256":sha(p)}
def plain(e):return "".join(n.text or "" for n in e.iter() if n.tag.endswith("}t"))

def main():
    HERE.mkdir(parents=True,exist_ok=True)
    manifest=SNAP/"snapshot_manifest.json"
    assert sha(manifest)=="958b7bd88b206b5ea7693bffedf5bbdde141e37eb52d17c1cb6340244eb557af"
    frozen={x["file"]:x["sha256"] for x in read(manifest)["files"]}
    paths=[SNAP/"逐轮指标.csv",SNAP/"终点信息.csv",SNAP/"终点指标.csv"]
    for p in paths:assert sha(p)==frozen[p.name]
    rows,endpoint_info,endpoint_metrics=map(csv_rows,paths)
    assert len(rows)==1280 and len(endpoint_info)==8 and len(endpoint_metrics)==96
    expected_arms={"BASE","FULL","NO_MPS","NO_MSCA"}
    assert set(r["arm"] for r in rows)==expected_arms
    cache={};normalized={};source_counts=collections.Counter();arm_counts=collections.Counter()
    for row in rows:
        path=Path(row["source_path"])
        if str(path) not in cache:
            raw=path.read_bytes();digest=sha_bytes(raw)
            if row["source_locator"]=="JSON root":
                cached={"raw":raw,"sha256":digest,"root":json.loads(raw)}
            else:
                lines=raw.splitlines();assert all(line.strip() for line in lines)
                cached={"raw":raw,"sha256":digest,"lines":lines,"rows":[json.loads(line) for line in lines]}
            cache[str(path)]=cached
        cached=cache[str(path)]
        assert cached["sha256"]==row["source_sha256"]
        index=int(row["source_record_index_0based"])
        if row["source_locator"]=="JSON root":
            assert index==0 and row["normalization"]=="remove checkpoint_sha256 only for curve-row equality"
            source=dict(cached["root"]);assert source["checkpoint_sha256"]==row["checkpoint_sha256_present_in_original_source"]
            source.pop("checkpoint_sha256");raw_record=cached["raw"]
        else:
            assert row["source_locator"]==f"line {index+1}" and row["normalization"]=="none"
            source=cached["rows"][index];raw_record=cached["lines"][index]
        assert sha_bytes(raw_record)==row["source_raw_record_sha256"]
        canonical=json.dumps(source,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
        assert sha_bytes(canonical)==row["normalized_row_sha256"]
        arm,round_num=row["arm"],int(row["round_1based"])
        # The frozen NO_MSCA export retains its ingestion-stage label, even for
        # the final appended rows. Completion is determined separately below
        # from the full retained sequence and the bound endpoint metadata.
        expected_csv_status="accepted_prefix_only" if arm=="NO_MSCA" else "completed_320_rounds"
        assert row["series_status"]==expected_csv_status and row["seed"]=="0"
        assert source["epoch"]==int(row["actual_epoch_index_0based"])==round_num-1
        for csv_key,native_key in [("last_epoch_state_counter","last_epoch"),("optimizer_step_attempts","optimizer_step_attempts"),("ema_updates","ema_updates")]:
            assert int(row[csv_key])==source[native_key]
        assert int(row["optimizer_step_attempts"])==round_num*1617
        assert float(row["train_loss_mean"])==source["train_loss_mean"] and math.isfinite(source["train_loss_mean"])
        assert set(source["metrics"])==set(METRICS)
        for name in METRICS:
            assert float(row[name])==source["metrics"][name] and math.isfinite(source["metrics"][name])
        for csv_key,native_key in [
            ("weight_source","weight_source"),("evaluated_ema_tensor_sha256","ema_tensor_sha256"),
            ("prediction_sha256","prediction_sha256"),("evaluation_arrays_sha256","evaluation_arrays_sha256"),
            ("created_utc","created_utc"),("protocol_sha256","protocol_sha256"),("configuration_sha256","config_sha256"),
        ]:assert row[csv_key]==source[native_key]
        assert row["weight_source"]=="ema.module"
        for key in ["first_executed_lrs","post_final_update_lrs"]:
            assert json.loads(row[key])==source[key]
        assert float(row["seconds"])==source["seconds"]
        assert (arm,round_num) not in normalized
        normalized[(arm,round_num)]=source
        arm_counts[arm]+=1;source_counts[row["source_id"]]+=1
    assert dict(arm_counts)=={"BASE":320,"FULL":320,"NO_MPS":320,"NO_MSCA":320}
    for arm in expected_arms:assert {r for a,r in normalized if a==arm}==set(range(1,321))
    endpoint_checks=[]
    info={(r["arm"],r["endpoint_role"]):r for r in endpoint_info}
    assert len(info)==8 and set(info)=={(a,r) for a in expected_arms for r in ["primary","secondary"]}
    for arm in sorted(expected_arms):
        best_round=max(range(1,321),key=lambda r:(normalized[(arm,r)]["metrics"]["AP"],-r))
        for role,wanted_round in [("primary",320),("secondary",best_round)]:
            entry=info[(arm,role)];round_num=int(entry["round_1based"])
            assert round_num==wanted_round and int(entry["epoch_index_0based"])==wanted_round-1
            source=normalized[(arm,round_num)]
            assert entry["completed_rounds"]=="320" and entry["seed"]=="0" and entry["validation_images"]=="548"
            assert entry["weight_source"]=="EMA"
            for key in ["prediction_sha256","evaluation_arrays_sha256"]:assert entry[key]==source[key]
            assert entry["ema_tensor_sha256"]==source["ema_tensor_sha256"]
            assert int(entry["optimizer_step_attempts_at_evaluation"])==source["optimizer_step_attempts"]
            assert int(entry["total_optimizer_step_attempts"])==320*1617 and int(entry["predictions"])==548*300
            assert entry["protocol_sha256"]==source["protocol_sha256"] and entry["configuration_sha256"]==source["config_sha256"]
            metric_items=[r for r in endpoint_metrics if r["arm"]==arm and r["endpoint_role"]==role]
            assert len(metric_items)==12 and set(r["metric"] for r in metric_items)==set(METRICS)
            for item in metric_items:
                assert int(item["round_1based"])==round_num
                assert float(item["value_fraction"])==source["metrics"][item["metric"]]
                assert abs(float(item["value_percent"])-100*float(item["value_fraction"]))<1e-12
            endpoint_checks.append({"arm":arm,"role":role,"round":round_num,"AP":source["metrics"]["AP"],"APs":source["metrics"]["APs"],"all12_metrics_and_EMA_prediction_array_binding_match":True})
    summary_path=SNAP/"实验汇总.json"
    assert sha(summary_path)==frozen[summary_path.name]
    summary=read(summary_path)
    assert any("accepted_prefix_only" in note for note in summary["source_metadata_notes"])
    completion_dir=SNAP/"source_evidence"/"completion_evidence_round320_20261111"
    completion_paths=[completion_dir/name for name in ["monitor_status.json","completed_run_receipt.json","completed_supervisor.json"]]
    for p in completion_paths:assert sha(p)==frozen[p.relative_to(SNAP).as_posix()]
    monitor,receipt,supervisor=map(read,completion_paths)
    assert monitor["status"]=="COMPLETED_AND_BACKED_UP" and monitor["epoch"]==319
    assert monitor["supervisor_sha256"]==sha(completion_paths[2])
    assert receipt["status"]==supervisor["status"]=="COMPLETED" and supervisor["exit_code"]==0
    assert supervisor["final_receipt"]==receipt
    assert receipt["completed_epochs"]==320 and receipt["actual_loop_epoch"]==319 and receipt["optimizer_step_attempts"]==517440
    assert receipt["latest_metrics"]==normalized[("NO_MSCA",320)]["metrics"]
    manuscript=ROOT/"稿件"/"MPSMA-DEIM_conference_EN_revised.docx"
    assert sha(manuscript)=="0e76fa742521cd1cdcacf9e610347af84e14186d8529816304f302db6d6dbe4c"
    with zipfile.ZipFile(manuscript) as z:doc=ET.fromstring(z.read("word/document.xml"))
    tables=[[[plain(c) for c in row.findall("w:tc",NS)] for row in table.findall("w:tr",NS)] for table in doc.findall("w:body/w:tbl",NS)]
    assert len(tables)==2 and len(tables[0])==5 and len(tables[1])==4
    for row in tables[0][1:]:
        assert len(row)==8
        role,arm,round_str,*values=row;assert round_str==info[(arm,role.lower())]["round_1based"]
        source=normalized[(arm,int(round_str))]
        for name,value in zip(["AP","AP50","AP75","APs","AR100"],values):assert value==f"{source['metrics'][name]:.4f}"
    for row in tables[1][1:]:
        assert len(row)==6
        arm,*values=row;source=normalized[(arm,320)]
        for name,value in zip(["AP","APs","AR100","ARl"],values[:4]):assert value==f"{source['metrics'][name]:.4f}"
        assert values[4]==f"{int(info[(arm,'primary')]['trainable_parameters'])/1e6:.4f}"
    report={
        "status":"ALL1280_NATIVE_ROWS_AND8_ENDPOINTS_READBACK_PASSED",
        "observed_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"manuscript":rec(manuscript),"snapshot_manifest":rec(manifest),
        "input_tables":[rec(p) for p in paths],"rows_per_arm":dict(arm_counts),"source_slice_counts":dict(source_counts),
        "source_files":[{"path":p,"sha256":v["sha256"],"bytes":len(v["raw"])} for p,v in cache.items()],
        "verified_fields":["native source-file SHA","raw JSONL-line or JSON-root SHA","normalized sorted-key compact JSON SHA","all12 metric values","train loss","epoch and update/EMA counters","weight source","EMA/prediction/evaluation-array hash fields","learning-rate vectors","protocol/configuration SHA","record time and round seconds"],
        "all_arms_have_exactly_rounds1_to320":True,"maximum_metric_readback_difference":0,
        "NO_MSCA_completion_readback":{"native_status":"COMPLETED_AND_BACKED_UP","completed_rounds":320,"supervisor_exit_code":0,"metrics_match_native_round320":True,"files":[rec(p) for p in completion_paths],"frozen_summary_with_legacy_status_note":rec(summary_path)},
        "legacy_status_metadata_note":"The frozen per-round CSV labels all320 NO_MSCA rows accepted_prefix_only. This is an inherited export-stage label, not a current completion flag. The endpoint table records completed_rounds=320 and all320 native rows are present. The frozen CSV is retained unchanged; consumers must use the current completion metadata rather than treating this legacy label as a live status.",
        "endpoint_checks":endpoint_checks,"endpoint_rule":"Primary=actual round320. Secondary=highest AP over the same320 rounds,earliest tie; all12 metrics from that row.",
        "both_manuscript_tables_match":True,
        "scope_limit":"This verifies archived row identity and numerical provenance. It does not rerun model inference/evaluators, rehash all checkpoint/prediction/array payloads, establish historical source versions beyond existing bindings, or remove design confounding. Prior saved-payload audits remain separate evidence.",
        "new_model_forwards":0,"new_evaluator_runs":0,"new_training_runs":0,
    }
    (HERE/"all_native_rows_readback.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ["status","rows_per_arm","source_slice_counts","maximum_metric_readback_difference","endpoint_checks","both_manuscript_tables_match","scope_limit"]},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
