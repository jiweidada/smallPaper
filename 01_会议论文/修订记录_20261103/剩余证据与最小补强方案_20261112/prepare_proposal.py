from pathlib import Path
import csv
import datetime as dt
import hashlib
import io
import json
import shutil
import statistics
import tarfile

ROOT = Path(r"C:\Users\jiwei\Desktop\小论文投稿\01_会议论文")
REMOTE = ROOT / "远程公平对照_20261103"
HERE = Path(__file__).resolve().parent
SNAP = ROOT / "实验数据汇总_20261108" / "verified_snapshots" / "NO_MSCA_round320_verified_20261111"
NOW = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def digest_bytes(b): return hashlib.sha256(b).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()
def read(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def rec(p): return {"file":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)}
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");return rec(p)
def differences(a,b,path=""):
    if isinstance(a,dict) and isinstance(b,dict):
        return [x for k in sorted(set(a)|set(b)) for x in differences(a.get(k),b.get(k),path+"/"+k)]
    return [] if a==b else [{"path":path,"BASE":a,"FULL":b}]

def main():
    HERE.mkdir(parents=True,exist_ok=True)
    review=HERE/"review_inputs";review.mkdir(exist_ok=True)
    protocol_path=REMOTE/"execution_protocol.json"
    assert sha(protocol_path)=="8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b"
    protocol=read(protocol_path)
    archive=REMOTE/"secondary_frozen_inputs.tar.gz"
    receipt=read(REMOTE/"secondary_frozen_inputs_receipt.json")
    assert sha(archive)==receipt["sha256"]=="e127bf14dacd999ca3794c90dc16d778f34f22ff135c227d599ad43ac243c36d"
    frozen_checks=[];contents={}
    wanted=["execution_configs/BASE.yml","execution_configs/FULL.yml","preflight/common_initialization.json","preflight/BASE.first100.json","preflight/FULL.first100.json","dataset_bundle_manifest.json"]
    with tarfile.open(archive,"r:gz") as t:
        for item in protocol["frozen_files"]:
            member=t.getmember(item["path"]);assert member.isfile()
            data=t.extractfile(member).read();assert digest_bytes(data)==item["sha256"]
            frozen_checks.append({"path":item["path"],"sha256":item["sha256"],"bytes":len(data)})
            if item["path"] in wanted:contents[item["path"]]=data
    assert len(frozen_checks)==247 and set(contents)==set(wanted)
    configs={arm:json.loads(contents[f"execution_configs/{arm}.yml"]) for arm in ["BASE","FULL"]}
    for arm in ["BASE","FULL"]:
        (review/f"{arm}.frozen.json").write_bytes(contents[f"execution_configs/{arm}.yml"])
    (review/"common_initialization.json").write_bytes(contents["preflight/common_initialization.json"])
    diff=differences(configs["BASE"],configs["FULL"])
    init={arm:REMOTE/"模块消融"/"NO_MPS"/"reference"/f"{arm}.initial.pt" for arm in ["BASE","FULL"]}
    expected={"BASE":"4fbd01639df2f19e9178bc536805649c5edb0c5221968fe73cc7f8351d642517","FULL":"20624a8d3715ec0b31c14343e8a63f29ad90b695b4e6cd1296a7b1e2b971f598"}
    for arm,p in init.items():assert sha(p)==expected[arm]
    data_archive=REMOTE/"verified_dataset.zip";data_manifest=REMOTE/"dataset_bundle_manifest.json"
    assert sha(data_archive)==protocol["cache"]["archive_sha256"]
    assert sha(data_manifest)==protocol["cache"]["manifest_sha256"]
    assert contents["dataset_bundle_manifest.json"]==data_manifest.read_bytes()
    rows=list(csv.DictReader(io.StringIO((SNAP/"逐轮指标.csv").read_text(encoding="utf-8-sig"))))
    timing={}
    for arm in ["BASE","FULL"]:
        values=[float(r["seconds"]) for r in rows if r["arm"]==arm]
        assert len(values)==320
        timing[arm]={"recorded_rounds":len(values),"sum_seconds":sum(values),"sum_hours":sum(values)/3600,"median_round_seconds":statistics.median(values),"minimum_round_seconds":min(values),"maximum_round_seconds":max(values)}
    disk={}
    for drive in [Path("C:/"),Path("D:/")]:
        usage=shutil.disk_usage(drive);disk[str(drive)]={"free_bytes":usage.free,"free_GiB":usage.free/2**30,"total_bytes":usage.total}
    # Observed per-round components support an estimate; the reservation is a separate stop gate.
    base_run=REMOTE/"断点续训_20261105"/"BASE_30166_r308"/"terminal_evidence"/"run"
    full_manifest=read(REMOTE/"断点续训_20261105"/"FULL_30848_20261106"/"backups"/"epochs"/"epoch0319"/"manifest.json")
    full_sizes={x["name"]:x["bytes"] for x in full_manifest["files"]}
    checkpoint_sizes={"BASE":(base_run/"next_epoch.pt").stat().st_size,"FULL":full_sizes["next_epoch.pt"]}
    common=protocol["randomness"]["common_initialization"]
    inputs_audit=write(HERE/"frozen_inputs_and_resource_audit.json",{
        "status":"LOCAL_INPUT_BYTE_IDENTITY_CHECKED_NOT_EXECUTION_PREFLIGHT","observed_utc":NOW,
        "protocol":rec(protocol_path),"source_archive":rec(archive),"frozen_files_verified":frozen_checks,
        "dataset_archive":rec(data_archive),"dataset_manifest":rec(data_manifest),"initial_files":{arm:rec(p) for arm,p in init.items()},
        "configuration_differences":diff,"configuration_review_copies":{arm:rec(review/f"{arm}.frozen.json") for arm in ["BASE","FULL"]},
        "common_initial_tensor_key_count":len(common["keys"]),"existing_time_measurements":timing,"local_disk_observed":disk,
        "checkpoint_size_bytes_observed":checkpoint_sizes,"new_model_forwards":0,"new_evaluations":0,"new_training_runs":0,
    })
    plan={
        "schema":"bounded_training_proposal_v1","version":"BASE_FULL_FRESH_20261112_v1","status":"PROPOSED_NOT_AUTHORIZED_NOT_STARTED","created_utc":NOW,
        "material_passport":{"origin_skill":"academic-research-suite / experiment-agent","origin_mode":"plan","verification_status":"UNVERIFIED","scope":"Design proposal; local bytes/resources checked, no run or remote preflight."},
        "research_question":"Under the same frozen recipe from the archived initial states, without extra validation-selected rescue anchors or reseeding on recovery after training has begun, does the fixed-round FULL-minus-BASE AP/APs difference persist?",
        "why_existing_results_do_not_answer_fully":["BASE joins an original 214-round prefix with later continuations; the interrupted original rounds 215-225 are excluded.","The original BASE and FULL recovery anchors lacked main-process random states, and they resumed after different rounds.","The diagnostic is a separate trajectory; its completion and source checks do not remove history confounding.","Stored weights, arrays and logs can establish provenance but cannot reconstruct the missing random states or an unobserved alternative training path."],
        "minimum_arms":[{"name":"BASE_FRESH","architecture":"original n BASE","initial_file":rec(init["BASE"]),"configuration":rec(review/"BASE.frozen.json")},{"name":"FULL_FRESH","architecture":"complete n FULL","initial_file":rec(init["FULL"]),"configuration":rec(review/"FULL.frozen.json")}],
        "why_two_arms":"Both existing main trajectories have the relevant recovery history limitation. A fresh run of only one arm would retain that confound in the reference arm.",
        "scientific_recipe":{"parent_protocol_sha256":sha(protocol_path),"dataset_archive_sha256":sha(data_archive),"dataset_manifest_sha256":sha(data_manifest),"training_images":6471,"validation_images":548,"input_size":[640,640],"seed":0,"batch_size":4,"world_size_per_arm":1,"precision":"FP32; AMP and TF32 disabled","pretrained_loading":False,"optimizer_augmentation_EMA_losses_LR_and_native_late_rollback":"Exactly the archived frozen recipe. Only data/output path relocation and evidence instrumentation may differ.","common_initial_tensor_key_count":len(common["keys"]),"data_generator_seed_per_actual_epoch":"1000003 * actual_loop_epoch","reset_after_model_construction":True},
        "budget":{"groups":2,"training_rounds_per_arm":320,"total_training_rounds":640,"updates_per_round":1617,"max_optimizer_update_attempts_per_arm":517440,"max_optimizer_update_attempts_total":1034880,"full_validation_passes_per_arm":320,"max_full_validation_passes_total":640,"images_per_validation":548,"validation_image_passes_total":350720,"preflight_per_arm":"One first-batch forward/backward without optimizer update; first100 input hashes. No separate full validation pass.","hard_timeout_seconds_per_arm":604800,"no_extra_arms_seeds_tuning_or_speed_benchmarks":True},
        "evaluation":protocol["evaluation"],
        "analysis":{"primary":"Signed FULL_FRESH-minus-BASE_FRESH AP and APs at actual round320 EMA before any following native rollback.","secondary":"Highest AP in each completed trajectory; earliest tie; all metrics from the same chosen row.","positive_gain_required_for_acceptance":False,"report_negative_or_mixed_result":True,"no_p_values_or_significance_claim_from_one_seed":True,"existing_results":"Retain rev40 and the original evidence; identify new run IDs and report the new pair separately before deciding manuscript changes."},
        "resource_expectations":{"historical_recorded_hours_by_arm":{a:x["sum_hours"] for a,x in timing.items()},"historical_total_GPU_hours":sum(x["sum_hours"] for x in timing.values()),"parallel_recorded_hours_lower_reference":max(x["sum_hours"] for x in timing.values()),"time_note":"These are historical round-time sums, not a forecast or a full wall-clock measurement. Parallel execution needs at least the slower arm's roughly59.2h reference plus transfer, backup and interruptions; plan at least3 days and allow more.","local_backup_target":"D:/MPSMA_DEIM_experiment_backups/BASE_FULL_FRESH_20261112_v1 (to be created only after authorization)","minimum_local_free_before_start_GiB":120,"local_disk_snapshot":disk,"storage_estimate":"Roughly80-90 decimal GB from observed checkpoint/prediction sizes plus late rollback states; reserve120GiB and stop before exhaustion. Deduplication and actual sizes require preflight verification.","remote_minimum":"One sufficiently free compatible24GiB GPU per arm. If using two GPUs, at most one arm per GPU. Verify persistent storage and approximately70GiB per distinct arm volume, or140GiB when both share one volume.","devices_allowed":"Only still-valid user-provided servers; read-only identity, idle-GPU and persistent-storage preflight required. No new rental or paid resource purchase.","current_remote_availability":"Not checked in this proposal turn; historical availability is not a current live observation."},
        "execution_gates":["Do not launch an archived runner that lacks full state/round backup support. Prepare and inspect a separate bounded driver after approval, preserving scientific function semantics.","Verify all resolved config values,247 frozen-file digests, all7021 dataset files, initial raw/EMA state and554 shared tensors, dependencies and first100 input hashes on each allowed server.","Two compatible idle GPUs are required for the stated parallel schedule. If only one is available, stop at preflight and report rather than silently switching to a roughly100h sequential schedule.","Store actual evaluated state before native rollback separately from next-round state; save optimizer/EMA/warmup/main RNG and relevant generator states.","After every complete round, verify manifest/ACK and local files before the next round. Preserve all old runs and artifacts.","Stop on nonfinite values, errors, memory/storage failure or hard timeout; no automatic retry, outcome-dependent rescue anchor or budget extension.","An interruption that requires restarting from a checkpoint must be reported; do not silently reseed or automatically reuse the old authorization for a replacement run.","No exact startup command is claimed ready before the isolated driver and remote preflight exist; they must conform to this bounded design before GPU launch."],
        "limits_even_if_completed":["This is still one seed and local validation, not independent-test generalization or variance evidence.","BASE and FULL differ in capacity and several backbone components; the pair measures the complete configuration.","It does not isolate an MSCA attention-only effect, supply a capacity-matched control, establish MPS-MSCA interaction or measure deployment latency."],
        "authorization_basis":"Existing BASE/FULL, diagnostic and module-removal budgets have been consumed. The user explicitly required new experiments to wait for clear authorization. This proposal does not reuse prior approval as new training permission.",
        "local_input_audit":inputs_audit,"manuscript_current_sha256":"0e76fa742521cd1cdcacf9e610347af84e14186d8529816304f302db6d6dbe4c",
    }
    plan_ref=write(HERE/"proposed_BASE_FULL_FRESH_protocol.json",plan)
    note=(
        "最小主对照补强方案 v1（待授权，未启动）\n\n"
        "## Material Passport\nOrigin Skill: academic-research-suite / experiment-agent\nOrigin Mode: plan\nOrigin Date: "+NOW+"\nVerification Status: UNVERIFIED（只完成本地输入核查，未运行新实验）\nVersion Label: BASE_FULL_FRESH_20261112_v1\n\n"
        "一、要回答的问题\n完整FULL与原始BASE在共同冻结配方、从固定初始状态开始、没有额外按验证结果挑选旧恢复锚点或在训练开始后的恢复时重新设seed0时，固定320轮末端AP/APs差异是否仍存在？初始化后设seed0和原配方预先规定的297轮后恢复政策保持一致，不因结果修改。\n\n"
        "二、现有材料为什么不够\nBASE保留轨迹包含214轮原前缀和后续恢复；原215–225轮被排除。FULL从第1轮锚点恢复，两者早期锚点缺少主进程随机状态。已完成的11轮诊断是独立轨迹，不能补齐缺失状态。原始权重、预测与评价数组可以证明结果归属，不能重建没有发生的匹配训练路径。新训练一组也不够，因为另一参考组仍保留该历史混杂。\n\n"
        "三、最少两组及固定预算\nBASE_FRESH与FULL_FRESH，从已核验的各自初始权重开始；统一seed0、batch4、FP32（关闭AMP/TF32）、640输入、6471训练图和548验证图、原冻结优化器/增强/EMA/损失/学习率/后期恢复规则。各320轮、517440次更新尝试、320次完整验证；合计最多640轮、1034880次更新尝试、640次完整验证。每组启动预检仅一次首batch前向/反向且不更新优化器，另核对首100输入batch。\n\n"
        "四、数据与结论边界\n沿用当前字节固定的本地COCO转换协议（maxDets1/10/100、top300、EMA），保留已披露的忽略区与重复记录问题，不切换官方VisDrone评价口径。主端点为实际320轮当次恢复前EMA；次端点为同一完整轨迹AP最佳且其他指标取同一行。报告正负或混合结果，不把正增益当作运行成功门槛。它补完整配置主对照，仍不隔离MSCA容量效应，不新增seed、交互组或测速。\n\n"
        "五、耗时和空间\n已有320轮日志的轮内耗时合计：BASE约40.5小时、FULL约59.2小时，共约99.7 GPU小时。双卡并行参考下限约59.2小时，另有传输/备份/停顿，应预留约3天以上；每臂仍有7天硬超时，不承诺实际完成时间。新备份粗估80–90 GB，D盘至少预留120GiB；本次D盘约495GiB可用，C盘约47GiB，不选C盘。远程按每组一张空闲兼容24GiB GPU和足够持久存储预检；若只有一张卡可用则停在预检并报告，不自动改成约100小时串行。只用仍有效的已提供服务器，不购买资源。\n\n"
        "六、启动与失败处理\n授权后在独立目录准备驱动，冻结科学计算函数并检查完整状态保存与逐轮本地ACK；不能直接重用缺少完整随机状态的旧驱动。每轮完成后先备份并核对再进入下一轮。出错、非有限值、显存/空间不足或超时则停止并保留证据；不自动换精度、改batch、重试、额外选锚点、扩大预算或增加实验组。中断后的续跑需先说明缺口，不能静默重置随机数。\n\n"
        "七、当前状态\n协议、两份初始权重、247份冻结文件、1.74GB数据归档及数据清单的哈希均已在本地核对。两份原冻结配置已复制到review_inputs供审阅。当前没有新训练入口或远程预检通过的声明；已有授权预算已完成，本方案等待新的明确授权。rev40稿件及现有实验数据不因提案改变。\n\n"
        "机器可读协议：proposed_BASE_FULL_FRESH_protocol.json\nSHA256："+plan_ref["sha256"]+"\n本地核查：frozen_inputs_and_resource_audit.json\n"
    )
    (HERE/"最小主对照补强方案_待授权.txt").write_text(note,encoding="utf-8")
    print(json.dumps({"status":plan["status"],"protocol":plan_ref,"inputs_audit":inputs_audit,"frozen_files_verified":len(frozen_checks),"common_initial_tensor_key_count":len(common["keys"]),"config_differences":diff,"time_hours":{a:x["sum_hours"] for a,x in timing.items()},"D_free_GiB":disk["D:\\"]["free_GiB"]},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
